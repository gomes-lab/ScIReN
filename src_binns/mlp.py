import torch
import torch.nn as nn
import torch.nn.functional as F
from lipmlp import lipmlp
from fun_matrix_clm5_vectorized import fun_model_simu
from fun_matrix_clm5_vectorized_prediction import fun_model_prediction
from pe_gcn_model import GCN, PEGCN, GridCellSpatialRelationEncoder

class mlp(torch.nn.Module):
	"""
	New MLP from this repo: https://github.com/whitneychiu/lipmlp_pytorch/blob/main/models/mlp.py
	"""
	def __init__(self, dims, use_bn=False, dropout_prob=0.0, leaky_relu=False):
		"""
		dim[0]: input dim
		dim[1:-1]: hidden dims
		dim[-1]: out dim

		assume len(dims) >= 3
		"""
		super().__init__()

		self.layers = torch.nn.ModuleList()
		self.use_bn = use_bn
		self.dropout_prob = dropout_prob
		if use_bn:
			self.bns = torch.nn.ModuleList()
		if dropout_prob > 0:
			self.dropout = nn.Dropout(self.dropout_prob)

		for ii in range(len(dims)-2):
			self.layers.append(torch.nn.Linear(dims[ii], dims[ii+1]))
			if use_bn:
				self.bns.append(torch.nn.BatchNorm1d(dims[ii+1]))
		self.layer_output = torch.nn.Linear(dims[-2], dims[-1])
		if leaky_relu:
			self.relu = nn.LeakyReLU(negative_slope=0.3)
		else:
			self.relu = torch.nn.ReLU()

		# Power iteration for spectral norm
		self.sr_u = {}
		self.sr_v = {}
		self.num_power_iter = 4


	def forward(self, x):
		for ii in range(len(self.layers)):
			x = self.layers[ii](x)
			if self.use_bn:
				x = self.bns[ii](x)
			x = self.relu(x)
			if self.dropout_prob > 0:
				x = self.dropout(x)

		return self.layer_output(x)

	def spectral_norm_parallel(self, device):
		"""Code from https://github.com/NVlabs/NVAE/blob/master/model.py
			
		This method computes spectral normalization for all conv layers in parallel. This method should be called
		 after calling the forward method of all the conv layers in each iteration. """

		weights = {}   # a dictionary indexed by the shape of weights
		for ii in range(len(self.layers)):
			weight = self.layers[ii].weight
			weight_mat = weight.view(weight.size(0), -1)

			# Modify by batchnorm
			if self.use_bn:
				weight_mat = weight_mat * (self.bns[ii].weight.unsqueeze(1) / torch.sqrt(self.bns[ii].running_var.unsqueeze(1)))
			if weight_mat.shape not in weights:
				weights[weight_mat.shape] = []

			weights[weight_mat.shape].append(weight_mat)

		# record the output layer separately, as it's not listed in "layers"
		weight = self.layer_output.weight
		weight_mat = weight.view(weight.size(0), -1)
		if weight_mat.shape not in weights:
			weights[weight_mat.shape] = []
		weights[weight_mat.shape].append(weight_mat)

		loss = 0
		for i in weights:
			weights[i] = torch.stack(weights[i], dim=0)
			with torch.no_grad():
				num_iter = self.num_power_iter
				if i not in self.sr_u:
					num_w, row, col = weights[i].shape
					self.sr_u[i] = F.normalize(torch.ones(num_w, row).normal_(0, 1).to(device), dim=1, eps=1e-3)
					self.sr_v[i] = F.normalize(torch.ones(num_w, col).normal_(0, 1).to(device), dim=1, eps=1e-3)
					# increase the number of iterations for the first time
					num_iter = 10 * self.num_power_iter

				for j in range(num_iter):
					# Spectral norm of weight equals to `u^T W v`, where `u` and `v`
					# are the first left and right singular vectors.
					# This power iteration produces approximations of `u` and `v`.
					self.sr_v[i] = F.normalize(torch.matmul(self.sr_u[i].unsqueeze(1), weights[i]).squeeze(1),
											   dim=1, eps=1e-3)  # bx1xr * bxrxc --> bx1xc --> bxc
					self.sr_u[i] = F.normalize(torch.matmul(weights[i], self.sr_v[i].unsqueeze(2)).squeeze(2),
											   dim=1, eps=1e-3)  # bxrxc * bxcx1 --> bxrx1  --> bxr

			sigma = torch.matmul(self.sr_u[i].unsqueeze(1), torch.matmul(weights[i], self.sr_v[i].unsqueeze(2)))
			loss += torch.sum(sigma)
		return loss

	# def batchnorm_loss(self):
	# 	loss = 0
	# 	for l in self.bns:
	# 		if l.affine:
	# 			loss += torch.max(torch.abs(l.weight))
	# 	return loss


#---------------------------------------------------
# Wrapper for MLP and LipMLP from this repo: https://github.com/whitneychiu/lipmlp_pytorch/blob/main/models/mlp.py
#---------------------------------------------------
# define model
class mlp_wrapper(nn.Module):
	def __init__(self, input_vars, var_idx_to_emb, vertical_mixing, pos_enc,
			     lipschitz=False, one_hot=False, use_bn=False, dropout_prob=0.0, 
				 leaky_relu=False, rep_grad=False,
				 losses=["l1", "param_reg"], device="cpu"):
		super().__init__()

		# If one_hot is True, this is a Dict from categorical variable index to number of categories.
		# If one_hot is False, this is a Dict from categorical variable index -> Embedding layer we use
		self.one_hot = one_hot
		self.var_idx_to_emb = var_idx_to_emb
		self.vertical_mixing = vertical_mixing
		self.pos_enc = pos_enc

		# List of non-categorical variable indices
		self.non_categorical_indices = list(set(list(range(input_vars))).difference(var_idx_to_emb.keys()))
		self.new_input_size = len(self.non_categorical_indices)
		for idx, emb in self.var_idx_to_emb.items():
			if self.one_hot:
				self.new_input_size += emb
			else:
				self.new_input_size += emb.embedding_dim

		# Number of parameters
		if self.vertical_mixing == 'simple_two_intercepts':
			self.num_params = 22
		else:
			self.num_params = 21

		# Spatial Encoder from PE-GNN
		if pos_enc != "none":
			self.spatial_encoder = GridCellSpatialRelationEncoder(
				spa_embed_dim=self.num_params,
				coord_dim=2, # Longitude and latitude
				frequency_num=16, 
				max_radius=360,
				min_radius=1e-06,
				freq_init="geometric",
				ffn=True # Enable feedforward network for final spatial embeddings
			)
		if pos_enc == "early":
			self.new_input_size += self.num_params  # Add the spatial embeddings

		# MLP backbone
		if lipschitz:
			self.mlp = lipmlp((self.new_input_size, 256, 256, self.num_params),
					          use_bn=use_bn, dropout_prob=dropout_prob)  # TODO different initialization methods, leaky relu, dropout, etc. not supported
		else:
			self.mlp = mlp((self.new_input_size, 256, 256, self.num_params),
				  			use_bn=use_bn, dropout_prob=dropout_prob)

		# sigmoid parameter
		self.temp_sigmoid = nn.Parameter(torch.tensor(0.0), requires_grad=True)
		self.sigmoid = nn.Sigmoid()

		# LibMTL specific
		self.rep_grad = rep_grad  # Whether to compute gradients w.r.t. parameters as well
		self.task_name = losses
		self.task_num = len(self.task_name)
		self.device = device


	def forward(self, input_var, wosis_depth, coords, whether_predict,
			    return_spatial_embedding=False):
		predictor = input_var[:, :, 0, 0]
		forcing = input_var[:, :, :, :]
		obs_depth = wosis_depth
		coords = coords.unsqueeze(1).detach().cpu().numpy()  # coords should be a numpy array

		# Compute embeddings for all categorical variables
		embs = []
		for idx, embedding_layer in self.var_idx_to_emb.items():
			if self.one_hot:
				emb = 0.1*F.one_hot(predictor[:, idx].long(), num_classes=embedding_layer)  # if one_hot, "embedding_layer" is simply the number of classes
			else:
				emb = embedding_layer(predictor[:, idx].int())
				emb = F.normalize(emb, p=2, dim=1)  # New @joshuafan: normalize embeddings
			embs.append(emb)
		all_embs = torch.concatenate(embs, dim=1)
		new_input = torch.concatenate([predictor[:, self.non_categorical_indices], all_embs], dim=1)

		# Spatial Encoding
		if self.pos_enc == "early":
			spatial_embeddings = self.spatial_encoder(coords).squeeze(1) # Remove the channel dimension. [batch, params]
			new_input = torch.concatenate([spatial_embeddings, new_input], dim=1)

		# check if new_input is nan
		if torch.isnan(new_input).any() or torch.isinf(new_input).any():
			print("new_input was nan", new_input)
			exit(1)

		# Pass through MLP
		mlp_output = self.mlp(new_input)

		# check if mlp output is nan
		if torch.isnan(mlp_output).any() or torch.isinf(mlp_output).any():
			print("mlp_output was nan", mlp_output)
			exit(1)

		# Clamp temp_sigmoid to be between 10 and 200
		clamped_temp_sigmoid = 10 + 99 * self.sigmoid(self.temp_sigmoid) # try with a smaller range

		# Positional encoder correction (if using)
		if self.pos_enc == "late":
			spatial_embeddings = self.spatial_encoder(coords).squeeze(1) # Remove the channel dimension. [batch, params]
			mlp_output += spatial_embeddings

		# Pass parameters through sigmoid to constrain their range
		h5 = self.sigmoid(mlp_output / clamped_temp_sigmoid)
	
		# check if h5 is nan
		if torch.isnan(h5).any() or torch.isinf(h5).any():
			print("h5 was nan", h5)
			exit(1) 

		# CLM5 process-based model
		if whether_predict == 1:
			simu_soc = fun_model_prediction(h5, forcing, self.vertical_mixing)
		else:
			simu_soc = fun_model_simu(h5, forcing, obs_depth, self.vertical_mixing)

		if return_spatial_embedding:
			return simu_soc, h5, spatial_embeddings
		else:
			return simu_soc, h5		


	def partial_forward(self, new_input, input_var, wosis_depth):
		"""
		Helper function which skips the embedding layer and directly passes
		`new_input` through the MLP and process_based model. We only need
		this to use curvature regularization on the MLP."""
		forcing = input_var[:, :, :, :]
		obs_depth = wosis_depth

		# Pass through MLP
		mlp_output = self.mlp(new_input)

		# check if mlp output is nan
		if torch.isnan(mlp_output).any() or torch.isinf(mlp_output).any():
			print("mlp_output was nan", mlp_output)
			exit(1)

		# Clamp temp_sigmoid to be between 10 and 100
		clamped_temp_sigmoid = 10 + 90 * torch.sigmoid(self.temp_sigmoid) # constrain the temp_sigmoid between 10 and 100
		h5 = torch.sigmoid(mlp_output / clamped_temp_sigmoid)

		# check if h5 is nan
		if torch.isnan(h5).any() or torch.isinf(h5).any():
			print("h5 was nan", h5)
			exit(1) 

		# CLM5 process-based model
		simu_soc = fun_model_simu(h5, forcing, obs_depth, self.vertical_mixing)
		return simu_soc, h5


	"""
	Functions to be compatible with the LibMTL API
	"""
	def get_share_params(self):
		r"""Return the shared parameters of the model.
		"""
		all_params = list(self.mlp.parameters())
		for idx, emb in self.var_idx_to_emb.items():
			all_params.extend(list(emb.parameters()))
		return all_params


	def zero_grad_share_params(self):
		r"""Set gradients of the shared parameters to zero.
		"""
		self.mlp.zero_grad(set_to_none=False)
		for idx, emb in self.var_idx_to_emb.items():
			emb.zero_grad(set_to_none=False)


#---------------------------------------------------
# Use PEGCN as encoder for BINN model
#---------------------------------------------------
class GCN_BINN(nn.Module):
	def __init__(self, input_vars, var_idx_to_emb, vertical_mixing, pos_enc,
			  	 k=20, one_hot=False, rep_grad=False,
				 losses=["l1", "param_reg"], device="cpu"):
		super().__init__()

		# If one_hot is True, this is a Dict from categorical variable index to number of categories.
		# If one_hot is False, this is a Dict from categorical variable index -> Embedding layer we use
		self.one_hot = one_hot
		self.var_idx_to_emb = var_idx_to_emb
		self.vertical_mixing = vertical_mixing
		self.pos_enc = pos_enc
		self.k = k

		# List of non-categorical variable indices
		self.non_categorical_indices = list(set(list(range(input_vars))).difference(var_idx_to_emb.keys()))
		self.new_input_size = len(self.non_categorical_indices)
		for idx, emb in self.var_idx_to_emb.items():
			if self.one_hot:
				self.new_input_size += emb
			else:
				self.new_input_size += emb.embedding_dim

		# Number of parameters
		if self.vertical_mixing == 'simple_two_intercepts':
			self.num_params = 22
		else:
			self.num_params = 21

		# Spatial Encoder from PE-GNN
		if pos_enc == "late":
			self.spatial_encoder = GridCellSpatialRelationEncoder(
				spa_embed_dim=self.num_params,
				coord_dim=2, # Longitude and latitude
				frequency_num=16, 
				max_radius=360,
				min_radius=1e-06,
				freq_init="geometric",
				ffn=True # Enable feedforward network for final spatial embeddings
			)
		if pos_enc == "late" or pos_enc == "none":
			self.gnn = GCN(num_features_in=self.new_input_size,
				 	       num_features_out=self.num_params, k=k, MAT=False)
		else:
			# GCN backbone. TODO - maybe allow mlp before or afterwards?
			# TODO what is decoder in PEGCN?
			self.gnn = PEGCN(num_features_in=self.new_input_size,
						     num_features_out=self.num_params, k=k, MAT=False)

		# sigmoid parameter
		self.temp_sigmoid = nn.Parameter(torch.tensor(0.0), requires_grad=True)
		self.sigmoid = nn.Sigmoid()

		# LibMTL specific
		self.rep_grad = rep_grad  # Whether to compute gradients w.r.t. parameters as well
		self.task_name = losses
		self.task_num = len(self.task_name)
		self.device = device


	def forward(self, input_var, wosis_depth, coords, 
			 	whether_predict, ei=None, ew=None,
				return_spatial_embedding=False):
		predictor = input_var[:, :, 0, 0]
		forcing = input_var[:, :, :, :]
		obs_depth = wosis_depth

		# Compute embeddings for all categorical variables
		embs = []
		for idx, embedding_layer in self.var_idx_to_emb.items():
			if self.one_hot:
				emb = 0.1*F.one_hot(predictor[:, idx].long(), num_classes=embedding_layer)  # if one_hot, "embedding_layer" is simply the number of classes
			else:
				emb = embedding_layer(predictor[:, idx].int())
				emb = F.normalize(emb, p=2, dim=1)  # New @joshuafan: normalize embeddings
			embs.append(emb)
		all_embs = torch.concatenate(embs, dim=1)
		new_input = torch.concatenate([predictor[:, self.non_categorical_indices], all_embs], dim=1)

		# Pass ode feature vectors to GNN. If "ei" and "ew" are None, constructs
		# the k-nearest neighbor graph from this batch using "coords"
		gnn_output = self.gnn(new_input, coords, ei, ew)

		# Clamp temp_sigmoid to be between 10 and 200
		clamped_temp_sigmoid = 10 + 99 * self.sigmoid(self.temp_sigmoid) # try with a smaller range

		# Positional encoder correction (if using)
		if self.pos_enc == "late":
			coords = coords.unsqueeze(1).detach().cpu().numpy()  # coords should be a numpy array if passing directly to GridCellSpatialRelationEncoder
			spatial_embeddings = self.spatial_encoder(coords).squeeze(1) # Remove the channel dimension. [batch, params]
			gnn_output += spatial_embeddings

		# Pass parameters through sigmoid to constrain their range
		h5 = self.sigmoid(gnn_output / clamped_temp_sigmoid)
	
		# check if h5 is nan
		if torch.isnan(h5).any() or torch.isinf(h5).any():
			print("h5 was nan", h5)
			exit(1) 

		# CLM5 process-based model
		if whether_predict == 1:
			simu_soc = fun_model_prediction(h5, forcing, self.vertical_mixing)
		else:
			simu_soc = fun_model_simu(h5, forcing, obs_depth, self.vertical_mixing)

		if return_spatial_embedding:
			assert self.pos_enc == "late", "Can only return spatial embedding is pos_enc is 'late'"
			return simu_soc, h5, spatial_embeddings
		else:
			return simu_soc, h5


	def partial_forward(self, new_input, input_var, wosis_depth):
		"""
		Helper function which skips the embedding layer and directly passes
		`new_input` through the MLP and process_based model. We only need
		this to use curvature regularization on the MLP."""
		forcing = input_var[:, :, :, :]
		obs_depth = wosis_depth

		# Pass through MLP
		mlp_output = self.gnn(new_input)

		# check if mlp output is nan
		if torch.isnan(mlp_output).any() or torch.isinf(mlp_output).any():
			print("mlp_output was nan", mlp_output)
			exit(1)

		# Clamp temp_sigmoid to be between 10 and 100
		clamped_temp_sigmoid = 10 + 90 * torch.sigmoid(self.temp_sigmoid) # constrain the temp_sigmoid between 10 and 100
		h5 = torch.sigmoid(mlp_output / clamped_temp_sigmoid)

		# check if h5 is nan
		if torch.isnan(h5).any() or torch.isinf(h5).any():
			print("h5 was nan", h5)
			exit(1) 

		# CLM5 process-based model
		simu_soc = fun_model_simu(h5, forcing, obs_depth, self.vertical_mixing)
		return simu_soc, h5