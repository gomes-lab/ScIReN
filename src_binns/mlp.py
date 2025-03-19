import fun_matrix_clm5_experimental
import torch
import torch.nn as nn
import torch.nn.functional as F
from lipmlp import lipmlp
from fun_matrix_clm5_vectorized import fun_model_simu, fun_model_prediction
from pe_gcn_model import GCN, PEGCN, GridCellSpatialRelationEncoder, SpatialSmoother
from torch_geometric.nn import GCNConv, GATConv, SimpleConv, knn_graph
from torch_geometric.utils import get_laplacian, to_dense_adj, to_torch_coo_tensor
import misc_utils
from spatial_utils import *
import visualization_utils

class mlp(torch.nn.Module):
	"""
	New MLP from this repo: https://github.com/whitneychiu/lipmlp_pytorch/blob/main/models/mlp.py
	"""
	def __init__(self, dims, use_bn=False, dropout_prob=0.0, activation='relu', init='xavier_uniform', residual=False):
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
		self.residual = residual
		if use_bn:
			self.bns = torch.nn.ModuleList()
		if dropout_prob > 0:
			self.dropout = nn.Dropout(self.dropout_prob)

		for ii in range(len(dims)-2):
			self.layers.append(torch.nn.Linear(dims[ii], dims[ii+1]))

			if use_bn:
				self.bns.append(torch.nn.BatchNorm1d(dims[ii+1]))
		self.layer_output = torch.nn.Linear(dims[-2], dims[-1])
		self.act = misc_utils.get_activation(activation)

		# Initialize linear layers
		if init == "xavier_uniform":
			if activation == 'leaky_relu':
				gain_activation = nn.init.calculate_gain(activation, 0.3)
			else:
				gain_activation = nn.init.calculate_gain(activation)
			gain_sigmoid = nn.init.calculate_gain('sigmoid')
			for layer in self.layers:
				nn.init.xavier_uniform_(layer.weight, gain=gain_activation)
				nn.init.zeros_(layer.bias)
			nn.init.xavier_uniform_(self.layer_output.weight, gain=gain_sigmoid)
			nn.init.zeros_(self.layer_output.bias)
		elif init == "kaiming_uniform":
			for layer in self.layers + [self.layer_output]:
				if activation == 'leaky_relu':
					nn.init.kaiming_uniform_(layer.weight, nonlinearity=activation, a=0.3)
				else:
					nn.init.kaiming_uniform_(layer.weight, nonlinearity=activation)
			nn.init.zeros_(self.layer_output.bias)
		elif init == 'default':
			pass
		else:
			raise NotImplementedError("Unsupported init")

		# Power iteration for spectral norm
		self.sr_u = {}
		self.sr_v = {}
		self.num_power_iter = 4


	def forward(self, x):
		for ii in range(len(self.layers)):
			old_x = x		
			x = self.layers[ii](x)
			if self.residual and old_x.shape == x.shape:  # Residual connection. Only around linear, TODO maybe it should wrap aroudn linear/activaton/linear
				x += old_x

			if self.use_bn:
				x = self.bns[ii](x)
			x = self.act(x)

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

			# Modify by batchnorm?
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
	def __init__(self, input_vars, var_idx_to_emb, vertical_mixing, vectorized='yes', pos_enc='early',
				 base_model="new_mlp", one_hot=False, use_bn=False, dropout_prob=0.0,
				 activation='relu', param_constraint='sigmoid', rep_grad=False,
				 losses=["l1", "param_reg"], device="cpu", train_x=None,
				 min_temp=10, max_temp=109, init="xavier_uniform", width=128,
				 para_index=None, feature_dropout=0, num_layers=4, residual=False):
		"""
		var_idx_to_emb is a dictionary mapping from categorical variable index to either
		(1) Embedding layer (if one_hot is False)
		(2) Number of categories (if one_hot is True)

		If train_x is provided, use this to rescale the input. Specifically, compute mean/std
		for non-categorical variables as train_x[:, self.non_categorical_indices, 0, 0].mean(dim=0).

		para_index is a list of parameter indices that should be predicted by neural network. If None,
		neural network predicts all parameters (as usual).
		"""
		super().__init__()

		self.one_hot = one_hot
		self.var_idx_to_emb = var_idx_to_emb
		self.vertical_mixing = vertical_mixing
		self.vectorized = vectorized
		self.pos_enc = pos_enc
		self.base_model = base_model

		# List of non-categorical variable indices.
		self.input_vars = input_vars
		self.non_categorical_indices = list(set(list(range(input_vars))).difference(var_idx_to_emb.keys()))
		self.new_input_size = len(self.non_categorical_indices)

		# Setup categorical encoding
		if self.one_hot:
			# If one-hot, just calculate the input size (after one-hot encoding)
			for _, num_classes in self.var_idx_to_emb.items():
				self.new_input_size += num_classes
		else:
			# If using Embedding layers, create ModuleDict so that all 
			# Embedding layers in var_idx_to_emb are registered as parameters
			# Note: ModuleDict requires string keys, so convert index to string
			self.var_idx_to_emb = nn.ModuleDict({str(idx): emb for idx, emb in self.var_idx_to_emb.items()})
			for _, emb in self.var_idx_to_emb.items():
				self.new_input_size += emb.embedding_dim

		# Parameter indices that NN will predict (other parameters will be directly passed
		# as "PRODA para")
		if para_index is None:
			if self.vertical_mixing == 'simple_two_intercepts':
				self.para_index = np.arange(22)
			else:
				self.para_index = np.arange(21)
		else:
			self.para_index = para_index
		self.num_params = len(self.para_index)

		# Standardize input to (mean 0, std 1) if desired
		if train_x is not None:
			train_features = train_x[:, self.non_categorical_indices, 0, 0]
			self.input_mean = train_features.mean(dim=0, keepdim=True)
			self.input_std = train_features.std(dim=0, keepdim=True)
		else:
			self.input_mean = None
			self.input_std = None

		# Spatial Encoder from PE-GNN
		if pos_enc == "early":
			self.spatial_encoder = GridCellSpatialRelationEncoder(
				spa_embed_dim=128,
				coord_dim=2, # Longitude and latitude
				frequency_num=16,
				max_radius=360,
				min_radius=1e-06,
				freq_init="geometric",
				ffn=True,  # Enable feedforward network for final spatial embeddings
			)
			self.new_input_size += 128  # Add the spatial embeddings
		elif pos_enc == "late":
			self.spatial_encoder = GridCellSpatialRelationEncoder(
				spa_embed_dim=self.num_params,
				coord_dim=2, # Longitude and latitude
				frequency_num=16,
				max_radius=360,
				min_radius=1e-06,
				freq_init="geometric",
				ffn=True,  # Enable feedforward network for final spatial embeddings
			)
			self.new_input_size += self.num_params  # Add the spatial embeddings

		# Neural network: mapping input features to biogeochemical parameters
		layer_sizes = [self.new_input_size] + [width] * (num_layers-1) + [self.num_params] 
		if base_model == "lipmlp":
			# EXPERIMENTAL: LipMLP (Lipschitz-regularized neural network)
			if residual: raise NotImplementedError("residual network not implemented for lipmlp")
			self.mlp = lipmlp(layer_sizes, use_bn=use_bn, dropout_prob=dropout_prob)  # TODO different initialization methods, activations, dropout, etc. not supported
		elif base_model == "senn":
			# EXPERIMENTAL: SENN (Self-Explaining Neural Network, like a
			# linear model but coefficients also depend on the data)
			assert self.one_hot, "SENN makes most sense with one-hot encodings"
			self.mlp = SENN(num_inputs=self.new_input_size, num_outputs=self.num_params,
							num_hidden=width, num_layers=num_layers,
				   			use_bn=use_bn, dropout_prob=dropout_prob, activation=activation, init=init, residual=residual)
		elif base_model in ["nam", "nam_joint"]:
			# EXPERIMENTAL: Neural Additive Models
			assert pos_enc in ["none", "late"], "With NAM, positional embedding size (if it exists) should equal the number of outputs"
			assert not self.one_hot, "With NAM, you should use `--categorical embedding --embed_dim NUM_PARAMS`"
			if len(self.var_idx_to_emb) > 0:
				assert next(iter(self.var_idx_to_emb.values())).embedding_dim == self.num_params, "With NAM, categorical embedding dim should equal the number of outputs"			

			# Process activation
			from nam_models import ExULayer, ReLULayer, MultiOutputNAM, MultiOutputJointNAM
			if activation == 'exu':
				shallow_layer = ExULayer
				shallow_units = width  # Section 3 of https://arxiv.org/pdf/2004.13912
				hidden_units = ()
			elif activation == 'relu':
				shallow_layer = ReLULayer
				shallow_units = width
				hidden_units = (width, width)
			else:
				raise ValueError("For NAM, activation must be exu or relu")
			
			if base_model == "nam":
				self.mlp = MultiOutputNAM(input_size=len(self.non_categorical_indices), shallow_units=shallow_units,
								    	  hidden_units=hidden_units, shallow_layer=shallow_layer,
										  feature_dropout=feature_dropout, hidden_dropout=dropout_prob, n_outputs=self.num_params)
			elif base_model == "nam_joint":
				self.mlp = MultiOutputJointNAM(input_size=len(self.non_categorical_indices), shallow_units=shallow_units,
								    	       hidden_units=hidden_units, shallow_layer=shallow_layer,
										       feature_dropout=feature_dropout, hidden_dropout=dropout_prob, n_outputs=self.num_params)
		elif base_model == "nag":
			raise NotImplementedError()
		elif base_model == "new_mlp":
			# Basic MLP
			self.mlp = mlp(layer_sizes, use_bn=use_bn, dropout_prob=dropout_prob,
				  		   activation=activation, init=init, residual=residual)
		elif base_model == "kan":
			import kan
			self.mlp = kan.KAN(width=layer_sizes, grid=3, k=3, seed=42, device=device)
			# self.mlp.speed()  # Disable symbolic branch
		else:
			raise ValueError("Unsupported base_model")

		# Parameter constraint
		self.sigmoid = misc_utils.get_param_constraint(param_constraint)

		# If using sigmoid: the "temperature" we divide by before the sigmoid
		self.temp_sigmoid = nn.Parameter(torch.tensor(0.0), requires_grad=True)
		self.min_temp = min_temp
		self.max_temp = max_temp

		# LibMTL specific
		self.rep_grad = rep_grad  # Whether to compute gradients w.r.t. parameters as well
		self.task_name = losses
		self.task_num = len(self.task_name)
		self.device = device


	def forward(self, input_var, wosis_depth, coords, whether_predict, PRODA_para=None,
			 	return_spatial_embedding=False, one_param_only=False, ignore_input=False):

		predictor = input_var[:, 0:self.input_vars, 0, 0]
		forcing = input_var[:, :, :, :]
		obs_depth = wosis_depth

		# For some reason spatial encoder requires coords to have shape [batch, 1, 2] 
		coords = coords.unsqueeze(1).detach()

		# Preprocess numeric (non-categorical) features
		features = predictor[:, self.non_categorical_indices]  # [batch, n_features]
		if self.input_mean is not None and self.input_std is not None:
			features = (features - self.input_mean) / self.input_std
		
		if torch.isnan(features).any():
			print("Features nan")
			print(features)

		# Compute embeddings for all categorical variables
		if len(self.var_idx_to_emb) >= 1:
			embs = []
			for idx, embedding_layer in self.var_idx_to_emb.items():
				idx = int(idx)
				if self.one_hot:
					emb = F.one_hot(predictor[:, idx].long(), num_classes=embedding_layer)  # if one_hot, "embedding_layer" is simply the number of classes
				else:
					# NOTE (2025-03-05): Not normalizing embeddings anymore.
					emb = embedding_layer(predictor[:, idx].int())
				embs.append(emb)

				if torch.isnan(emb).any():
					print("Emb nan", idx)
					print(emb.data)
			all_embs = torch.concatenate(embs, dim=1)

			# Combine numeric features and categorical embeddings
			new_input = torch.concatenate([features, all_embs], dim=1)
		else:
			new_input = features

		# Spatial Encoding
		if self.pos_enc == "early":
			spatial_embeddings = self.spatial_encoder(coords).squeeze(1)  # Remove the channel dimension. [batch, params]
			new_input = torch.concatenate([spatial_embeddings, new_input], dim=1)

			if torch.isnan(spatial_embeddings).any():
				print("Spatial emb nan")
				print(spatial_embeddings)

		# check if new_input is nan
		if torch.isnan(new_input).any() or torch.isinf(new_input).any():
			print("new_input was nan", new_input)
			exit(1)

		# Pass through MLP to obtain (unconstrained) biogeochemical parameters
		self.new_input = new_input
		mlp_output = self.mlp(new_input)

		# check if mlp output is nan
		if torch.isnan(mlp_output).any() or torch.isinf(mlp_output).any():
			print("mlp_output was nan", mlp_output)
			exit(1)

		# Clamp temp_sigmoid to be within a range
		clamped_temp_sigmoid = self.min_temp + (self.max_temp - self.min_temp) * F.sigmoid(self.temp_sigmoid)  # 10 + 90*self.sigmoid(self.temp_sigmoid)   #10 + 99 * self.sigmoid(self.temp_sigmoid) # try with a smaller range

		# Positional encoder correction (if using)
		if self.pos_enc == "late":
			spatial_embeddings = self.spatial_encoder(coords).squeeze(1)  # Remove the channel dimension. [batch, params]
			mlp_output += spatial_embeddings

		# Pass biogeochemical parameters through sigmoid, constraining them between [0, 1]
		self.unconstrained_params = mlp_output / clamped_temp_sigmoid
		constrained_params = self.sigmoid(self.unconstrained_params)
		if PRODA_para is None or len(self.para_index) == constrained_params.shape[1]:  # If we are predicting all params, don't need to copy PRODA_para
			 # If PRODA parameters not provided, neural network must output all params
			if self.vertical_mixing == 'simple_two_intercepts':
				assert np.array_equal(self.para_index, np.arange(22))
			else:
				assert np.array_equal(self.para_index, np.arange(21))
			predicted_para = constrained_params
		else:
			# Initialize predicted parameters to PRODA parameters; then overwrite some with NN predictions
			predicted_para = PRODA_para.detach().clone()
			predicted_para[:, self.para_index] = constrained_params

		if predicted_para.requires_grad:
			predicted_para.retain_grad()

		if one_param_only:
			# EXPERIENTAL: Choose one parameter to update, stop gradient w.r.t. other params
			import random
			param_idx = random.randint(0, predicted_para.shape[1])
			prev = predicted_para[:, :param_idx].detach()
			param_vals = predicted_para[:, param_idx:param_idx+1]
			next = predicted_para[:, param_idx:].detach()
			predicted_para = torch.cat([prev, param_vals, next], dim=1)

		# Ignore examples where predicted_para was nan. This should only happen when PRODA_para
		# contains nan values.
		predicted_para.requires_grad_ =True
		self.predicted_para = predicted_para
		valid_mask = ~torch.any((torch.isnan(predicted_para) | torch.isinf(predicted_para)), dim=1)
		if torch.sum(~valid_mask) > 0:
			print("predicted_para had nan")

		# CLM5 process-based model
		if whether_predict == 1:
			# simu_soc = fun_model_prediction(predicted_para[valid_mask], forcing, self.vertical_mixing, self.vectorized)
			simu_soc = fun_matrix_clm5_experimental.fun_model_prediction(predicted_para[valid_mask], forcing, self.vertical_mixing, self.vectorized)
		else:
			#simu_soc = fun_model_simu(predicted_para[valid_mask], forcing, obs_depth, self.vertical_mixing, self.vectorized)
			simu_soc = fun_matrix_clm5_experimental.fun_model_simu(predicted_para[valid_mask], forcing, obs_depth, self.vertical_mixing, self.vectorized)
			# print("=================================")
			# print("Depths", obs_depth[0:5, 0:10])
			# print("SIMU SOC OLD", simu_soc[0:5, 0:10])
			# print("SIMU SOC NEW", simu_soc_experimental[0:5, 0:10])

			# unequal_idx = torch.nonzero((simu_soc_experimental - simu_soc).abs() > 1e-5)
			# if unequal_idx.shape[0] > 0:
			# 	print("Unequal idx", unequal_idx)
			# 	print("SIMU SOC OLD", simu_soc[unequal_idx[0, 0], 0:15])
			# 	print("SIMU SOC NEW", simu_soc_experimental[unequal_idx[0, 0], 0:15])
			# assert torch.allclose(simu_soc, simu_soc_experimental, atol=1e-5, equal_nan=True)
		simu_soc_with_nan = torch.full((predicted_para.shape[0], simu_soc.shape[1]), float('nan'), device=input_var.device)
		simu_soc_with_nan[valid_mask] = simu_soc

		if return_spatial_embedding:
			return simu_soc_with_nan, predicted_para, spatial_embeddings
		else:
			return simu_soc_with_nan, predicted_para


	def forward_ignoring_input(self, input_var, wosis_depth):
		"""
		Use same parameter set for all sites (determined by layer_output's bias).
		"""
		predictor = input_var[:, 0:self.input_vars, 0, 0]
		forcing = input_var[:, :, :, :]
		obs_depth = wosis_depth

		# Use "layer_output.bias" as globally-fixed parameter set
		clamped_temp_sigmoid = self.min_temp + (self.max_temp - self.min_temp) * F.sigmoid(self.temp_sigmoid)
		bias = self.mlp.layer_output.bias  # [n_params]
		bias = bias.repeat((predictor.shape[0], 1))  # [batch, n_params]
		h5 = self.sigmoid(bias / clamped_temp_sigmoid)
		# print("forward_ignoring_input Current params", h5[0, :])
		simu_soc = fun_model_simu(h5, forcing, obs_depth, self.vertical_mixing)  # [batch, n_depths]
		return simu_soc, h5


	def predict_params_summed(self, func, input):
		# Returns predicted parameters (post-sigmoid), summed across the batch dim
		mlp_output = func(input)
		clamped_temp_sigmoid = self.min_temp + (self.max_temp - self.min_temp) * F.sigmoid(self.temp_sigmoid)
		predicted_para = self.sigmoid(mlp_output / clamped_temp_sigmoid)
		return predicted_para.sum(0)

	def get_jacobian(self, input=None):
		"""
		Returns Jacobian, of shape [batch, n_param, n_input].
		For each batch item, it is dParam/dInput.

		If input is not provided, assume something is cached in self.new_input
		"""
		if input is None:
			input = self.new_input  

		# # Naive jacobian. Includes a lot of zero entries as one example's output
		# # is not influenced by other examples' input.
		# batch_jacobian0 = torch.autograd.functional.jacobian(self.mlp, self.new_input)
		# print("Jacobian0", batch_jacobian0.shape)
		# batch_jacobian0 = batch_jacobian0.sum(0)

		# Using jacrev, summing the outputs across batch as each example's output
		# only depends on that example's input
		# perturbed_input = self.new_input + 0.1*torch.randn_like(self.new_input)  # Add noise to input
		batch_jacobian1 = torch.func.jacrev(self.predict_params_summed, argnums=1)(self.mlp, input)  # [n_params, batch, n_inputs]
		batch_jacobian1 = batch_jacobian1.permute((1, 0, 2))  # [batch, n_params, n_inputs]
		# print("Batch jacobian", batch_jacobian1.shape)
		# assert torch.allclose(batch_jacobian0, batch_jacobian1)

		# # Finite difference check
		# x = self.new_input[0, :]
		# eps = 0.01 * torch.randn_like(x)
		# f_x = self.mlp(x.unsqueeze(0))[:, 5]
		# f_x_eps = self.mlp((x+eps).unsqueeze(0))[:, 5]  # f(x+eps)
		# gradient = batch_jacobian1[0, 5, :]
		# f_x_eps_approx = f_x + torch.dot(gradient, eps)
		# assert torch.allclose(f_x_eps, f_x_eps_approx)
		# print("f(x)", f_x)
		# print("f(x+eps)", f_x_eps)
		# print("f(x) + grad f(x) * eps", f_x_eps_approx)


		return batch_jacobian1
		

	def senn_robustness_loss(self, input=None):
		"""
		Computes robustness loss of Self-Explaining Neural Networks:
	
			\| \grad_x f(x) - \theta(x) \|_2^2

		base_model must be SENN. input should have shape [batch, num_inputs].
		We must have previously called forward() on the model, so that self.mlp.coefs
		(theta(x)) are populated.
		"""
		assert self.base_model == "senn"
		if input is None:
			input = self.new_input

		# TODO: Possibly perturb the input	
		J_yx = torch.func.jacrev(self.predict_params_summed, argnums=1)(self.mlp, input)  # [n_params, batch, n_inputs]
		J_yx = J_yx.permute((1, 0, 2))  # [batch, n_params, n_inputs]
		robustness_loss = J_yx - self.mlp.coefs
		# print("Robustness loss", J_yx.shape, self.mlp.coefs.shape, self.mlp.biases.shape)
		return robustness_loss.norm(p='fro')

	def senn_l1_loss(self):
		assert self.base_model == "senn"
		return self.mlp.coefs.abs().mean()

	def senn_sparsity(self):
		assert self.base_model == "senn"
		return (self.mlp.coefs.abs() < 1e-5).float().mean()


	# def partial_forward(self, new_input, input_var, wosis_depth):
	# 	"""
	# 	Helper function which skips the embedding layer and directly passes
	# 	`new_input` through the MLP and process_based model. We only need
	# 	this to use curvature regularization on the MLP."""
	# 	forcing = input_var[:, :, :, :]
	# 	obs_depth = wosis_depth

	# 	# Pass through MLP
	# 	mlp_output = self.mlp(new_input)

	# 	# check if mlp output is nan
	# 	if torch.isnan(mlp_output).any() or torch.isinf(mlp_output).any():
	# 		print("mlp_output was nan", mlp_output)
	# 		exit(1)

	# 	# Clamp temp_sigmoid to be between 10 and 100
	# 	clamped_temp_sigmoid = 10 + 99 * torch.sigmoid(self.temp_sigmoid) # constrain the temp_sigmoid between 10 and 100
	# 	h5 = torch.sigmoid(mlp_output / clamped_temp_sigmoid)

	# 	# check if h5 is nan
	# 	if torch.isnan(h5).any() or torch.isinf(h5).any():
	# 		print("h5 was nan", h5)
	# 		exit(1)

	# 	# CLM5 process-based model
	# 	simu_soc = fun_model_simu(h5, forcing, obs_depth, self.vertical_mixing)
	# 	return simu_soc, h5


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


class ConstantParameters(nn.Module):
	"""
	Finds single set of parameters that works best across all sites.
	Tries to use the same param_constraint as down the line.
	"""
	def __init__(self, num_params, vertical_mixing, param_constraint='sigmoid', min_temp=10, max_temp=109):
		super().__init__()

		# Unconstrained params: learnable tensor
		self.best_params = torch.nn.Parameter(torch.zeros((num_params)), requires_grad=True)

		# Parameter constraint
		self.sigmoid = misc_utils.get_param_constraint(param_constraint)

		# If using sigmoid: the "temperature" we divide by before the sigmoid
		self.temp_sigmoid = nn.Parameter(torch.tensor(0.0), requires_grad=True)
		self.min_temp = min_temp
		self.max_temp = max_temp
		self.vertical_mixing = vertical_mixing


	def forward(self, input_var, wosis_depth, *args, **kwargs):
		forcing = input_var[:, :, :, :]
		obs_depth = wosis_depth

		# Duplicate the constant parameters for each example (site)
		self.unconstrained_params = self.best_params.repeat(input_var.shape[0], 1)

		# Pass parameters through param constraint
		clamped_temp_sigmoid = self.min_temp + (self.max_temp - self.min_temp) * F.sigmoid(self.temp_sigmoid)
		h5 = self.sigmoid(self.unconstrained_params / clamped_temp_sigmoid)

		# print("forward_ignoring_input Current params", h5[0, :])
		simu_soc = fun_matrix_clm5_experimental.fun_model_simu(h5, forcing, obs_depth, self.vertical_mixing)  # [batch, n_depths]
		return simu_soc, h5


#---------------------------------------------------
# Pure NN without process-based model
#---------------------------------------------------
class nn_only(nn.Module):
	def __init__(self, input_vars, var_idx_to_emb, pos_enc, output_dim=140,
				 base_model="new_mlp", one_hot=False, use_bn=False, dropout_prob=0.0,
				 activation="relu", rep_grad=False,
				 losses=["l1", "param_reg"], device="cpu", output_mean=None, output_std=None, train_x=None,
				 min_temp=10, max_temp=109, init="xavier_uniform", width=128, num_layers=4, residual=False):
		"""
		var_idx_to_emb is a dictionary mapping from categorical variable index to either
		(1) Embedding layer (if one_hot is False)
		(2) Number of categories (if one_hot is True)

		If output_mean and output_std are provided, uses them to rescale the output.
		If train_x is provided, use this to rescale the input. Specifically, compute mean/std
		for non-categorical variables as train_x[:, self.non_categorical_indices, 0, 0].mean(dim=0).
		"""
		super().__init__()

		self.one_hot = one_hot
		self.var_idx_to_emb = var_idx_to_emb
		self.pos_enc = pos_enc
		self.output_dim = output_dim

		# List of non-categorical variable indices
		self.non_categorical_indices = list(set(list(range(input_vars))).difference(var_idx_to_emb.keys()))
		self.new_input_size = len(self.non_categorical_indices)

		# Setup categorical encoding
		if self.one_hot:
			# If one-hot, just calculate the input size
			for _, num_classes in self.var_idx_to_emb.items():
				self.new_input_size += num_classes
		else:
			# Create ModuleDict so that all Embedding layers in var_idx_to_emb
			# are registered as parameters
			self.var_idx_to_emb = nn.ModuleDict({str(idx): emb for idx, emb in self.var_idx_to_emb.items()})
			for _, emb in self.var_idx_to_emb.items():
				self.new_input_size += emb.embedding_dim

		# Number of parameters (totally fake)
		self.num_params = 21

		# Standardize input to (mean 0, std 1) if desired
		if train_x is not None:
			train_features = train_x[:, self.non_categorical_indices, 0, 0]
			self.input_mean = train_features.mean(dim=0, keepdim=True)
			self.input_std = train_features.std(dim=0, keepdim=True)
		else:
			self.input_mean = None
			self.input_std = None

		# Output transformation
		self.output_mean = output_mean
		self.output_std = output_std

		# Spatial Encoder from PE-GNN
		if pos_enc == "early":
			self.spatial_encoder = GridCellSpatialRelationEncoder(
				spa_embed_dim=128,
				coord_dim=2, # Longitude and latitude
				frequency_num=16,
				max_radius=360,
				min_radius=1e-06,
				freq_init="geometric",
				ffn=True,  # Enable feedforward network for final spatial embeddings
			)
			self.new_input_size += 128  # Add the spatial embeddings
		elif pos_enc == "late":
			self.spatial_encoder = GridCellSpatialRelationEncoder(
				spa_embed_dim=self.num_params,
				coord_dim=2, # Longitude and latitude
				frequency_num=16,
				max_radius=360,
				min_radius=1e-06,
				freq_init="geometric",
				ffn=True,  # Enable feedforward network for final spatial embeddings
			)

		# MLP backbone
		layer_sizes = [self.new_input_size] + [width] * (num_layers-1) + [self.num_params] 
		if base_model == "lipmlp":
			if residual: raise NotImplementedError("residual network not implemented for lipmlp")
			self.mlp = lipmlp(layer_sizes, use_bn=use_bn, dropout_prob=dropout_prob)  # TODO different initialization methods, activations, dropout, etc. not supported
		elif base_model == "new_mlp":
			self.mlp = mlp(layer_sizes, use_bn=use_bn, dropout_prob=dropout_prob,
				  		   activation=activation, init=init, residual=residual)
		else:
			raise ValueError("Unsupported base_model")

		self.act = misc_utils.get_activation(activation)

		# Final layer: "params" -> SOC pools
		self.final_layer = nn.Linear(self.num_params, self.output_dim)
		if init == "xavier_uniform":
			nn.init.xavier_uniform_(self.final_layer.weight)
			nn.init.zeros_(self.final_layer.bias)
		elif init == "kaiming_uniform":
			nn.init.kaiming_uniform_(self.final_layer.weight)
			nn.init.zeros_(self.final_layer.bias)

		# sigmoid parameter
		self.sigmoid = nn.Sigmoid()


	def forward(self, input_var, wosis_depth, coords, whether_predict,
				return_spatial_embedding=False, **kwargs):
		predictor = input_var[:, 0:self.input_vars, 0, 0]
		forcing = input_var[:, :, :, :]
		obs_depth = wosis_depth

		# For some reason spatial encoder requires coords to have shape [batch, 1, 2] 
		coords = coords.unsqueeze(1).detach()

		# Compute embeddings for all categorical variables
		embs = []
		for idx, embedding_layer in self.var_idx_to_emb.items():
			idx = int(idx)
			if self.one_hot:
				emb = F.one_hot(predictor[:, idx].long(), num_classes=embedding_layer)  # if one_hot, "embedding_layer" is simply the number of classes
			else:
				# NOTE (2025-03-05): Not normalizing embeddings anymore.
				emb = embedding_layer(predictor[:, idx].int())
			embs.append(emb)
		all_embs = torch.concatenate(embs, dim=1)

		# Preprocess numeric (non-categorical) features
		features = predictor[:, self.non_categorical_indices]  # [batch, n_features]
		if self.input_mean is not None and self.input_std is not None:
			features = (features - self.input_mean) / self.input_std

		# Combine numeric features and categorical embeddings
		new_input = torch.concatenate([features, all_embs], dim=1)

		# Spatial Encoding
		if self.pos_enc == "early":
			spatial_embeddings = self.spatial_encoder(coords).squeeze(1) # Remove the channel dimension. [batch, params]
			# print("Spatial emb", spatial_embeddings.shape, "new input", new_input.shape, "coords", coords.shape)
			new_input = torch.concatenate([spatial_embeddings, new_input], dim=1)
		elif self.pos_enc == "late":
			raise ValueError("Late pos_enc not supported for nn_only")

		# check if new_input is nan
		if torch.isnan(new_input).any() or torch.isinf(new_input).any():
			print("new_input was nan", new_input)
			if torch.isnan(spatial_embeddings).any():
				print("Nan in spatial emb")
			if torch.isnan(all_embs).any():
				print("Nan in categorical emb")
			if torch.isnan(features):
				print("Nan in features")
			exit(1)

		# Pass through MLP to get fake pred_para
		self.new_input = new_input
		pred_para = self.mlp(new_input)
		pred_output = self.final_layer(self.act(pred_para))
		if self.output_mean is not None and self.output_std is not None:
			pred_output = pred_output * self.output_std + self.output_mean

		# Convert 140 pools to 20 layers
		pred_output = pred_output.reshape((pred_output.shape[0], 20, 7)).mean(dim=2)

		if whether_predict == 1:
			return pred_output, self.sigmoid(pred_para)
		else:
			simu_soc = misc_utils.select_depth(pred_output, forcing, obs_depth)
			return simu_soc, self.sigmoid(pred_para)



"""
=====================================================================================
BELOW MODELS ARE EXPERIMENTAL. NOT ALL OPTIONS ARE IMPLEMENTED CURRENTLY.
=====================================================================================
"""

class SENN(nn.Module):
	"""
	EXPERIMENTAL. Following the idea of 'Self-Explaining Neural Networks', learns a function

		f(x) = \theta(x)^T x
	
	where x \in R^{num_inputs}, and theta(x): R^{num_inputs} -> R^{(num_inputs + 1) * num_outputs}
	is parameterized by a neural network. The commandline arguments
	all refer to the theta(x) neural network, except for num_outputs.
	"""
	def __init__(self,
				 num_inputs: int,
				 num_outputs: int,
				 num_hidden: int,
				 num_layers: int,
				 use_bn: bool = False,
				 dropout_prob: float = 0.0,
				 activation: str = 'relu',
				 init: str = 'xavier_uniform',
				 residual: bool = False) -> None:

		super().__init__()
		self.num_inputs = num_inputs
		self.num_outputs = num_outputs
		layer_sizes = [self.num_inputs] + ([num_hidden] * (num_layers-1)) + [(self.num_inputs + 1) * self.num_outputs]
		self.theta = mlp(layer_sizes, use_bn=use_bn, dropout_prob=dropout_prob, activation=activation, init=init, residual=residual)


	def forward(self, x):
		"""
		x should have shape [batch, num_inputs]
		
		Returns [batch, num_outputs]
		"""
		coefs_and_biases = self.theta(x)
		self.biases = coefs_and_biases[:, 0:self.num_outputs]
		self.coefs = coefs_and_biases[:, self.num_outputs:].reshape((x.shape[0], self.num_outputs, self.num_inputs))
		self.predictions = torch.bmm(self.coefs, x.unsqueeze(-1)).squeeze(-1) + self.biases  # [batch, num_outputs]
		return self.predictions
	

class BINN_Hybrid(nn.Module):
	"""
	EXPERIMENTAL: BINN Hybrid: process-based model predicts SOC, but NN can correct it.
	Not using this currently.
	"""
	def __init__(self, input_vars, var_idx_to_emb, vertical_mixing, pos_enc,
				 base_model="new_mlp", one_hot=False, use_bn=False, dropout_prob=0.0,
				 activation='relu', param_constraint='sigmoid', rep_grad=False,
				 losses=["l1", "param_reg"], device="cpu"):
		"""
		var_idx_to_emb is a dictionary mapping from categorical variable index to either
		(1) Embedding layer (if one_hot is False)
		(2) Number of categories (if one_hot is True)
		"""
		super().__init__()

		self.one_hot = one_hot
		self.var_idx_to_emb = var_idx_to_emb
		self.vertical_mixing = vertical_mixing
		self.pos_enc = pos_enc

		# List of non-categorical variable indices
		self.input_vars = input_vars
		self.non_categorical_indices = list(set(list(range(input_vars))).difference(var_idx_to_emb.keys()))
		self.new_input_size = len(self.non_categorical_indices)

		# Setup categorical encoding
		if self.one_hot:
			# If one-hot, just calculate the input size
			for _, num_classes in self.var_idx_to_emb.items():
				self.new_input_size += num_classes
		else:
			# Create ModuleDict so that all Embedding layers in var_idx_to_emb
			# are registered as parameters
			self.var_idx_to_emb = nn.ModuleDict({str(idx): emb for idx, emb in self.var_idx_to_emb.items()})
			for _, emb in self.var_idx_to_emb.items():
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
				ffn=True,  # Enable feedforward network for final spatial embeddings
			)
		if pos_enc == "early":
			self.new_input_size += self.num_params  # Add the spatial embeddings

		# MLP backbone
		if base_model == "lipmlp":
			self.mlp = lipmlp((self.new_input_size, 256, 256, self.num_params*2),
							  use_bn=use_bn, dropout_prob=dropout_prob)  # TODO different initialization methods, leaky relu, dropout, etc. not supported
		elif base_model == "new_mlp":
			self.mlp = mlp((self.new_input_size, 256, 256, self.num_params*2),
							  use_bn=use_bn, dropout_prob=dropout_prob, activation=activation)
		else:
			raise ValueError("Unsupported base_model")

		# Use second half of MLP output to directly predict residual (error)
		# of process-based model
		self.predict_residual = nn.Linear(self.num_params, 140)

		# sigmoid parameter
		self.temp_sigmoid = nn.Parameter(torch.tensor(0.0), requires_grad=True)
		self.sigmoid = misc_utils.get_param_constraint(param_constraint)

		# LibMTL specific
		self.rep_grad = rep_grad  # Whether to compute gradients w.r.t. parameters as well
		self.task_name = losses
		self.task_num = len(self.task_name)
		self.device = device


	def forward(self, input_var, wosis_depth, coords, whether_predict,
				return_residual=False):
		predictor = input_var[:, 0:self.input_vars, 0, 0]
		forcing = input_var[:, :, :, :]
		obs_depth = wosis_depth
		coords = coords.unsqueeze(1).detach().cpu().numpy()  # coords should be a numpy array

		# Compute embeddings for all categorical variables
		embs = []
		for idx, embedding_layer in self.var_idx_to_emb.items():
			idx = int(idx)
			if self.one_hot:
				emb = F.one_hot(predictor[:, idx].long(), num_classes=embedding_layer)  # if one_hot, "embedding_layer" is simply the number of classes
			else:
				emb = embedding_layer(predictor[:, idx].int())
				emb = F.normalize(emb, p=2, dim=1)  # New @joshuafan: normalize embeddings
			embs.append(emb)
		all_embs = torch.concatenate(embs, dim=1)
		new_input = torch.concatenate([predictor[:, self.non_categorical_indices], all_embs], dim=1)

		# Spatial Encoding
		if self.pos_enc == "early":
			spatial_embeddings = self.spatial_encoder(coords).squeeze(1) # Remove the channel dimension. [batch, params]
			# print("Spatial emb", spatial_embeddings.shape, "new input", new_input.shape, "coords", coords.shape)
			new_input = torch.concatenate([spatial_embeddings, new_input], dim=1)

		# check if new_input is nan
		if torch.isnan(new_input).any() or torch.isinf(new_input).any():
			print("new_input was nan", new_input)
			exit(1)

		# Pass through MLP
		self.new_input = new_input
		mlp_output = self.mlp(new_input)

		# check if mlp output is nan
		if torch.isnan(mlp_output).any() or torch.isinf(mlp_output).any():
			print("mlp_output was nan", mlp_output)
			exit(1)

		# Clamp temp_sigmoid to be between 10 and 200
		clamped_temp_sigmoid = 10 + 99*F.sigmoid(self.temp_sigmoid)   #10 + 99 * self.sigmoid(self.temp_sigmoid) # try with a smaller range

		# Positional encoder correction (if using)
		if self.pos_enc == "late":
			spatial_embeddings = self.spatial_encoder(coords).squeeze(1) # Remove the channel dimension. [batch, params]
			mlp_output[:, 0:self.num_params] += spatial_embeddings

		# Pass parameters through sigmoid to constrain their range
		pred_para = self.sigmoid(mlp_output[:, 0:self.num_params] / clamped_temp_sigmoid)

		# check if h5 is nan
		if torch.isnan(pred_para).any() or torch.isinf(pred_para).any():
			print("pred_para was nan", pred_para)
			exit(1)

		# Predict residual
		pbm_residual = self.predict_residual(mlp_output[:, self.num_params:]).reshape((mlp_output.shape[0], 20, 7)).mean(dim=2)

		# CLM5 process-based model
		if whether_predict == 1:
			simu_soc = fun_model_prediction(pred_para, forcing, self.vertical_mixing, residual=pbm_residual)
		else:
			simu_soc = fun_model_simu(pred_para, forcing, obs_depth, self.vertical_mixing, residual=pbm_residual)

		if return_residual:
			return simu_soc, pred_para, pbm_residual
		else:
			return simu_soc, pred_para


class GNN_BINN(nn.Module):
	"""
	EXPERIMENTAL: Use PEGCN as encoder for BINN model
	"""
	def __init__(self, input_vars, var_idx_to_emb, vertical_mixing, pos_enc,
				 k=20, one_hot=False, use_bn=False, dropout_prob=0.0,
				 activation='relu', param_constraint='sigmoid', rep_grad=False, graph_conv='gcn',
				 gnn_input_dim=256, emb_hidden_dim=128,
				 losses=["l1", "param_reg"], device="cpu"):
		"""
		var_idx_to_emb is a dictionary mapping from categorical variable index to either
		(1) Embedding layer (if one_hot is False)
		(2) Number of categories (if one_hot is True)
		"""
		super().__init__()

		self.one_hot = one_hot
		self.var_idx_to_emb = var_idx_to_emb
		self.vertical_mixing = vertical_mixing
		self.pos_enc = pos_enc
		self.k = k

		# List of non-categorical variable indices
		self.input_vars = input_vars
		self.non_categorical_indices = list(set(list(range(input_vars))).difference(var_idx_to_emb.keys()))
		self.new_input_size = len(self.non_categorical_indices)

		# Setup categorical encoding
		if self.one_hot:
			# If one-hot, just calculate the input size
			for _, num_classes in self.var_idx_to_emb.items():
				self.new_input_size += num_classes
		else:
			# Create ModuleDict so that all Embedding layers in var_idx_to_emb
			# are registered as parameters
			self.var_idx_to_emb = nn.ModuleDict({str(idx): emb for idx, emb in self.var_idx_to_emb.items()})
			for _, emb in self.var_idx_to_emb.items():
				self.new_input_size += emb.embedding_dim

		# Add spatial embedding to new_input size
		if pos_enc == "early":
			self.new_input_size += self.num_params

		# Number of parameters
		if self.vertical_mixing == 'simple_two_intercepts':
			self.num_params = 22
		else:
			self.num_params = 21

		# Spatial Encoder
		if pos_enc != "none":
			self.spenc = GridCellSpatialRelationEncoder(spa_embed_dim=emb_hidden_dim,ffn=True,min_radius=1e-06,max_radius=360)
			self.dec = nn.Sequential(
				nn.Linear(emb_hidden_dim, emb_hidden_dim // 2),
				nn.Tanh(),
				nn.Linear(emb_hidden_dim // 2, emb_hidden_dim // 4),
				nn.Tanh(),
				nn.Linear(emb_hidden_dim // 4, self.num_params)
			)

		# GNN
		self.initial_fc = mlp((self.new_input_size, 256, 32), use_bn=use_bn, dropout_prob=dropout_prob, activation=activation)
		#nn.Linear(self.new_input_size, gnn_input_dim)
		self.graph_conv = graph_conv
		if graph_conv == 'gcn':
			self.conv1 = GCNConv(32, 32)  #32)
			self.conv2 = GCNConv(32, 32)  #32, 32)
		elif graph_conv == 'gat':
			self.conv1 = GATConv(32, 32) #32)
			self.conv2 = GATConv(32, 32)  #32, 32)
		elif graph_conv == 'gcn1':
			self.conv1 = GCNConv(32, 32)
			self.conv2 = None
		elif graph_conv == 'gat1':
			self.conv1 = GATConv(32, 32)
			self.conv2 = None
		self.fc = nn.Linear(32, self.num_params)  #    `32, self.num_params)

		# Parameter that influences graph edge weights
		self.length_scale = nn.Parameter(torch.tensor(1.0), requires_grad=True)

		# Dropout/activation
		self.dropout = nn.Dropout(dropout_prob)
		self.act = misc_utils.get_activation(activation)

		# sigmoid parameter
		self.temp_sigmoid = nn.Parameter(torch.tensor(0.0), requires_grad=True)
		self.sigmoid = misc_utils.get_param_constraint(param_constraint)

		# LibMTL specific
		self.rep_grad = rep_grad  # Whether to compute gradients w.r.t. parameters as well
		self.task_name = losses
		self.task_num = len(self.task_name)
		self.device = device


	def forward(self, input_var, wosis_depth, coords,
				 whether_predict, ei=None, ew=None, plot_dir=None, return_extra=False):
		predictor = input_var[:, 0:self.input_vars, 0, 0]
		forcing = input_var[:, :, :, :]
		obs_depth = wosis_depth
		# print("Predictor dtype", predictor.dtype, "Coords", coords.dtype)
		# predictor = predictor.float()
		# coords = coords.float()

		# Construct graph
		if torch.is_tensor(ei) & torch.is_tensor(ew):
			edge_index = ei
			edge_weight = ew
		else:
			edge_index = knn_graph(coords, k=self.k).to(self.device)
			edge_weight = makeEdgeWeight(coords, edge_index).to(self.device)
			edge_weight = torch.exp(-1.0 * edge_weight / self.length_scale)

		# Compute spatial embedding
		coords = coords.detach().cpu().numpy()
		if self.pos_enc != 'none':
			coords = coords.reshape(1, coords.shape[0], coords.shape[1])
			spatial_emb = self.spenc(coords)  #.detach().cpu().numpy())
			spatial_emb = spatial_emb.reshape(spatial_emb.shape[1], spatial_emb.shape[2])
			spatial_emb = self.dec(spatial_emb).float()
			coords = coords.reshape(coords.shape[1], coords.shape[2])
		else:
			spatial_emb = None

		# Compute embeddings for all categorical variables
		embs = []
		for idx, embedding_layer in self.var_idx_to_emb.items():
			if self.one_hot:
				emb = F.one_hot(predictor[:, idx].long(), num_classes=embedding_layer)  # if one_hot, "embedding_layer" is simply the number of classes
			else:
				emb = embedding_layer(predictor[:, idx].int())
				emb = F.normalize(emb, p=2, dim=1)  # New @joshuafan: normalize embeddings
			embs.append(emb)
		all_embs = torch.concatenate(embs, dim=1)
		x = torch.concatenate([predictor[:, self.non_categorical_indices], all_embs], dim=1)

		# If `pos_enc` is early, concatenate spatial embeddings before passing through GNN
		if self.pos_enc == "early":
			x = torch.concatenate([x, spatial_emb], dim=1)

		# Pass through GNN
		x = self.act(self.initial_fc(x))
		h1 = self.act(self.conv1(x, edge_index, edge_weight))
		h1 = self.dropout(h1)
		if self.conv2 is not None:
			h1 = self.act(self.conv2(h1, edge_index, edge_weight))
			h1 = self.dropout(h1)
		gnn_output = self.fc(h1)

		# Clamp temp_sigmoid to be between 10 and 200
		clamped_temp_sigmoid = 10 + 99 * F.sigmoid(self.temp_sigmoid) # try with a smaller range

		# Pass parameters through sigmoid to constrain their range
		if self.pos_enc == "late":
			pred_para = self.sigmoid(self.sigmoid(gnn_output / clamped_temp_sigmoid) + self.sigmoid(spatial_emb / clamped_temp_sigmoid))
		else:
			pred_para = self.sigmoid(gnn_output / clamped_temp_sigmoid)

		# Visualizations
		if plot_dir is not None:
			for param_idx in [0, 20]:
				# Initial GNN output
				visualization_utils.plot_observations_world_map(coords[:, 0], coords[:, 1],
																gnn_output[:, param_idx].detach().cpu().numpy(), plot_dir,
																f"param_{param_idx}_initial",
																title=f"Param {param_idx} initial", us_only=True,
																graph_edgeindex=edge_index, graph_edgeweights=edge_weight)

				# Spatial embedding
				if self.pos_enc != 'none':
					visualization_utils.plot_observations_world_map(coords[:, 0], coords[:, 1],
																	spatial_emb[:, param_idx].detach().cpu().numpy(), plot_dir,
																	f"param_{param_idx}_spatialemb",
																	title=f"Param {param_idx} spatial embedding", us_only=True,
																	graph_edgeindex=edge_index, graph_edgeweights=edge_weight)

				# Final param
				visualization_utils.plot_observations_world_map(coords[:, 0], coords[:, 1],
																pred_para[:, param_idx].detach().cpu().numpy(), plot_dir,
																f"param_{param_idx}_final",
																title=f"Param {param_idx} FINAL", us_only=True,
																graph_edgeindex=edge_index, graph_edgeweights=edge_weight)

		# check if h5 is nan
		if torch.isnan(pred_para).any() or torch.isinf(pred_para).any():
			print("pred_para was nan", pred_para)
			exit(1)

		# CLM5 process-based model
		if whether_predict == 1:
			simu_soc = fun_model_prediction(pred_para, forcing, self.vertical_mixing)
		else:
			simu_soc = fun_model_simu(pred_para, forcing, obs_depth, self.vertical_mixing)

		# Compute Laplacian
		l_ei, l_ew = get_laplacian(edge_index, edge_weight)
		laplacian = to_dense_adj(l_ei, batch=None, edge_attr=l_ew).squeeze(0)
		if return_extra:
			return simu_soc, pred_para, spatial_emb, laplacian
		else:
			return simu_soc, pred_para


class Spatial_BINN(nn.Module):
	"""
	EXPERIMENTAL: Simple spatial smoothing on predicted params.
	Positional encoding outputs spatially-correlated errors,
	use penalty to encourage spatial smoothness
	"""
	def __init__(self, input_vars, var_idx_to_emb, vertical_mixing, pos_enc,
				 k=20, one_hot=False, rep_grad=False,
				 use_bn=False, dropout_prob=0.0, 
				 activation='relu', param_constraint='sigmoid', emb_hidden_dim=128,
				 losses=["l1", "param_reg"], device="cpu"):
		"""
		var_idx_to_emb is a dictionary mapping from categorical variable index to either
		(1) Embedding layer (if one_hot is False)
		(2) Number of categories (if one_hot is True)
		"""
		super().__init__()

		self.one_hot = one_hot
		self.var_idx_to_emb = var_idx_to_emb
		self.vertical_mixing = vertical_mixing
		self.pos_enc = pos_enc
		self.k = k

		# List of non-categorical variable indices
		self.input_vars = input_vars
		self.non_categorical_indices = list(set(list(range(input_vars))).difference(var_idx_to_emb.keys()))
		self.new_input_size = len(self.non_categorical_indices)

		# Setup categorical encoding
		if self.one_hot:
			# If one-hot, just calculate the input size
			for _, num_classes in self.var_idx_to_emb.items():
				self.new_input_size += num_classes
		else:
			# Create ModuleDict so that all Embedding layers in var_idx_to_emb
			# are registered as parameters
			self.var_idx_to_emb = nn.ModuleDict({str(idx): emb for idx, emb in self.var_idx_to_emb.items()})
			for _, emb in self.var_idx_to_emb.items():
				self.new_input_size += emb.embedding_dim

		# Number of parameters
		if self.vertical_mixing == 'simple_two_intercepts':
			self.num_params = 22
		else:
			self.num_params = 21

		# Fully-connected
		self.mlp = mlp((self.new_input_size, 256, 256, self.num_params),
						use_bn=use_bn, dropout_prob=dropout_prob, activation=activation)

		# Spatial smoother
		self.smoother = SpatialSmoother(self.num_params, k=k)

		# Spatial Encoder (for spatially-dependent error)
		if self.pos_enc != "none":
			assert self.pos_enc == "late"
			self.spenc = GridCellSpatialRelationEncoder(spa_embed_dim=emb_hidden_dim,ffn=True,min_radius=1e-06,max_radius=360)
			self.dec = nn.Sequential(
				nn.Linear(emb_hidden_dim, emb_hidden_dim // 2),
				nn.Tanh(),
				nn.Linear(emb_hidden_dim // 2, emb_hidden_dim // 4),
				nn.Tanh(),
				nn.Linear(emb_hidden_dim // 4, self.num_params)
			)

		# sigmoid parameter
		self.temp_sigmoid = nn.Parameter(torch.tensor(0.0), requires_grad=True)
		self.sigmoid = misc_utils.get_param_constraint(param_constraint)

		# LibMTL specific
		self.rep_grad = rep_grad  # Whether to compute gradients w.r.t. parameters as well
		self.task_name = losses
		self.task_num = len(self.task_name)
		self.device = device


	def forward(self, input_var, wosis_depth, coords,
				whether_predict, ei=None, ew=None,
				plot_dir=None, return_extra=False):
		predictor = input_var[:, 0:self.input_vars, 0, 0]
		forcing = input_var[:, :, :, :]
		obs_depth = wosis_depth

		# Compute edges and edge weights. We do this here (even though SpatialSmoother
		# also has this functionality) so we can compute spatial losses based on the
		# Laplacian.
		if torch.is_tensor(ei) & torch.is_tensor(ew):
			edge_index = ei
			edge_weight = ew
		else:
			# print("Coords", coords[0:10])
			edge_index = knn_graph(coords, k=self.k).to(self.device)
			edge_weight = makeEdgeWeight(coords, edge_index).to(self.device)
			# print("Edge index", edge_index[:, 0:10])
			# print("Edge weight", edge_weight[0:10])
			edge_weight = torch.exp(-1.0 * (edge_weight**2) / (2*(self.smoother.length_scale**2)))
			# print("After RBF", edge_weight[0:10])

		# Compute embeddings for all categorical variables
		embs = []
		for idx, embedding_layer in self.var_idx_to_emb.items():
			if self.one_hot:
				emb = F.one_hot(predictor[:, idx].long(), num_classes=embedding_layer)  # if one_hot, "embedding_layer" is simply the number of classes
			else:
				emb = embedding_layer(predictor[:, idx].int())
				emb = F.normalize(emb, p=2, dim=1)  # New @joshuafan: normalize embeddings
			embs.append(emb)
		all_embs = torch.concatenate(embs, dim=1)
		new_input = torch.concatenate([predictor[:, self.non_categorical_indices], all_embs], dim=1)

		# Pass feature vectors through MLP to get initial predicitions
		mlp_output = self.mlp(new_input)

		# Clamp temp_sigmoid to be between 10 and 200
		clamped_temp_sigmoid = 10 + 99 * F.sigmoid(self.temp_sigmoid) # try with a smaller range

		# Positional encoder correction
		if self.pos_enc == "late":
			coords = coords.detach().cpu().numpy()
			coords = coords.reshape(1, coords.shape[0], coords.shape[1])  # coords should be a numpy array if passing directly to GridCellSpatialRelationEncoder
			spatial_emb = self.spenc(coords)
			spatial_emb = spatial_emb.reshape(spatial_emb.shape[1], spatial_emb.shape[2])  # Remove the channel dimension. [batch, params]
			spatial_emb = self.dec(spatial_emb).float()
			coords = torch.tensor(coords.reshape(coords.shape[1], coords.shape[2]), device=self.device)
			# print("MLP output", mlp_output[0:10], "Spatial emb", spatial_emb[0:10])
			mlp_output = mlp_output + spatial_emb

			# Pass parameters through sigmoid to constrain their range
			# pred_para = self.sigmoid(self.sigmoid(smoothed_output / clamped_temp_sigmoid) + self.sigmoid(spatial_emb / clamped_temp_sigmoid))
		else:
			# pred_para = self.sigmoid(smoothed_output / clamped_temp_sigmoid)
			spatial_emb = None

		unsmoothed_para = self.sigmoid(mlp_output / clamped_temp_sigmoid)

		# Smooth the predictions
		pred_para = self.smoother(unsmoothed_para, coords, edge_index, edge_weight)
		coords = coords.detach().cpu().numpy()

		# Visualizations
		if plot_dir is not None:
			print("Smoothness length_scale", self.smoother.length_scale)
			for param_idx in [0, 20]:
				# Initial MLP output
				visualization_utils.plot_observations_world_map(coords[:, 0], coords[:, 1],
																mlp_output[:, param_idx].detach().cpu().numpy(), plot_dir,
																f"param_{param_idx}_initial",
																title=f"Param {param_idx} initial", us_only=True,
																graph_edgeindex=edge_index, graph_edgeweights=edge_weight)
				# Spatial correction
				if self.pos_enc != "none":
					visualization_utils.plot_observations_world_map(coords[:, 0], coords[:, 1],
																	spatial_emb[:, param_idx].detach().cpu().numpy(), plot_dir,
																	f"param_{param_idx}_correction",
																	title=f"Param {param_idx} correction", us_only=True,
																	graph_edgeindex=edge_index, graph_edgeweights=edge_weight)

				# Unsmoothed param
				visualization_utils.plot_observations_world_map(coords[:, 0], coords[:, 1],
																unsmoothed_para[:, param_idx].detach().cpu().numpy(), plot_dir,
																f"param_{param_idx}_unsmoothed",
																title=f"Param {param_idx} UNSMOOTHED", us_only=True,
																graph_edgeindex=edge_index, graph_edgeweights=edge_weight)
				# Smoothed param
				visualization_utils.plot_observations_world_map(coords[:, 0], coords[:, 1],
																pred_para[:, param_idx].detach().cpu().numpy(), plot_dir,
																f"param_{param_idx}_smoothed",
																title=f"Param {param_idx} smoothed", us_only=True,
																graph_edgeindex=edge_index, graph_edgeweights=edge_weight)

		# check if h5 is nan
		if torch.isnan(pred_para).any() or torch.isinf(pred_para).any():
			print("pred_para was nan", pred_para)
			exit(1)

		# CLM5 process-based model
		if whether_predict == 1:
			simu_soc = fun_model_prediction(pred_para, forcing, self.vertical_mixing)
		else:
			simu_soc = fun_model_simu(pred_para, forcing, obs_depth, self.vertical_mixing)

		# Compute Laplacian
		l_ei, l_ew = get_laplacian(edge_index, edge_weight)
		laplacian = to_dense_adj(l_ei, batch=None, edge_attr=l_ew).squeeze(0)
		# print("Laplacian", laplacian[0:10, 0:10])
		# print("Spatial emb", spatial_emb[0:10])
		if return_extra:
			return simu_soc, pred_para, spatial_emb, laplacian
		else:
			return simu_soc, pred_para