import fun_matrix_clm5_experimental
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pe_gcn_model import GridCellSpatialRelationEncoder
import misc_utils


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
		"""NOT USED CURRENTLY, this could be another way to perform Lipschitz (smoothness) regularization.

		Code from https://github.com/NVlabs/NVAE/blob/master/model.py
			
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
				 activation='relu', param_constraint='sigmoid',
				 device="cpu", train_x=None,
				 min_temp=10, max_temp=109, init="xavier_uniform", final_bias="none", width=128,
				 para_index=None, num_layers=4, residual=False,
				 kan_grid=3, kan_grid_margin=0.0, kan_noise=0.3, kan_scale_base_sigma=1.0, 
				 kan_base_fun="silu", kan_affine_trainable=False,
				 kan_absolute_deviation=False, kan_drop_rate=0.0, kan_drop_mode="postact", kan_drop_scale=True):
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
		# elif base_model == "kan":
		# 	# If KAN, shift input to [-1, 1] range
		# 	print("KAN - Shifting inputs to [-1, 1]")
		# 	self.input_mean = 0.5
		# 	self.input_std = 0.5
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
		if base_model == "new_mlp":
			# Basic MLP
			self.mlp = mlp(layer_sizes, use_bn=use_bn, dropout_prob=dropout_prob,
				  		   activation=activation, init=init, residual=residual)
		elif base_model == "kan":
			import kan
			# grid_eps = 1: use evenly-spaced grid
			self.mlp = kan.KAN(width=layer_sizes, grid=kan_grid, k=3, seed=torch.initial_seed(), device=device,
					  		   input_size=len(self.non_categorical_indices), noise_scale=kan_noise, scale_base_sigma=kan_scale_base_sigma,
							   base_fun=kan_base_fun, affine_trainable=kan_affine_trainable, grid_eps=1.0, 
							   grid_margin=kan_grid_margin, absolute_deviation=kan_absolute_deviation,
							   drop_rate=kan_drop_rate, drop_mode=kan_drop_mode, drop_scale=kan_drop_scale,
							   batch_norm_spline=use_bn)
			# self.mlp.speed()  # Disable symbolic branch
		else:
			raise ValueError("Unsupported base_model")

		# Parameter constraint
		self.param_constraint = param_constraint
		self.sigmoid = misc_utils.get_param_constraint(param_constraint)

		# If using sigmoid: the "temperature" we divide by before the sigmoid
		self.temp_sigmoid = nn.Parameter(torch.tensor(0.0), requires_grad=True)
		self.min_temp = min_temp
		self.max_temp = max_temp
		if final_bias == "none":
			self.final_bias = 0.
		elif final_bias == "zero_init":
			self.final_bias = nn.Parameter(torch.zeros(self.num_params), requires_grad=True)
		elif final_bias == "uniform2_init":  # Unif[-2, 2]
			self.final_bias = nn.Parameter(torch.rand(self.num_params) * 4 - 2, requires_grad=True)
		elif final_bias == "uniform1_init":
			self.final_bias = nn.Parameter(torch.rand(self.num_params) * 2 - 1, requires_grad=True)
		else:
			raise ValueError("Invalid value of final_bias")


	def forward(self, input_var, wosis_depth, coords, whether_predict, PRODA_para=None,
			 	return_spatial_embedding=False, ignore_input=False):

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
			return None, None

		# Clamp temp_sigmoid to be within a range
		clamped_temp_sigmoid = self.min_temp + (self.max_temp - self.min_temp) * F.sigmoid(self.temp_sigmoid)  # 10 + 90*self.sigmoid(self.temp_sigmoid)   #10 + 99 * self.sigmoid(self.temp_sigmoid) # try with a smaller range

		# Positional encoder correction (if using)
		if self.pos_enc == "late":
			spatial_embeddings = self.spatial_encoder(coords).squeeze(1)  # Remove the channel dimension. [batch, params]
			mlp_output += spatial_embeddings

		# Pass biogeochemical parameters through sigmoid, constraining them between [0, 1]
		# print("Dist of mlp_output", mlp_output.mean(dim=0), mlp_output.std(dim=0))
		# print("Final bias", self.final_bias)
		self.unconstrained_params = (mlp_output / clamped_temp_sigmoid) + self.final_bias
		# print("Dist of unconstrained params", self.unconstrained_params.mean(dim=0), self.unconstrained_params.std(dim=0))
		constrained_params = self.sigmoid(self.unconstrained_params)
		# print("Dist of constrained params", constrained_params.mean(dim=0), constrained_params.std(dim=0))

		if PRODA_para is None:  #  or len(self.para_index) == constrained_params.shape[1]:  # If we are predicting all params, don't need to copy PRODA_para
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

		# Ignore examples where predicted_para was nan. This should only happen when PRODA_para
		# contains nan values.
		self.predicted_para = predicted_para
		valid_mask = ~torch.any((torch.isnan(predicted_para) | torch.isinf(predicted_para)), dim=1)
		if torch.sum(~valid_mask) > 0:
			print("predicted_para had nan")

		# CLM5 process-based model
		if whether_predict == 1:
			simu_soc = fun_matrix_clm5_experimental.fun_model_prediction(predicted_para[valid_mask], forcing, self.vertical_mixing, self.vectorized)
		else:
			simu_soc = fun_matrix_clm5_experimental.fun_model_simu(predicted_para[valid_mask], forcing, obs_depth, self.vertical_mixing, self.vectorized)

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
		simu_soc = fun_matrix_clm5_experimental.fun_model_simu(h5, forcing, obs_depth, self.vertical_mixing)  # [batch, n_depths]
		return simu_soc, h5


	def update_grid(self, *args, **kwargs):
		"""
		Wrapper around update_grid for KAN inside."""
		assert self.base_model == "kan"

		# Call forward to obtain "new_input" (the input to the KAN)
		# Then use that to update the grid in the KAN
		self.forward(*args, **kwargs)
		self.mlp.update_grid_from_samples(self.new_input)


	def predict_params_summed(self, func, input):
		# Returns predicted parameters (post-sigmoid), summed across the batch dim
		mlp_output = func(input)
		clamped_temp_sigmoid = self.min_temp + (self.max_temp - self.min_temp) * F.sigmoid(self.temp_sigmoid)
		if self.param_constraint == "hardsigmoid":
			# For some reason, taking Jacobian across hardsigmoid doesn't work, but that's
			# ok as hardisgmoid is linear in the valid range
			predicted_para = mlp_output / clamped_temp_sigmoid
		else:
			predicted_para = self.sigmoid(mlp_output / clamped_temp_sigmoid)
		return predicted_para.sum(0)


	def get_jacobian(self, input=None, noise_std=0):
		"""
		Returns Jacobian, of shape [batch, n_param, n_input].
		For each batch item, it is dParam/dInput.

		If input is not provided, assume something is cached in self.new_input
		"""
		if input is None:
			input = self.new_input
		if noise_std > 0:
			input += torch.randn(input.shape) * input.std(dim=0, keepdim=True) * noise_std

		# Using jacrev, summing the outputs across batch as each example's output
		# only depends on that example's input
		# perturbed_input = self.new_input + 0.1*torch.randn_like(self.new_input)  # Add noise to input
		batch_jacobian1 = torch.func.jacrev(self.predict_params_summed, argnums=1)(self.mlp, input)  # [n_params, batch, n_inputs]
		batch_jacobian1 = batch_jacobian1.permute((1, 0, 2))  # [batch, n_params, n_inputs]
		return batch_jacobian1


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
				 activation="relu",
				 output_mean=None, output_std=None, train_x=None,
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
		if base_model == "new_mlp":
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

		# Preprocess numeric (non-categorical) features
		features = predictor[:, self.non_categorical_indices]  # [batch, n_features]
		if self.input_mean is not None and self.input_std is not None:
			features = (features - self.input_mean) / self.input_std

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
			spatial_embeddings = self.spatial_encoder(coords).squeeze(1) # Remove the channel dimension. [batch, params]
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

