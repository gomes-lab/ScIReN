import torch
import torch.nn as nn
import torch.nn.functional as F
from lipmlp import lipmlp
from fun_matrix_clm5_vectorized import fun_model_simu

class mlp(torch.nn.Module):
    """
	New MLP from this repo: https://github.com/whitneychiu/lipmlp_pytorch/blob/main/models/mlp.py
	"""
    def __init__(self, dims, use_bn=False):
        """
        dim[0]: input dim
        dim[1:-1]: hidden dims
        dim[-1]: out dim

        assume len(dims) >= 3
        """
        super().__init__()

        self.layers = torch.nn.ModuleList()
        self.use_bn = use_bn
        if use_bn:
            self.bns = torch.nn.ModuleList()
        for ii in range(len(dims)-2):
            self.layers.append(torch.nn.Linear(dims[ii], dims[ii+1]))
            if use_bn:
                self.bns.append(torch.nn.BatchNorm1d(dims[ii+1]))
        self.layer_output = torch.nn.Linear(dims[-2], dims[-1])
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

#---------------------------------------------------
# Wrapper for MLP and LipMLP from this repo: https://github.com/whitneychiu/lipmlp_pytorch/blob/main/models/mlp.py
#---------------------------------------------------
# define model
class mlp_wrapper(nn.Module):
	def __init__(self, input_vars, var_idx_to_emb, lipschitz=False, one_hot=False, use_bn=False):
		super().__init__()

		# If one_hot is True, this is a Dict from categorical variable index to number of categories.
		# If one_hot is False, this is a Dict from categorical variable index -> Embedding layer we use
		self.one_hot = one_hot
		self.var_idx_to_emb = var_idx_to_emb

		# List of non-categorical variable indices
		self.non_categorical_indices = list(set(list(range(input_vars))).difference(var_idx_to_emb.keys()))
		new_input_size = len(self.non_categorical_indices)
		for idx, emb in self.var_idx_to_emb.items():
			if self.one_hot:
				new_input_size += emb
			else:
				new_input_size += emb.embedding_dim

		# MLP backbone
		if lipschitz:
			self.mlp = lipmlp((new_input_size, 256, 256, 256, 21), use_bn=use_bn)  # TODO different initialization methods, leaky relu, dropout, etc. not supported
		else:
			self.mlp = mlp((new_input_size, 256, 256, 256, 21), use_bn=use_bn)

		# sigmoid parameter
		# self.temp_sigmoid = nn.Parameter(torch.tensor(0.0), requires_grad=True)


	def forward(self, input_var, wosis_depth):
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

		# Pass through MLP
		mlp_output = self.mlp(new_input)

		# Clamp temp_sigmoid to be between 10 and 100
		# clamped_temp_sigmoid = 10 + 90 * torch.sigmoid(self.temp_sigmoid) # constrain the temp_sigmoid between 10 and 100
		clamped_temp_sigmoid = 1
		h5 = torch.sigmoid(mlp_output / clamped_temp_sigmoid)

		# check if h5 is nan
		if torch.isnan(h5).any() or torch.isinf(h5).any():
			print("h5 was nan", h5)
			exit(1) 

		# CLM5 process-based model
		simu_soc = fun_model_simu(h5, forcing, obs_depth)
		return simu_soc, h5, clamped_temp_sigmoid
		
