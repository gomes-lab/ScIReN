import torch
import torch.nn as nn
import torch.nn.functional as F
from lipmlp import lipmlp
from fun_matrix_clm5_vectorized import fun_model_simu

class mlp(torch.nn.Module):
    """
	New MLP from this repo: https://github.com/whitneychiu/lipmlp_pytorch/blob/main/models/mlp.py
	"""
    def __init__(self, dims):
        """
        dim[0]: input dim
        dim[1:-1]: hidden dims
        dim[-1]: out dim

        assume len(dims) >= 3
        """
        super().__init__()

        self.layers = torch.nn.ModuleList()
        for ii in range(len(dims)-2):
            self.layers.append(torch.nn.Linear(dims[ii], dims[ii+1]))

        self.layer_output = torch.nn.Linear(dims[-2], dims[-1])
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        for ii in range(len(self.layers)):
            x = self.layers[ii](x)
            x = self.relu(x)
        return self.layer_output(x)
    

#---------------------------------------------------
# Wrapper for MLP and LipMLP from this repo: https://github.com/whitneychiu/lipmlp_pytorch/blob/main/models/mlp.py
#---------------------------------------------------
# define model
class mlp_wrapper(nn.Module):
	def __init__(self, input_vars, var_idx_to_emb, lipschitz=False):
		super().__init__()

		# Dict from categorical variable index -> Embedding layer we use
		self.var_idx_to_emb = var_idx_to_emb

		# List of non-categorical variable indices
		self.non_categorical_indices = list(set(list(range(input_vars))).difference(var_idx_to_emb.keys()))
		new_input_size = len(self.non_categorical_indices)
		for idx, emb in self.var_idx_to_emb.items():
			new_input_size += emb.embedding_dim

		# MLP backbone
		if lipschitz:
			self.mlp = lipmlp((new_input_size, 256, 256, 256, 21))  # TODO different initialization methods, leaky relu, dropout, etc. not supported
		else:
			self.mlp = mlp((new_input_size, 256, 256, 256, 21))

		# sigmoid parameter
		# self.temp_sigmoid = nn.Parameter(torch.tensor(0.0), requires_grad=True)


	def forward(self, input_var, wosis_depth):
		predictor = input_var[:, :, 0, 0]
		forcing = input_var[:, :, :, :]
		obs_depth = wosis_depth

		# Compute embeddings for all categorical variables
		embs = []
		for idx, embedding_layer in self.var_idx_to_emb.items():
			emb = embedding_layer(predictor[:, idx].int())
			emb = F.normalize(emb, p=2, dim=1)  # New @joshuafan: normalize embeddings
			embs.append(emb)
		all_embs = torch.concatenate(embs, dim=1)
		new_input = torch.concatenate([predictor[:, self.non_categorical_indices], all_embs], dim=1)

		# Pass through MLP
		mlp_output = self.mlp(new_input)

		# Clamp temp_sigmoid to be between 10 and 100
		# clamped_temp_sigmoid = 10 + 90 * torch.sigmoid(self.temp_sigmoid) # constrain the temp_sigmoid between 10 and 100
		clamped_temp_sigmoid = 50
		h5 = torch.sigmoid(mlp_output / clamped_temp_sigmoid)

		# check if h5 is nan
		if torch.isnan(h5).any() or torch.isinf(h5).any():
			print("h5 was nan", h5)
			exit(1) 

		# CLM5 process-based model
		simu_soc = fun_model_simu(h5, forcing, obs_depth)
		return simu_soc, h5, clamped_temp_sigmoid
