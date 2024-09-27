import torch
import torch.nn as nn
import torch.nn.functional as F
from lipmlp import lipmlp
from fun_matrix_clm5_vectorized import fun_model_simu
from fun_matrix_clm5_vectorized_prediction import fun_model_prediction
from pe_gcn_model import GCN, PEGCN, GridCellSpatialRelationEncoder, SpatialSmoother
from torch_geometric.nn import GCNConv, GATConv, SimpleConv, knn_graph
from torch_geometric.utils import get_laplacian, to_dense_adj, to_torch_coo_tensor
from misc_utils import select_depth
from spatial_utils import *
import visualization_utils

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

        # Initialize linear layers
        for layer in self.layers + [self.layer_output]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

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
                              use_bn=use_bn, dropout_prob=dropout_prob, leaky_relu=leaky_relu)

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
            # print("Spatial emb", spatial_embeddings.shape, "new input", new_input.shape, "coords", coords.shape)
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
        clamped_temp_sigmoid = 1 + 4*self.sigmoid(self.temp_sigmoid)   #10 + 99 * self.sigmoid(self.temp_sigmoid) # try with a smaller range

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
        clamped_temp_sigmoid = 10 + 99 * torch.sigmoid(self.temp_sigmoid) # constrain the temp_sigmoid between 10 and 100
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
# EXPERIMENTAL: BINN Hybrid: process-based model predicts SOC, but NN can correct it
#---------------------------------------------------
# define model
class BINN_Hybrid(nn.Module):
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
            self.mlp = lipmlp((self.new_input_size, 256, 256, self.num_params*2),
                              use_bn=use_bn, dropout_prob=dropout_prob)  # TODO different initialization methods, leaky relu, dropout, etc. not supported
        else:
            self.mlp = mlp((self.new_input_size, 256, 256, self.num_params*2),
                              use_bn=use_bn, dropout_prob=dropout_prob, leaky_relu=leaky_relu)

        # Use second half of MLP output to directly predict residual (error)
        # of process-based model
        self.predict_residual = nn.Linear(self.num_params, 140)

        # sigmoid parameter
        self.temp_sigmoid = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.sigmoid = nn.Sigmoid()

        # LibMTL specific
        self.rep_grad = rep_grad  # Whether to compute gradients w.r.t. parameters as well
        self.task_name = losses
        self.task_num = len(self.task_name)
        self.device = device


    def forward(self, input_var, wosis_depth, coords, whether_predict,
                return_residual=False):
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
            # print("Spatial emb", spatial_embeddings.shape, "new input", new_input.shape, "coords", coords.shape)
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
        clamped_temp_sigmoid = 10 + 99*self.sigmoid(self.temp_sigmoid)   #10 + 99 * self.sigmoid(self.temp_sigmoid) # try with a smaller range

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





#---------------------------------------------------
# Pure NN without process-based model
#---------------------------------------------------
class nn_only(nn.Module):
    def __init__(self, input_vars, var_idx_to_emb, pos_enc, output_dim=140,
                 lipschitz=False, one_hot=False, use_bn=False, dropout_prob=0.0,
                 leaky_relu=False, rep_grad=False,
                 losses=["l1", "param_reg"], device="cpu", min_val=None, max_val=None):
        super().__init__()
        print("NN only")

        # If one_hot is True, this is a Dict from categorical variable index to number of categories.
        # If one_hot is False, this is a Dict from categorical variable index -> Embedding layer we use
        self.one_hot = one_hot
        self.var_idx_to_emb = var_idx_to_emb
        self.pos_enc = pos_enc
        self.output_dim = output_dim

        # List of non-categorical variable indices
        self.non_categorical_indices = list(set(list(range(input_vars))).difference(var_idx_to_emb.keys()))
        self.new_input_size = len(self.non_categorical_indices)
        for idx, emb in self.var_idx_to_emb.items():
            if self.one_hot:
                self.new_input_size += emb
            else:
                self.new_input_size += emb.embedding_dim

        # Number of parameters (totally fake)
        self.num_params = 21

        # Output transformation
        self.min_val = min_val
        self.max_val = max_val

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
                              use_bn=use_bn, dropout_prob=dropout_prob, leaky_relu=leaky_relu)
        self.final_layer = nn.Linear(self.num_params, self.output_dim)
        nn.init.xavier_uniform_(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)

        # sigmoid parameter
        self.temp_sigmoid = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.sigmoid = nn.Sigmoid()



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
            # print("Spatial emb", spatial_embeddings.shape, "new input", new_input.shape, "coords", coords.shape)
            new_input = torch.concatenate([spatial_embeddings, new_input], dim=1)
        elif self.pos_enc == "late":
            raise ValueError("Late pos_enc not supported for nn_only")

        # check if new_input is nan
        if torch.isnan(new_input).any() or torch.isinf(new_input).any():
            print("new_input was nan", new_input)
            exit(1)

        # Pass through MLP to get fake pred_para
        pred_para = self.mlp(new_input)
        pred_output = self.final_layer(F.relu(pred_para))
        if self.min_val is not None and self.max_val is not None:
            pred_output = pred_output*(self.max_val-self.min_val) + self.min_val

        # Convert 140 pools to 20 layers
        pred_output = pred_output.reshape((pred_output.shape[0], 20, 7)).mean(dim=2)

        if whether_predict == 1:
            return pred_output, self.sigmoid(pred_para)
        else:
            simu_soc = select_depth(pred_output, forcing, obs_depth)
            return simu_soc, self.sigmoid(pred_para)


#---------------------------------------------------
# EXPERIMENTAL: Use PEGCN as encoder for BINN model
#---------------------------------------------------
class GNN_BINN(nn.Module):
    def __init__(self, input_vars, var_idx_to_emb, vertical_mixing, pos_enc,
                 k=20, one_hot=False, use_bn=False, dropout_prob=0.0,
                 leaky_relu=False, rep_grad=False, graph_conv='gcn',
                 gnn_input_dim=256, emb_hidden_dim=128,
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
        self.initial_fc = mlp((self.new_input_size, 256, 32), use_bn=use_bn, dropout_prob=dropout_prob, leaky_relu=leaky_relu)
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
        if leaky_relu:
            self.relu = nn.LeakyReLU(negative_slope=0.3)
        else:
            self.relu = torch.nn.ReLU()

        # sigmoid parameter
        self.temp_sigmoid = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.sigmoid = nn.Sigmoid()

        # LibMTL specific
        self.rep_grad = rep_grad  # Whether to compute gradients w.r.t. parameters as well
        self.task_name = losses
        self.task_num = len(self.task_name)
        self.device = device


    def forward(self, input_var, wosis_depth, coords,
                 whether_predict, ei=None, ew=None, plot_dir=None, return_extra=False):
        predictor = input_var[:, :, 0, 0]
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
                emb = 0.1*F.one_hot(predictor[:, idx].long(), num_classes=embedding_layer)  # if one_hot, "embedding_layer" is simply the number of classes
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
        x = self.relu(self.initial_fc(x))
        h1 = self.relu(self.conv1(x, edge_index, edge_weight))
        h1 = self.dropout(h1)
        if self.conv2 is not None:
            h1 = self.relu(self.conv2(h1, edge_index, edge_weight))
            h1 = self.dropout(h1)
        gnn_output = self.fc(h1)

        # Clamp temp_sigmoid to be between 10 and 200
        clamped_temp_sigmoid = 10 + 99 * self.sigmoid(self.temp_sigmoid) # try with a smaller range

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



#---------------------------------------------------
# EXPERIMENTAL: Simple spatial smoothing on predicted params.
# Positional encoding outputs spatially-correlated errors,
# use penalty to encourage spatial smoothness
#---------------------------------------------------
class Spatial_BINN(nn.Module):
    def __init__(self, input_vars, var_idx_to_emb, vertical_mixing, pos_enc,
                 k=20, one_hot=False, rep_grad=False,
                 use_bn=False, dropout_prob=0.0, leaky_relu=False, emb_hidden_dim=128,
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

        # Fully-connected
        self.mlp = mlp((self.new_input_size, 256, 256, self.num_params),
                        use_bn=use_bn, dropout_prob=dropout_prob, leaky_relu=leaky_relu)

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
        self.sigmoid = nn.Sigmoid()

        # LibMTL specific
        self.rep_grad = rep_grad  # Whether to compute gradients w.r.t. parameters as well
        self.task_name = losses
        self.task_num = len(self.task_name)
        self.device = device


    def forward(self, input_var, wosis_depth, coords,
                whether_predict, ei=None, ew=None,
                plot_dir=None, return_extra=False):
        predictor = input_var[:, :, 0, 0]
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
                emb = 0.1*F.one_hot(predictor[:, idx].long(), num_classes=embedding_layer)  # if one_hot, "embedding_layer" is simply the number of classes
            else:
                emb = embedding_layer(predictor[:, idx].int())
                emb = F.normalize(emb, p=2, dim=1)  # New @joshuafan: normalize embeddings
            embs.append(emb)
        all_embs = torch.concatenate(embs, dim=1)
        new_input = torch.concatenate([predictor[:, self.non_categorical_indices], all_embs], dim=1)

        # Pass feature vectors through MLP to get initial predicitions
        mlp_output = self.mlp(new_input)

        # Clamp temp_sigmoid to be between 10 and 200
        clamped_temp_sigmoid = 10 + 99 * self.sigmoid(self.temp_sigmoid) # try with a smaller range

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

