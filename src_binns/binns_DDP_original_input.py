"""
Main training script for BINN, using DistributedDataParallel.

See run_binn_interactive.sh for example usage with reasonable hyperparameters.
"""
import csv
import functools
import math
import sys
import time
import random
import warnings
import subprocess
import argparse
from collections import OrderedDict
import misc_utils
from sklearn.model_selection import KFold
from mlp import ConstantParameters
from pe_gcn_model import GridCellSpatialRelationEncoder
from torch.optim.swa_utils import AveragedModel, SWALR
from spatial_utils import *
from losses import binns_loss, compute_param_matching_loss, compute_param_violation_loss, compute_unconstrained_param_loss
import visualization_utils

# sys.path.append('C:/Users/hx293/Research_Data/BINN/')
# sys.path.append('/glade/u/home/haodixu/BINN')
# sys.path.append(r'/User/homes/ftao/Projects/BINNS/src_binns')
sys.path.append('/glade/work/haodixu/BINN')

# Set HDF5_DISABLE_VERSION_CHECK to suppress version mismatch error
import os
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2'

from datetime import datetime, timedelta
import pandas as pd
from pandas import DataFrame as df
import numpy as np
from scipy.interpolate import pchip_interpolate

print("Start binns_DDP")

# @joshuafan: previously we set default dtype to float64 to avoid underflow in process-based model.
# Now checking float32 with fixed process-based model.
torch.set_default_dtype(torch.float32)

# Temporary hack to avoid printing np.float64(...) when printing out numpy scalars.
# TODO fix this
np.set_printoptions(legacy="1.21")

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import multiprocessing
# from torch.multiprocessing import Process  # TODO CHECK
from multiprocessing import Process
from scipy.io import loadmat
import netCDF4 as ncread 
import mat73
from matplotlib import pyplot as plt

# LibMTL is a library for advanced multi-task loss weighting methods.
# Commenting these out as they are not essential for BINN training.
# import LibMTL.weighting as weighting_method
# import LibMTL.architecture as architecture_method

###################################
# Import CLM5 process-based model #
###################################
# fun_model_simu predicts at user-specified depths. fun_model_prediction predicts at 20 default layers.
from fun_matrix_COMPAS_Hardy import fun_model_simu, fun_model_prediction

################################################
# Command-line arguments
################################################
parser = argparse.ArgumentParser()

# Model architecture
parser.add_argument("--note", type=str, default="", help="Optional name to give to the model")
parser.add_argument("--model", type=str, default="old_mlp", choices=['old_mlp', 'new_mlp', 'lipmlp', 'senn', 'nam', 'nam_joint', 'nam_joint2', 'nag', 'kan', 'gnn', 'spatial', 'nn_only', 'binn_hybrid'], help="Model type")
parser.add_argument("--width", type=int, default=128, help="Size of hidden layers (new_mlp or nn_only)")
parser.add_argument("--num_layers", type=int, default=4, help="Size of hidden layers (new_mlp or nn_only)")
parser.add_argument("--residual", action='store_true', help="Whether to add residual connections in neural network portion (MLP)")
parser.add_argument("--categorical", type=str, default="embedding", choices=["embedding", "one_hot"], help="How to embed categorical variables")
parser.add_argument("--embed_dim", type=int, default=5, help="Embedding dim for each categorical variable (if using embeddings)")
parser.add_argument("--use_bn", action='store_true', help="Whether to use batchnorm")
parser.add_argument("--dropout_prob", default=0., type=float, help="Dropout prob")
parser.add_argument("--feature_dropout", default=0., type=float, help="Probability of dropping out entire feature. ONLY SUPPORTED FOR NAM MODELS.")
parser.add_argument("--activation", type=str, choices=['relu', 'leaky_relu', 'tanh', 'exu'], default='relu', help="Activation function inside neural network. exu is only supported for NAM (neural additive model)")
parser.add_argument("--param_constraint", type=str, choices=['sigmoid', 'hardsigmoid', 'none'], default='sigmoid', help="Activation function used to constrain parameter predictions. If sigmoid, we suggest using param_reg loss. If hardsigmoid, use param_violation loss")

# KAN specific
parser.add_argument("--kan_grid", type=int, default=3, help="Number of grid intervals in KAN")
parser.add_argument("--kan_update_grid", type=int, default=1, help="Whether to update grids for KAN every epoch (default true)")
parser.add_argument("--kan_grid_margin", type=float, default=1.0, help="How much margin to use (in units of input range) when creating grids for KAN. Only used if kan_update_grid is 1.")
parser.add_argument("--kan_noise", type=float, default=0.3, help="Noise scale for KAN")
parser.add_argument("--kan_base_fun", type=str, default="silu", choices=["silu", "identity"], help="Base function for KAN")
parser.add_argument("--kan_affine_trainable", action='store_true')

# Process-based model settings
parser.add_argument("--vertical_mixing", type=str, default='original', choices=['original', 'simple_one_intercept', 'simple_two_intercepts'], help="""Vertical mixing matrix parameterization. Original explicitly models diffusion.
						 simple_one_intercept approximates with a log-log relationship with depth (upwards/downwards
						 having the same intercept). simple_two_intercepts allows upwards/downwards transfers to
						 have different intercepts.""")
parser.add_argument("--vectorized", type=str, default='yes', choices=['yes', 'no', 'compare'], help="""yes to use vectorized version of process-based model,
						 no to use old for-loop version, compare to run both and assert they produce the same result""")
parser.add_argument("--para_to_predict", type=str, default="all", choices=["all", "four", "fifteen"], help="Which parameters to predict using NN. If 'four', the NN only predicts the four most sensitive parameters, and other parameters are prescribed to PRODA-predicted values.")

# Sigmoid temp and initialization
parser.add_argument("--min_temp", type=float, default=10., help="Min temp for sigmoid")
parser.add_argument("--max_temp", type=float, default=109., help="Max temp for sigmoid")
parser.add_argument("--init", type=str, default="xavier_uniform", choices=["default", "xavier_uniform", "kaiming_uniform"], help="Initialization for weights. For xavier_uniform/kaiming_uniform, biases are initialized to zero.")

# Data split
parser.add_argument("--data_seed", type=int, default=-1, help="Random seed for splitting data. -1 means use same as args.seed")
parser.add_argument("--n_datapoints", type=int, default=-1, help="Set this to train on a random subset of this many datapoints (for train+val+test). -1 to use the whole dataset")
parser.add_argument("--representative_sample", action='store_true', help="Whether to restrict to ~1000 'representative profiles'")
parser.add_argument("--cross_val_idx", type=int, default=0, help="""Cross-validation index. 0 means no cross-validation (fixed train/val/test split), 
							while a number between 1 and k means to use that index's fold (1-based). Note that this script only runs one fold;
							you need to manually combine results from multiple folds.""")
parser.add_argument("--n_folds", type=int, default=10, help="Number of folds if using cross-validation")
parser.add_argument("--split", type=str, default='random', choices=['random', 'horizontal', 'vertical', 'us_vs_world', 'grid2'],
					help="""How to split val/test sets. If `random`, just hold out random examples (cross-validation or fixed split).
						 If horizontal or vertical, split into 10 horizontal or vertical folds; only cross-validation is supported.
						 If us_vs_world, use US as train/val sets and rest-of-world as test set; cross-validation is not supported (only fixed split).""")
parser.add_argument("--val_ratio", type=float, default=0.1, help="Fraction of datapoints in validation set. Only used if not doing cross-validation.")
parser.add_argument("--test_ratio", type=float, default=0.1, help="Fraction of datapoints in test set. Only used if not doing cross-validation.")
parser.add_argument("--batching", type=str, default='random', choices=['random', 'block'],
					help='How to generate minibatches. If `block`, samples examples from contiguous spatial block for each batch.')
parser.add_argument("--synthetic_labels", action='store_true', help="Whether to use synthetic SOC labels (generated from running CLM5 on PRODA parameters)")

# Transformations
parser.add_argument("--standardize_input", action='store_true', help="If set, standardize numeric features to mean 0, std 1. Otherwise, features vary between 0 and 1.")
parser.add_argument("--standardize_output", action='store_true', help="ONLY APPLIES IF MODEL IS NN_ONLY. If set, standardize outputs (labels) to mean 0, std 1.")

# Training
parser.add_argument("--seed", type=int, default=0, help="Random seed for model initialization")
parser.add_argument("--optimizer", type=str, choices=["SGD", "AdamW"], default="AdamW")
parser.add_argument("--scheduler", type=str, choices=["none", "reduce_on_plateau", "step", "cosine"], default="none")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum (SGD ONLY)")
parser.add_argument("--noise_std", type=float, default=0., help="How much noise to add to NN weights during each optimizer step")
parser.add_argument("--jacobian_noise_std", type=float, default=0., help="If set, compute the Jacobian at perturbed inputs.")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--bias_only_epochs", type=int, default=0, help="Number of epochs where we train ONLY FINAL-LAYER BIAS. This helps initialize params to a good value globally.")
parser.add_argument("--patience", type=int, default=20)
parser.add_argument("--one_param_only", action='store_true', help='If set, target updating only one param per batch')
parser.add_argument("--save_freq", type=int, default=5, help="How often (epochs) to save the latest checkpoint, in case the job crashes")

# Regularization
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--use_swa", action='store_true', help="Whether to use Stochastic Weight Averaging")
parser.add_argument("--clip_value", type=float, default=-1, help="Clip value for gradient clipping. -1 for no clipping.")

# Losses and loss weights
parser.add_argument("--losses", nargs="+", choices=["l1_loss", "smooth_l1", "l2_loss", "Smooth_l1_loss_SOC", "Smooth_l1_loss_POM", "Smooth_l1_loss_MAOM", "param_reg", "param_violation", "unconstrained_param", "param_matching", "jacobian",
													"jacobian_sparsity", "spectral", "lipmlp", "cure", "senn_robustness", "senn_l1", "senn_sparsity", 
													"nam_l2", "nam_entropy", "kan_l1", "kan_entropy", "kan_coef", "kan_coefdiff", "kan_entropy_output", "kan_entropy_input", "kan_output_var",
													"spatial_error", "spatial_emb_smoothness", "param_smoothness", "residual"], default=["smooth_l1", "param_reg"],
					help="Losses to use (can list any number). Note jacobian_sparsity cannot be optimized (non-differentiable): it is just something we track.")
parser.add_argument("--loss_weighting", default="manual", choices=["manual", "relobralo", "IMTL", "two_stage"])
parser.add_argument("--lambdas", nargs="+", type=float, default=[1.0, 10.0], help="If loss_weighting is manual, provide weights in the same order as `args.losses`")
parser.add_argument("--second_start", type=int, default=30, help="If loss_weighting is two_stage, epoch the second phase starts")
parser.add_argument("--second_lambdas", nargs="+", type=float, default=[1.0, 10.0], help="If loss_weighting is two_stage, weights for the second stage - in the same order as `args.losses`")

# Relobralo specific hyperparams (specific method of loss balancing: only used if you set `--loss_weighting relobralo`)
parser.add_argument("--relobralo_alpha", type=float, default=0.9, help="Exponential decay rate for Relobralo")
parser.add_argument("--relobralo_temp", type=float, default=0.1, help="Softmax temperature for Relobralo")
parser.add_argument("--relobralo_saudade", type=float, default=0.999, help="Saudade (1 minus probability of looking back to epoch 0)")

# Positional encoding / GNN
parser.add_argument("--features", type=str, choices=["all", "all_including_lonlat", "ten"], default="all", help="Which features to use. Default `all` includes all features except lon/lat. To include lon/lat explicitly, use `all_including_lonlat`. `ten` is 10 handcrafted features")
parser.add_argument("--pos_enc", type=str, default='none', choices=['none', 'early', 'late'],
					help="How lon/lat features are encoded. 'none' means not used. 'early' means that positional encoding is concatenated with other features. 'late' means that it is only used as an error term for the latent parameters.")
parser.add_argument("--graph_conv", type=str, default="gcn", choices=["gcn", "gat", "gcn1", "gat1"], help="For GNN, which graph conv to use")
parser.add_argument("--k", type=int, default=20, help="Nearest neighbors for graph (GNN/Spatial only)")

# Computational environment
parser.add_argument("--use_ddp", type=int, default=1, help="Whether to use DDP")
parser.add_argument("--num_CPU", type=int, default=32, help="Number of processes for Torch DDP")
parser.add_argument("--job_scheduler", type=str, default="pbs", choices=["pbs", "slurm"], help="Job scheduler system. Job scheduler. PBS for NCAR computers, slurm for AIDA server.")
parser.add_argument("--whether_resume", type=int, default=0, help="Whether to resume training from a previous model")
parser.add_argument("--previous_job_id", type=str, default="", help="Previous job id to resume from (otherwise, resumes using envir variable PREVIOUS_JOB_ID)")
parser.add_argument("--time_limit", type=float, default=11.5, help="Time limit for this job in HOURS. If trainng is not finished yet, start another job to continue.")
parser.add_argument("--plot", action='store_true', help="Whether to produce plots.")

args = parser.parse_args()

# If loss_weighting is manual, make sure the correct number of lambdas were provided
if args.loss_weighting in ["manual", "relobralo", "two_stage"]:
	assert(len(args.losses) == len(args.lambdas))
	args.lambdas = torch.tensor(args.lambdas)
	args.second_lambdas = torch.tensor(args.second_lambdas)


def set_seeds(seed):
	"""
	Attempts to set all random seeds to improve reproducibility.
	"""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True

# Set seeds for reproducibility
set_seeds(args.seed)
if args.data_seed == -1:
	args.data_seed = args.seed

# Keep track of when the job started
job_begin_time = time.time()

print(datetime.now(), '------------all packages loaded------------')
#################################################################################
# ATTENTION: There is a lot of code that is not inside any function,
# which sets up datasets. On CPU (start_method=fork), this gets run
# ONCE, and the forked processes have access to all variables created.
# On GPU (start_method=spawn), this gets run once initially, and AGAIN
# for each process. It would be nice to factor this out eventually.
################################################################################# 

################################################
# Data Directories (CHANGE THIS!!!)
################################################
# data_dir_input = '/Users/phoenix/Google_Drive/Tsinghua_Luo/Projects/DATAHUB/ENSEMBLE/INPUT_DATA/'
# data_dir_output = '/Users/phoenix/Google_Drive/Tsinghua_Luo/Projects/DATAHUB/BINNS/OUTPUT_DATA/'
# data_dir_input = 'C:/Users/hx293/Research_Data/BINN/ENSEMBLE/INPUT_DATA/'
# data_dir_output = 'C:/Users/hx293/Unsync_Data/BINN_output/'
# server path
job_submit_path = '/glade/u/home/haodixu/BINN/PBS_Submit/KAN_COMPAS2/'
data_dir_input = '/glade/u/home/haodixu/BINN/ENSEMBLE/INPUT_DATA/'
data_dir_output = '/glade/work/haodixu/BINN/BINNS/OUTPUT_DATA/'
os.makedirs(job_submit_path, exist_ok=True)

################################################
# Setup datasets
################################################
cesm2_case_name = 'sasu_f05_g16_checked_step4'
start_year = 661
end_year = 680

time_domain = 'whole_time' # 'whole_time', 'before_1985', 'after_1985', 'random_half_1', 'random_half_2'
model_name = 'cesm2_clm5_cen_vr_v2'

start_id = 1
end_id = 5000
is_resubmit = 0

# constants
month_num = 12 
soil_cpool_num = 7
soil_decom_num = 20

#-------------------------------
# Load wosis data
#-------------------------------
# The site information for each SOC profile. 
# Names for each column are "profile_id" "country_id" "country_name" "lon" "lat" "layer_num" "date". 
nc_data_middle = ncread.Dataset(data_dir_input + 'wosis_2019_snap_shot/soc_profile_wosis_2019_snapshot_hugelius_mishra.nc')  # wosis profile info
wosis_profile_info = nc_data_middle['soc_profile_info'][:].data.transpose()
nc_data_middle.close()

# The full dataset which contains SOC content information at each layer
# layer_info: "profile_id, date, upper_depth, lower_depth, node_depth, soc_layer_weight, soc_stock, bulk_denstiy, is_pedo"
nc_data_middle = ncread.Dataset(data_dir_input + 'wosis_2019_snap_shot/soc_data_integrate_wosis_2019_snapshot_hugelius_mishra.nc')  # wosis SOC info
wosis_soc_info = nc_data_middle['data_soc_integrate'][:].data.transpose()
nc_data_middle.close()

#-------------------------------
# POM and MAOM data
#-------------------------------
# Load POM and MAOM data from excel file
data_POM_MAOM = pd.read_excel(data_dir_input + 'POM_MAOM/new-data-no wetland and bareland-80-120.xlsx')
# Drop the unit row 
data_POM_MAOM = data_POM_MAOM.drop(0)
# Store the shape of the data
POM_MAOM_profile_num = data_POM_MAOM.shape
print("Shape of data_POM_MAOM: ", POM_MAOM_profile_num)


#-------------------------------
# COMPAS constants
#-------------------------------
# Parameter names
if args.vertical_mixing == 'original':
	para_names = ['diffus', 'cryo', 'q10', 'efolding', 'taucwd', 'taul1', 'taul2', 'tau4doc', 'tau4mic', 'tau4poc', 'tau4maom','fl1_MIC', 'fl2_MIC', 'fMIC_DOC', 'fMIC_POC', 'fDOC_MAOM', 'CUEl1', 'CUEl2', 'CUEDOC', 'w-scaling', 'beta']
else:
	# If using the simpler vertical mixing parameterization, replace diffus/cryo with slope/intercept.
	para_names = ['slope', 'intercept', 'q10', 'efolding', 'taucwd', 'taul1', 'taul2', 'tau4s1', 'tau4s2', 'tau4s3', 'fl1s1', 'fl2s1', 'fl3s2', 'fs1s2', 'fs1s3', 'fs2s1', 'fs2s3', 'fs3s1', 'fcwdl2', 'w-scaling', 'beta']
	if args.vertical_mixing == 'simple_two_intercepts':
		para_names.append('intercept_leach')

# Parameter indices that the neural network predicts. Usually we predict all the parameters, but for
# the retrieval test we may prescribe some and only predict 4 or 15 most sensitive parameters. 
if args.para_to_predict == "all":  # All parameters
	para_index = np.arange(0, len(para_names))
elif args.para_to_predict == "four":
	assert len(para_names) == 21
	para_index = np.array([3, 9, 14, 19])
elif args.para_to_predict == "fifteen":
	assert len(para_names) == 21
	para_index = np.array([0, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20])
else:
	raise ValueError("Invalid value of --para_to_predict")

# Soil depths info
# width between two interfaces
dz = np.array([2.000000000000000E-002, 4.000000000000000E-002, 6.000000000000000E-002, \
8.000000000000000E-002, 0.120000000000000, 0.160000000000000, \
0.200000000000000, 0.240000000000000, 0.280000000000000, \
0.320000000000000, 0.360000000000000, 0.400000000000000, \
0.440000000000000, 0.540000000000000, 0.640000000000000, \
0.740000000000000, 0.840000000000000, 0.940000000000000, \
1.04000000000000, 1.14000000000000, 2.39000000000000, \
4.67553390593274, 7.63519052838329, 11.1400000000000, \
15.1154248593737])

# depth of the interface
zisoi = np.array([2.000000000000000E-002, 6.000000000000000E-002, \
0.120000000000000, 0.200000000000000, 0.320000000000000, \
0.480000000000000, 0.680000000000000, 0.920000000000000, \
1.20000000000000, 1.52000000000000, 1.88000000000000, \
2.28000000000000, 2.72000000000000, 3.26000000000000, \
3.90000000000000, 4.64000000000000, 5.48000000000000, \
6.42000000000000, 7.46000000000000, 8.60000000000000, \
10.9900000000000, 15.6655339059327, 23.3007244343160, \
34.4407244343160, 49.5561492936897])

zisoi_0 = 0

# depth of the node
zsoi = np.array([1.000000000000000E-002, 4.000000000000000E-002, 9.000000000000000E-002, \
0.160000000000000, 0.260000000000000, 0.400000000000000, \
0.580000000000000, 0.800000000000000, 1.06000000000000, \
1.36000000000000, 1.70000000000000, 2.08000000000000, \
2.50000000000000, 2.99000000000000, 3.58000000000000, \
4.27000000000000, 5.06000000000000, 5.95000000000000, \
6.94000000000000, 8.03000000000000, 9.79500000000000, \
13.3277669529664, 19.4831291701244, 28.8707244343160, \
41.9984368640029])

# depth between two node
dz_node = zsoi - np.append(np.array([0]), zsoi[:-1], axis = 0)


# cesm2 resolution
cesm2_resolution_lat = 180/384
cesm2_resolution_lon = 360/576
lon_grid = np.arange((-180 + cesm2_resolution_lon/2), 180, cesm2_resolution_lon)
lat_grid = np.arange((90 - cesm2_resolution_lat/2), -90, -cesm2_resolution_lat)

# load cesm2 input
var_name_list = ['nbedrock', 'ALTMAX', 'ALTMAX_LASTYEAR', 'CELLSAND', 'NPP', \
	'SOILPSI', 'TSOI', \
	'W_SCALAR', 'T_SCALAR', 'O_SCALAR', 'FPI_vr', \
	'LITR1_INPUT_ACC_VECTOR', 'LITR2_INPUT_ACC_VECTOR', 'LITR3_INPUT_ACC_VECTOR', 'CWD_INPUT_ACC_VECTOR', \
	'TOTSOMC']

var_name_list_rename =  ['cesm2_simu_nbedrock', 'cesm2_simu_altmax', 'cesm2_simu_altmax_last_year', 'cesm2_simu_cellsand', 'cesm2_simu_npp', \
	'cesm2_simu_soil_water_potnetial', 'cesm2_simu_soil_temperature', \
	'cesm2_simu_w_scalar', 'cesm2_simu_t_scalar', 'cesm2_simu_o_scalar', 'cesm2_simu_n_scalar', \
	'cesm2_simu_input_vector_litter1', 'cesm2_simu_input_vector_litter2', 'cesm2_simu_input_vector_litter3', 'cesm2_simu_input_vector_cwd', \
	'cesm2_simu_soc_stock']

for ivar in np.arange(0, len(var_name_list)):
	# load simulation from CESM2
	var_record_monthly_mean = mat73.loadmat(data_dir_input + 'cesm2_simu/spinup_ss/' + cesm2_case_name + '_cesm2_ss_4da_' + str(start_year) + '_' + str(end_year) + '_' + var_name_list[ivar] + '.mat')
	var_record_monthly_mean = var_record_monthly_mean['var_record_monthly_mean']
	exec(var_name_list_rename[ivar] + ' = var_record_monthly_mean')
# end

for ilayer in np.arange(0, soil_decom_num):
	cesm2_simu_input_vector_litter1[:, :, ilayer, :] = cesm2_simu_input_vector_litter1[:, :, ilayer, :]*dz[ilayer]
	cesm2_simu_input_vector_litter2[:, :, ilayer, :] = cesm2_simu_input_vector_litter2[:, :, ilayer, :]*dz[ilayer]
	cesm2_simu_input_vector_litter3[:, :, ilayer, :] = cesm2_simu_input_vector_litter3[:, :, ilayer, :]*dz[ilayer]
	cesm2_simu_input_vector_cwd[:, :, ilayer, :] = cesm2_simu_input_vector_cwd[:, :, ilayer, :]*dz[ilayer]
#end

cesm2_simu_input_sum_litter1 = np.sum(cesm2_simu_input_vector_litter1, axis = 2)
cesm2_simu_input_sum_litter2 = np.sum(cesm2_simu_input_vector_litter2, axis = 2)
cesm2_simu_input_sum_litter3 = np.sum(cesm2_simu_input_vector_litter3, axis = 2)
cesm2_simu_input_sum_cwd = np.sum(cesm2_simu_input_vector_cwd, axis = 2)

del cesm2_simu_input_vector_litter1, cesm2_simu_input_vector_litter2, cesm2_simu_input_vector_litter3, cesm2_simu_input_vector_cwd


############################################
# Select subset of observations (profiles) #
############################################
# Representative points
sample_profile_id = loadmat(data_dir_input + 'POM_MAOM/eligible_profile_loc_1_cesm2_clm5_cen_vr_v2_whole_time.mat')

# NOTE: Not sure why "sample_profile_id" shape is [100, 50] before flattening?
sample_profile_id = sample_profile_id['eligible_loc_1']

# Representative points
# sample_profile_id = loadmat(data_dir_input + 'wosis_2019_snap_shot/wosis_2019_snapshot_hugelius_mishra_representative_profiles.mat')

# # NOTE: Not sure why "sample_profile_id" shape is [100, 50] before flattening?
# sample_profile_id = sample_profile_id['sample_profile_id'].flatten()

# convert the number to be starting from 0 in python world
sample_profile_id = sample_profile_id - 1

profile_collection = sample_profile_id

# # Choose the profile id with lat and lon within the range of the United States
# profile_collection = np.where(
#     (wosis_profile_info[:, 2] == 156) & 
# 	(wosis_profile_info[:, 3] >= -124.763068) & 
#     (wosis_profile_info[:, 3] <= -66.949895) & 
#     (wosis_profile_info[:, 4] >= 24.5) & 
#     (wosis_profile_info[:, 4] <= 49.384358)
# )[0]

# Reshape the profile collection to be a column vector
profile_collection = profile_collection.flatten()
# profile_collection = np.reshape(profile_collection, [profile_collection.shape[0], 1])

profile_collection = np.random.choice(profile_collection, 30000, replace=False)

profile_collection = np.reshape(profile_collection, [profile_collection.shape[0], 1])

print("Shape of profile_collection for SOC: ", profile_collection.shape)

# Assign profile id to POM and MAOM data, starting from the end of the WOSIS data
data_POM_MAOM['profile_id'] = wosis_profile_info.shape[0] + np.arange(0, POM_MAOM_profile_num[0])
# Get the length of the combined dataset
combined_profile_num = wosis_profile_info.shape[0] + POM_MAOM_profile_num[0]
print("Shape of combined dataset: ", combined_profile_num)

# Add the POM and MAOM data to the profile collection
# profile_collection = np.append(profile_collection, data_POM_MAOM['profile_id'].values)
# profile_collection = np.append(profile_collection, np.random.choice(data_POM_MAOM['profile_id'].values, 1000, replace=False))

# # Select profiles within the US
# profile_collection_POM_MAOM = np.where(
# 		(data_POM_MAOM['Longitude'] >= -125) & 
# 		(data_POM_MAOM['Longitude'] <= -66) &
# 		(data_POM_MAOM['Latitude'] >= 24.5) &
# 		(data_POM_MAOM['Latitude'] <= 49.5)
# )[0]

# if len(profile_collection_POM_MAOM) > 1000:
# 	profile_collection_POM_MAOM = np.random.choice(profile_collection_POM_MAOM, 1000, replace=False)

# # Select profiles within the US
# profile_collection_POM_MAOM = np.where(
# 		(data_POM_MAOM['Longitude'] >= -125) & 
# 		(data_POM_MAOM['Longitude'] <= -66) &
# 		(data_POM_MAOM['Latitude'] >= 24.5) &
# 		(data_POM_MAOM['Latitude'] <= 49.5)
# )[0]

# profile_collection = np.append(profile_collection, data_POM_MAOM['profile_id'].values[profile_collection_POM_MAOM])

# Add the POM and MAOM data to the profile collection
profile_collection = np.append(profile_collection, data_POM_MAOM['profile_id'].values)

profile_collection = np.reshape(profile_collection, [profile_collection.shape[0], 1])

profile_range = np.arange(0, len(profile_collection))

print('number of profiles: ', len(profile_collection))

print(datetime.now(), '------------all input data loaded------------')

#---------------------------------------------------
# wrap up soc data for NN
#---------------------------------------------------
obs_soc_matrix = np.ones([len(profile_collection), 200])*np.nan  # Each row is a profile. Each non-nan column is an SOC observation
# POM and MAOM data
obs_POM_matrix = np.ones([len(profile_collection), 200])*np.nan
obs_MAOM_matrix = np.ones([len(profile_collection), 200])*np.nan
obs_depth_matrix = np.ones([len(profile_collection), 200])*np.nan  # Each row is a profile. Each column represents the depth of the corresponding SOC observation in "obs_soc_matrix"
obs_upper_depth_matrix = np.ones([len(profile_collection), 200])*np.nan  # Each row is a profile. Each column represents the upper depth of the corresponding SOC observation in "obs_soc_matrix"
obs_lower_depth_matrix = np.ones([len(profile_collection), 200])*np.nan  # Each row is a profile. Each column represents the lower depth of the corresponding SOC observation in "obs_soc_matrix"
obs_lon_lat_loc = np.ones([len(profile_collection), 2])*np.nan  # Each row is a profile. First column is longitude, second column is latitude

model_force_input_vector_cwd = np.ones([len(profile_collection), month_num])*np.nan
model_force_input_vector_litter1 = np.ones([len(profile_collection), month_num])*np.nan
model_force_input_vector_litter2 = np.ones([len(profile_collection), month_num])*np.nan
model_force_input_vector_litter3 = np.ones([len(profile_collection), month_num])*np.nan

model_force_altmax_lastyear_profile = np.ones([len(profile_collection), month_num])*np.nan
model_force_altmax_current_profile = np.ones([len(profile_collection), month_num])*np.nan
model_force_nbedrock = np.ones([len(profile_collection), month_num])*np.nan

model_force_xio = np.ones([len(profile_collection), soil_decom_num, month_num])*np.nan
model_force_xin = np.ones([len(profile_collection), soil_decom_num, month_num])*np.nan

model_force_sand_vector = np.ones([len(profile_collection), soil_decom_num, month_num])*np.nan

model_force_soil_temp_profile = np.ones([len(profile_collection), soil_decom_num, month_num])*np.nan
model_force_soil_water_profile = np.ones([len(profile_collection), soil_decom_num, month_num])*np.nan

# record the sum of recorded layers for all profiles
layer_num_record = 0

for iprofile_hat in profile_range:
	# profile num
	iprofile = profile_collection[iprofile_hat]
	# profile id
	if iprofile < wosis_profile_info.shape[0]:
		profile_id = wosis_profile_info[iprofile, 0]
		# find currently using profile
		loc_profile = np.where(wosis_soc_info[:, 0] == profile_id)[0]
		# find the lon and lat info of soil profile
		lon_profile = wosis_profile_info[iprofile, 3]
		lat_profile = wosis_profile_info[iprofile, 4]
		
		lat_loc = np.where(abs(lat_profile - lat_grid) == min(abs(lat_profile - lat_grid)))[0][0]
		lon_loc = np.where(abs(lon_profile - lon_grid) == min(abs(lon_profile - lon_grid)))[0][0]
		
		# info of the node depth of profile  
		wosis_layer_depth = wosis_soc_info[loc_profile, 4]
		# observed C info (gC/m3)
		wosis_layer_obs = wosis_soc_info[loc_profile, 6]
		# check how many layers are recorded
		layer_num_record = layer_num_record + len(wosis_layer_obs)
		# observced upper depth of each layer
		wosis_layer_upper_depth = wosis_soc_info[loc_profile, 2]
		# observced lower depth of each layer
		wosis_layer_lower_depth = wosis_soc_info[loc_profile, 3]
		# exclude nan values
		valid_soc_loc = np.where((np.isnan(wosis_layer_obs) == False) & (np.isnan(wosis_layer_depth) == False) & (np.isnan(wosis_layer_upper_depth) == False) & (np.isnan(wosis_layer_lower_depth) == False))
		# valid layer number
		num_layers = len(valid_soc_loc[0])

		if num_layers > 0:
			wosis_layer_depth = wosis_layer_depth[valid_soc_loc]/100 # convert unit from cm to m
			wosis_layer_obs = wosis_layer_obs[valid_soc_loc]
			wosis_layer_upper_depth = wosis_layer_upper_depth[valid_soc_loc]/100
			wosis_layer_lower_depth = wosis_layer_lower_depth[valid_soc_loc]/100
			
			obs_depth_matrix[iprofile_hat, 0:num_layers] = wosis_layer_depth
			obs_soc_matrix[iprofile_hat, 0:num_layers] = wosis_layer_obs
			obs_upper_depth_matrix[iprofile_hat, 0:num_layers] = wosis_layer_upper_depth
			obs_lower_depth_matrix[iprofile_hat, 0:num_layers] = wosis_layer_lower_depth
		# end if num_layers > 0:
	else:
		profile_id = iprofile[0]
		# print('profile_id: ', profile_id)
		layer_num_record += 1
		# find the lon and lat info of soil profile
		lon_profile = data_POM_MAOM.loc[data_POM_MAOM['profile_id'] == profile_id, 'Longitude'].values[0]
		lat_profile = data_POM_MAOM.loc[data_POM_MAOM['profile_id'] == profile_id, 'Latitude'].values[0]

		lat_loc = np.where(abs(lat_profile - lat_grid) == min(abs(lat_profile - lat_grid)))[0][0]
		lon_loc = np.where(abs(lon_profile - lon_grid) == min(abs(lon_profile - lon_grid)))[0][0]

		# info of the node depth of profile
		obs_depth_matrix[iprofile_hat, 0] = data_POM_MAOM.loc[data_POM_MAOM['profile_id'] == profile_id, 'Node_Depth'].values / 100 # convert unit from cm to m
		# obs_soc_matrix[iprofile_hat, 0] = data_POM_MAOM.loc[data_POM_MAOM['profile_id'] == profile_id, 'SOC_stock_gC_m3'].values # Unit: gC/m3
		obs_POM_matrix[iprofile_hat, 0] = data_POM_MAOM.loc[data_POM_MAOM['profile_id'] == profile_id, 'POC_stock_gC_m3'].values
		obs_MAOM_matrix[iprofile_hat, 0] = data_POM_MAOM.loc[data_POM_MAOM['profile_id'] == profile_id, 'MAOC_stock_gC_m3'].values
		obs_upper_depth_matrix[iprofile_hat, 0] = data_POM_MAOM.loc[data_POM_MAOM['profile_id'] == profile_id, 'DEPTH-top'].values / 100 # convert unit from cm to m
		obs_lower_depth_matrix[iprofile_hat, 0] = data_POM_MAOM.loc[data_POM_MAOM['profile_id'] == profile_id, 'DEPTH-bottom'].values / 100 # convert unit from cm to m

	obs_lon_lat_loc[iprofile_hat, :] = [lon_loc, lat_loc]

	# input vector
	model_force_input_vector_cwd[iprofile_hat, :] = cesm2_simu_input_sum_cwd[lat_loc, lon_loc, :]
	model_force_input_vector_litter1[iprofile_hat, :] = cesm2_simu_input_sum_litter1[lat_loc, lon_loc, :]
	model_force_input_vector_litter2[iprofile_hat, :] = cesm2_simu_input_sum_litter2[lat_loc, lon_loc, :]
	model_force_input_vector_litter3[iprofile_hat, :] = cesm2_simu_input_sum_litter3[lat_loc, lon_loc, :]
	# altmax current and last year
	model_force_altmax_lastyear_profile[iprofile_hat, :] = cesm2_simu_altmax_last_year[lat_loc, lon_loc, :]
	model_force_altmax_current_profile[iprofile_hat, :] = cesm2_simu_altmax[lat_loc, lon_loc, :]
	# nbedrock
	model_force_nbedrock[iprofile_hat, :] = cesm2_simu_nbedrock[lat_loc, lon_loc, :]
	# oxygen scalar
	model_force_xio[iprofile_hat, :, :] = cesm2_simu_o_scalar[lat_loc, lon_loc, 0:soil_decom_num, :]
	# nitrogen scalar
	model_force_xin[iprofile_hat, :, :] = cesm2_simu_n_scalar[lat_loc, lon_loc, 0:soil_decom_num, :]
	# sand content
	model_force_sand_vector[iprofile_hat, :, :] = cesm2_simu_cellsand[lat_loc, lon_loc, 0:soil_decom_num, :]
	# soil temperature and water potential
	model_force_soil_temp_profile[iprofile_hat, :, :] = cesm2_simu_soil_temperature[lat_loc, lon_loc, 0:soil_decom_num, :]
	model_force_soil_water_profile[iprofile_hat, :, :] = cesm2_simu_w_scalar[lat_loc, lon_loc, 0:soil_decom_num, :]
# end

# check the overall number of layers in the profile
print("Number of layers in profile: " + str(layer_num_record))
print(datetime.now(), '------------soc data prepared------------')

########################################################
# neural network (BINNS)
########################################################
nn_split_ratio = 0.1
test_split_ratio = 0.1

#################################################
# Environmental feature list                    #
#################################################
env_info_names = ['ProfileNum', 'ProfileID', 'LayerNum', 'Lon', 'Lat', 'Date', \
'Rmean', 'Rmax', 'Rmin', \
'ESA_Land_Cover', \
'ET', \
'IGBP', 'Climate', 'Soil_Type', 'NPPmean', 'NPPmax', 'NPPmin', \
'Veg_Cover', \
'BIO1', 'BIO2', 'BIO3', 'BIO4', 'BIO5', 'BIO6', 'BIO7', 'BIO8', 'BIO9', 'BIO10', 'BIO11', 'BIO12', 'BIO13', 'BIO14', 'BIO15', 'BIO16', 'BIO17', 'BIO18', 'BIO19', \
'Abs_Depth_to_Bedrock', \
'Bulk_Density_0cm', 'Bulk_Density_30cm', 'Bulk_Density_100cm',\
'CEC_0cm', 'CEC_30cm', 'CEC_100cm', \
'Clay_Content_0cm', 'Clay_Content_30cm', 'Clay_Content_100cm', \
'Coarse_Fragments_v_0cm', 'Coarse_Fragments_v_30cm', 'Coarse_Fragments_v_100cm', \
'Depth_Bedrock_R', \
'Garde_Acid', \
'Occurrence_R_Horizon', \
'pH_Water_0cm', 'pH_Water_30cm', 'pH_Water_100cm', \
'Sand_Content_0cm', 'Sand_Content_30cm', 'Sand_Content_100cm', \
'Silt_Content_0cm', 'Silt_Content_30cm', 'Silt_Content_100cm', \
'SWC_v_Wilting_Point_0cm', 'SWC_v_Wilting_Point_30cm', 'SWC_v_Wilting_Point_100cm', \
'Texture_USDA_0cm', 'Texture_USDA_30cm', 'Texture_USDA_100cm', \
'USDA_Suborder', \
'WRB_Subgroup', \
'Drought', \
'Elevation', \
'Max_Depth', \
'Koppen_Climate_2018', \
'cesm2_npp', 'cesm2_npp_std', \
'cesm2_gpp', 'cesm2_gpp_std', \
'cesm2_vegc', \
'nbedrock', \
'R_Squared', \
'Ald_0_20',	'Ald_20_40', 'Ald_40_60', 'Ald_60_80', 'Ald_80_100', 'Alo_0_20', 'Alo_20_40', 'Alo_40_60', 'Alo_60_80', 'Alo_80_100',\
'Fed_0_20', 'Fed_20_40', 'Fed_40_60', 'Fed_60_80', 'Fed_80_100', 'Feo_0_20', 'Feo_20_40', 'Feo_40_60', 'Feo_60_80', 'Feo_80_100'
]

# Variables used in training the NN. NOTE the order changed from before.
if args.features in ["all", "all_including_lonlat"]:
	GEOGRAPHY_VARS = ['Lon', 'Lat', 'Elevation', 'Abs_Depth_to_Bedrock', 'Occurrence_R_Horizon', 'nbedrock']
	if args.features != "all_including_lonlat":
		GEOGRAPHY_VARS.remove('Lon')
		GEOGRAPHY_VARS.remove('Lat')
	CLIMATE_VARS = ['Koppen_Climate_2018', 'BIO1', 'BIO2', 'BIO3', 'BIO4', 'BIO5', 'BIO6', 'BIO7', 'BIO8', 'BIO9', 'BIO10', 'BIO11', 'BIO12', 'BIO13', 'BIO14', 'BIO15', 'BIO16', 'BIO17', 'BIO18', 'BIO19']
	SOIL_TEXTURE_VARS = ['USDA_Suborder', 'WRB_Subgroup', 'Coarse_Fragments_v_0cm', 'Coarse_Fragments_v_30cm', 'Coarse_Fragments_v_100cm',
						'Clay_Content_0cm', 'Clay_Content_30cm', 'Clay_Content_100cm', 'Silt_Content_0cm', 'Silt_Content_30cm', 'Silt_Content_100cm',
						'Texture_USDA_0cm', 'Texture_USDA_30cm', 'Texture_USDA_100cm', 'Sand_Content_0cm', 'Sand_Content_30cm', 'Sand_Content_100cm',
						'Bulk_Density_0cm', 'Bulk_Density_30cm', 'Bulk_Density_100cm']
	SOIL_CHEMICAL_VARS = ['SWC_v_Wilting_Point_0cm', 'SWC_v_Wilting_Point_30cm', 'SWC_v_Wilting_Point_100cm', 'pH_Water_0cm', 'pH_Water_30cm', 'pH_Water_100cm',
						'CEC_0cm', 'CEC_30cm', 'CEC_100cm', 'Garde_Acid']
	VEGETATION_VARS = ['ESA_Land_Cover', 'cesm2_npp', 'cesm2_npp_std', 'cesm2_vegc']
	var4nn = GEOGRAPHY_VARS + CLIMATE_VARS + SOIL_TEXTURE_VARS + SOIL_CHEMICAL_VARS + VEGETATION_VARS
elif args.features == "ten":
	# Ten handcrafted features
	# var4nn = ['BIO1', 'BIO2', 'BIO3', 'BIO4', 'BIO5', 'BIO6', 'BIO7', 'BIO8', 'BIO9', 'BIO10', 'BIO11', 'BIO12', 'BIO13', 'BIO14', 'BIO15', 'BIO16', 'BIO17', 'BIO18', 'BIO19', \
	# 	'Clay_Content_0cm', 'Clay_Content_30cm', 'Clay_Content_100cm', \
	# 	'Silt_Content_0cm', 'Silt_Content_30cm', 'Silt_Content_100cm',\
	# 	'SWC_v_Wilting_Point_0cm', 'SWC_v_Wilting_Point_30cm', 'SWC_v_Wilting_Point_100cm', \
	# 	'pH_Water_0cm', 'pH_Water_30cm', 'pH_Water_100cm', \
	# 	'CEC_0cm', 'CEC_30cm', 'CEC_100cm',\
	# 	'Coarse_Fragments_v_0cm', 'Coarse_Fragments_v_30cm', 'Coarse_Fragments_v_100cm', \
	# 	# "cesm2_npp", "cesm2_vegc", \
	# 	'Ald_avg', 'Alo_avg',\
	# 	'Fed_avg', 'Feo_avg', \
	# 	]
	var4nn = ['BIO1', 'BIO2', 'BIO3', 'BIO4', 'BIO5', 'BIO6', 'BIO7', 'BIO8', 'BIO9', 'BIO10', 'BIO11', 'BIO12', 'BIO13', 'BIO14', 'BIO15', 'BIO16', 'BIO17', 'BIO18', 'BIO19', \
		"Clay_Content_avg", "Sand_Content_avg", "Silt_Content_avg", \
		"Bulk_Density_avg",\
		"SWC_v_Wilting_Point_avg", "pH_Water_avg", "CEC_avg", \
		"Coarse_Fragments_avg", \
		# "cesm2_npp", "cesm2_vegc", \
		'Ald_avg', 'Alo_avg',\
		'Fed_avg', 'Feo_avg', \
		]
else:
	raise ValueError("Invalid features")


################################################
# Categorical variables                        #
################################################
# List of categorical variables. Inner lists group categorical
# variables that share the same categories. For example, the category IDs
# in Texture_USDA_0cm, Texture_USDA_30cm have the same semantic meaning,
# so they share an embedding space.
categorical_vars = [['ESA_Land_Cover'], ['Texture_USDA_0cm', 'Texture_USDA_30cm', 'Texture_USDA_100cm'], 
					['USDA_Suborder'], ['WRB_Subgroup'], ['Koppen_Climate_2018']]  # Variables inside a sub-list share the same categories
categorical_vars_flattened = [item for sublist in categorical_vars for item in sublist]


#############################################################################################
# Load environmental covariates, and transform to [0, 1] range based on precomputed min/max #
#############################################################################################
# Load environmental covariates
# Environment info data for WOSIS profiles
env_info = np.genfromtxt(data_dir_input + 'wosis_2019_snap_shot/env_info_SOC_with_Al_Fe.csv', delimiter = ',', skip_header = 1)
# Environment info data for POM and MAOM profiles
env_info_POM_MAOM = np.genfromtxt(data_dir_input + 'POM_MAOM/env_info_POM_MAOM_with_Al_Fe.csv', delimiter = ',', skip_header = 1)
# Merge the two datasets
env_info = np.vstack((env_info, env_info_POM_MAOM))
original_lons = env_info[:, 3].copy()  # Save the original (unscaled) lon/lat
original_lats = env_info[:, 4].copy()
env_info = df(env_info)
env_info.columns = env_info_names

# Min/max for each feature
col_max_min = loadmat(data_dir_input + 'wosis_2019_snap_shot/world_grid_envinfo_present_cesm2_clm5_cen_vr_v2_whole_time_col_max_min.mat')
col_max_min = col_max_min['col_max_min']
print("Shape of col_max_min: ", col_max_min.shape)
col_max_min_minerals = np.genfromtxt(data_dir_input + 'wosis_2019_snap_shot/Al_Fe_max_min_values.csv', delimiter = ',', skip_header = 1)
print("Shape of col_max_min_minerals: ", col_max_min_minerals.shape)
# Merge the two datasets
col_max_min = np.vstack((col_max_min, col_max_min_minerals))
print("Shape of col_max_min after merging: ", col_max_min.shape)

# Don't want to transform categorical variables, so set max/min to nan
for group in categorical_vars:
	for var in group:
		idx = env_info_names.index(var)
		col_max_min[idx, :] = np.nan

# Logic to add columns for "average" variables (e.g. average over layers).
# Also record the min/max for these new columns.
all_col_max_mins = [col_max_min]
for v in var4nn:
	if v not in env_info_names:
		var_prefix = v.split("_avg")[0]
		columns = env_info.filter(regex=rf"^{var_prefix}.*")
		env_info[v] = columns.mean(axis=1)
		indices = [env_info_names.index(col) for col in columns.columns]
		new_max_min = np.mean(col_max_min[indices, :], axis=0, keepdims=True)  # keep shape [1, 2]
		all_col_max_mins.append(new_max_min)
	else:
		idx_var = env_info_names.index(v)
		if np.isnan(col_max_min[idx_var, 0]) or np.isnan(col_max_min[idx_var, 1]):
    			# Get the min/max from the input data
			min_temp = np.nanmin(env_info[v])
			max_temp = np.nanmax(env_info[v])
			col_max_min[idx_var, 0] = min_temp
			col_max_min[idx_var, 1] = max_temp

col_max_min = np.concatenate(all_col_max_mins, axis=0)  # shape [num_columns_new, 2]



# Update var4nn to include the new columns
if args.features == "ten":
    # Ten handcrafted features
	var4nn = ['BIO1', 'BIO2', 'BIO3', 'BIO4', 'BIO5', 'BIO6', 'BIO7', 'BIO8', 'BIO9', 'BIO10', 'BIO11', 'BIO12', 'BIO13', 'BIO14', 'BIO15', 'BIO16', 'BIO17', 'BIO18', 'BIO19', \
		'Clay_Content_0cm', 'Clay_Content_30cm', 'Clay_Content_100cm', \
		'Silt_Content_0cm', 'Silt_Content_30cm', 'Silt_Content_100cm',\
		'SWC_v_Wilting_Point_0cm', 'SWC_v_Wilting_Point_30cm', 'SWC_v_Wilting_Point_100cm', \
		'pH_Water_0cm', 'pH_Water_30cm', 'pH_Water_100cm', \
		'CEC_0cm', 'CEC_30cm', 'CEC_100cm',\
		'Coarse_Fragments_v_0cm', 'Coarse_Fragments_v_30cm', 'Coarse_Fragments_v_100cm', \
		# "cesm2_npp", "cesm2_vegc", \
		# 'Ald_avg', 'Alo_avg',\
		# 'Fed_avg', 'Feo_avg', \
		]
	
	var4nn = ['BIO1', 'BIO2', 'BIO3', 'BIO4', 'BIO5', 'BIO6', 'BIO7', 'BIO8', 'BIO9', 'BIO10', 'BIO11', 'BIO12', 'BIO13', 'BIO14', 'BIO15', 'BIO16', 'BIO17', 'BIO18', 'BIO19', \
		"Clay_Content_avg", "Sand_Content_avg", "Silt_Content_avg", \
		"Bulk_Density_avg",\
		"SWC_v_Wilting_Point_avg", "pH_Water_avg", "CEC_avg", \
		"Coarse_Fragments_avg", \
		# "cesm2_npp", "cesm2_vegc", \
		# 'Ald_avg', 'Alo_avg',\
		# 'Fed_avg', 'Feo_avg', \
		]

print("Size of col_max_min: ", col_max_min.shape)
# Scale numeric features to [0, 1] based on precomputed min/max 
warnings.filterwarnings("ignore")  # Ignore warnings about subtracting nan
for ivar in np.arange(3, len(col_max_min[:, 0])):
	if np.isnan(col_max_min[ivar, :]).any():
		pass
	else:
		env_info.iloc[:, ivar] = (env_info.iloc[:, ivar] - col_max_min[ivar, 0])/(col_max_min[ivar, 1] - col_max_min[ivar, 0])
		env_info.iloc[(env_info.iloc[:, ivar] > 1), ivar] = 1
		env_info.iloc[(env_info.iloc[:, ivar] < 0), ivar] = 0
warnings.resetwarnings()

# Retain orginal lat/lon
env_info["original_lon"] = original_lons
env_info["original_lat"] = original_lats




# env_info["Ald_avg"] = (env_info["Ald_0_20"] + env_info["Ald_20_40"]) / 2
# env_info["Alo_avg"] = (env_info["Alo_0_20"] + env_info["Alo_20_40"]) / 2
# env_info["Fed_avg"] = (env_info["Fed_0_20"] + env_info["Fed_20_40"]) / 2
# env_info["Feo_avg"] = (env_info["Feo_0_20"] + env_info["Feo_20_40"]) / 2


########################################################################
# Preprocessing of categorical variables. Perhaps this should be moved
# inside the neural network itself to make usage easier.
#########################################################################
# Determine how many indices are in each categorical group
var_to_categories = dict()  # varname to number of categories
for group in categorical_vars:
	n_categories = int(np.nanmax(env_info[group]) + 1)
	for var in group:
		var_to_categories[var] = n_categories
print("Var to categories", var_to_categories)

# Compute indices of each variable after categorical variables are expanded
var_to_indices = dict()
curr_idx = 0
for var in var4nn:
	if var in var_to_categories:
		if args.categorical == "embedding":
			n_indices = args.embed_dim
		elif args.categorical == "one_hot":
			n_indices = var_to_categories[var]
		else:
			raise ValueError("Invalid args.categorical")
	else:
		n_indices = 1
	var_to_indices[var] = list(range(curr_idx, curr_idx + n_indices))
	curr_idx += n_indices

# Indices of each group after categorical variables are expanded
if args.features in ["all", "all_including_lonlat"]:
	GEOGRAPHY_INDICES = [i for var in GEOGRAPHY_VARS for i in var_to_indices[var]]
	CLIMATE_INDICES = [i for var in CLIMATE_VARS for i in var_to_indices[var]]
	SOIL_TEXTURE_INDICES = [i for var in SOIL_TEXTURE_VARS for i in var_to_indices[var]]
	SOIL_CHEMICAL_INDICES = [i for var in SOIL_CHEMICAL_VARS for i in var_to_indices[var]]
	VEGETATION_INDICES = [i for var in VEGETATION_VARS for i in var_to_indices[var]]
print("Var to indices", var_to_indices)


#---------------------------------------------------
# training data
#---------------------------------------------------
# Input features (environmental covariates)
current_data_x = np.ones((len(profile_collection), max(len(var4nn), 20), 12, 13))*np.nan

# Fill in input features
# NOTE: env_info is indexed starting from 0, and profile_collection
# is also using zero-based indices
current_data_x[:, 0:len(var4nn), 0, 0] = np.array(env_info.loc[profile_collection[:, 0], var4nn])

# Monthly forcing variables
current_data_x[:, 0:12, 0, 1] = model_force_input_vector_cwd
current_data_x[:, 0:12, 0, 2] = model_force_input_vector_litter1
current_data_x[:, 0:12, 0, 3] = model_force_input_vector_litter2
current_data_x[:, 0:12, 0, 4] = model_force_input_vector_litter3
current_data_x[:, 0:12, 0, 5] = model_force_altmax_lastyear_profile
current_data_x[:, 0:12, 0, 6] = model_force_altmax_current_profile
current_data_x[:, 0:12, 0, 7] = model_force_nbedrock

# Forcing variables that apply for each depth layer
current_data_x[:, 0:20, 0:12, 8] = model_force_xio
current_data_x[:, 0:20, 0:12, 9] = model_force_xin
current_data_x[:, 0:20, 0:12, 10] = model_force_sand_vector
current_data_x[:, 0:20, 0:12, 11] = model_force_soil_temp_profile
current_data_x[:, 0:20, 0:12, 12] = model_force_soil_water_profile

# SOC labels: each row represents a location, each column represents an observation
# at a certain depth. Since locations have different numbers of observations, many entries are NaN.
current_data_y = np.ones((len(profile_collection), 3, 200))*np.nan
current_data_y[:, 0, :] = obs_soc_matrix
current_data_y[:, 1, :] = obs_POM_matrix
current_data_y[:, 2, :] = obs_MAOM_matrix

# Depth of each observation in current_data_y (same shape)
current_data_z = obs_depth_matrix

# Geographic coordinates
lons = np.array(env_info.loc[profile_collection[:, 0], "original_lon"])
lats = np.array(env_info.loc[profile_collection[:, 0], "original_lat"])
current_data_c = np.stack([lons, lats], axis=1)  # [profile, 2]: lon/lat of each site

# Remove sites with missing features or forcing variables
nan_loc = np.sum(current_data_x[:, 0:len(var4nn), 0, 0], axis = 1) + \
			np.sum(model_force_input_vector_cwd, axis = 1) + \
			np.sum(model_force_input_vector_litter1, axis = 1) + \
			np.sum(model_force_input_vector_litter2, axis = 1) + \
			np.sum(model_force_input_vector_litter3, axis = 1) + \
			np.sum(model_force_altmax_lastyear_profile, axis = 1) + \
			np.sum(model_force_altmax_current_profile, axis = 1) + \
			np.sum(model_force_nbedrock, axis = 1) + \
			np.sum(model_force_xio, axis = (1, 2)) + \
			np.sum(model_force_xin, axis = (1, 2)) + \
			np.sum(model_force_sand_vector, axis = (1, 2)) + \
			np.sum(model_force_soil_temp_profile, axis = (1, 2)) + \
			np.sum(model_force_soil_water_profile, axis = (1, 2))
# valid_profile_loc = np.where(np.isnan(nan_loc) == False)[0] 
nan_mask = ~np.isnan(nan_loc)

# Drop profiles with negative values in current_data_y
neg_mask = ~np.any(current_data_y < 0, axis=(1, 2))
# Combine the two masks
valid_profile_loc = np.where(nan_mask & neg_mask)[0]

current_data_y = current_data_y[valid_profile_loc, :, :]
current_data_z = current_data_z[valid_profile_loc, :]
current_data_x = current_data_x[valid_profile_loc, :, :, :]
current_data_c = current_data_c[valid_profile_loc, :]
current_data_profile_id = profile_collection[valid_profile_loc, 0]
obs_upper_depth_matrix = obs_upper_depth_matrix[valid_profile_loc, :]
obs_lower_depth_matrix = obs_lower_depth_matrix[valid_profile_loc, :]


print("Shape of current data x", current_data_x.shape)
print("Shape of current data y", current_data_y.shape)
print("Shape of current data z", current_data_z.shape)
print("Shape of current data c", current_data_c.shape)
print("Shape of current_data_profile_id", current_data_profile_id.shape)
print("Shape of obs upper depth matrix", obs_upper_depth_matrix.shape)
print("Shape of obs lower depth matrix", obs_lower_depth_matrix.shape)
print("Shape of env info", env_info.shape)



###############################################################
# Load checkpoint if resuming a previous run.
# We do this outside the main function, since the checkpoint
# stores the train/val/test split for setting up the datasets.
##################################################################
# If PREVIOUS_JOB_ID environment variable set, overwrite the
# commandline arg.
if 'PREVIOUS_JOB_ID' in os.environ:
	args.previous_job_id = os.environ.get('PREVIOUS_JOB_ID')
	print("Overrode previous_job_id. Now", args.previous_job_id)

# Load checkpoint if resuming. This checkpoint is only used to read the
# data splits, the actual weights will be loaded later.
if args.whether_resume == 1:
	checkpoint_path = data_dir_output + 'neural_network/' + args.previous_job_id + '/checkpoint_' + args.previous_job_id + '.pt'
	checkpoint_main = torch.load(checkpoint_path, weights_only=False)

	# Delete the job submit file if it exists
	try:
		os.remove(job_submit_path + 'Resume' + args.previous_job_id + '.submit')
	except OSError:
		pass

################################################################
# Data splitting
################################################################
n_datapoints = current_data_x.shape[0]

# Train, validation, test split
if args.whether_resume == 0:
	if args.cross_val_idx == 0:
		# Single train/val/test split, no cross-validation.
		# Compute train/val/test indices
		if args.split == 'random':
			# Randomly split US datapoints into train/val/test (ignoring geography) 
			rng = np.random.default_rng(seed=args.data_seed)
			# Determine the number of training samples based on the ratios
			train_loc = rng.choice(np.arange(0, n_datapoints), size=round((1 - args.val_ratio - args.test_ratio) * n_datapoints), replace=False)
			# The remaining data after removing the training samples
			remaining_loc = np.setdiff1d(np.arange(0, n_datapoints), train_loc)
			# Split the remaining data into validation and test sets
			num_val_samples = round(args.val_ratio / (args.val_ratio + args.test_ratio) * len(remaining_loc))
			val_loc = rng.choice(remaining_loc, size=num_val_samples, replace=False)
			test_loc = np.setdiff1d(remaining_loc, val_loc)
		elif args.split == 'us_vs_world':
			# Train set is southern US, validation set is northern US, test set is rest of world
			train_loc = np.flatnonzero(
				(wosis_profile_info[current_data_profile_id, 2] == 156) &
				(wosis_profile_info[current_data_profile_id, 3] >= -124.763068) &
				(wosis_profile_info[current_data_profile_id, 3] <= -66.949895) &
				(wosis_profile_info[current_data_profile_id, 4] >= 24.5) &
				(wosis_profile_info[current_data_profile_id, 4] <= 40.)
			)
			val_loc = np.flatnonzero(
				(wosis_profile_info[current_data_profile_id, 2] == 156) &
				(wosis_profile_info[current_data_profile_id, 3] >= -124.763068) &
				(wosis_profile_info[current_data_profile_id, 3] <= -66.949895) &
				(wosis_profile_info[current_data_profile_id, 4] > 40.) &
				(wosis_profile_info[current_data_profile_id, 4] <= 49.384358)
			)
			test_loc = np.setdiff1d(np.arange(0, n_datapoints), train_loc)
			test_loc = np.setdiff1d(test_loc, val_loc)
		else:
			raise ValueError("If cross_val_idx is 0, only random or us_vs_world split is supported. To use north/south or east/west splits, set cross_val_idx to a number between 1 and 10.")

	else:
		# Split the data into k-folds, either randomly or by spatial block
		# Recall cross_val_idx is one-based. Subtract one to make it zero-based. The test fold
		# is given by cross_val_idx-1, and validation fold is one larger (cross_val_idx % n_folds)
		test_fold = args.cross_val_idx - 1
		val_fold = args.cross_val_idx % args.n_folds
		if args.split == 'random':
			# Random split
			# Assign the test dataset based on the cross-validation index, and val dataset
			# Randomly split the remaining data into training and validation sets
			kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.data_seed)
			fold_indices = list(kf.split(np.arange(len(current_data_x[:, 0]))))
			test_loc = fold_indices[args.cross_val_idx - 1][1]
			train_val_idx = fold_indices[args.cross_val_idx - 1][0]
			train_loc = np.random.choice(train_val_idx, size=round((1 - nn_split_ratio - test_split_ratio)/(1 - test_split_ratio) * len(train_val_idx)), replace=False)
			val_loc = np.setdiff1d(train_val_idx, train_loc)
		elif args.split == "horizontal":
			# Compute latitude thresholds separating folds. If we have 10 folds, we have 11 boundary thresholds.
			sorted_lats = np.sort(current_data_c[:, 1])   # current_data_c[:, 1] contains latitudess
			indices = np.linspace(0, n_datapoints, num=args.n_folds, endpoint=False)  # Fold start/stop indices: [0, n_datapoints*0.1, n_datapoints*0.2, ... n_datapoints*0.9]
			lat_thresholds = [sorted_lats[int(i)] for i in indices]  # Lat boundaries between folds
			lat_thresholds.append(sorted_lats[-1] + 1)  # Add final threshold above all datapoints

			# Recall cross_val_idx is one-based. Subtract one to make it zero-based. The test fold
			# is given by cross_val_idx-1, and validation fold is one larger (cross_val_idx % n_folds)
			test_loc = np.flatnonzero((current_data_c[:, 1] >= lat_thresholds[test_fold]) & (current_data_c[:, 1] < lat_thresholds[test_fold+1]))
			val_loc = np.flatnonzero((current_data_c[:, 1] >= lat_thresholds[val_fold]) & (current_data_c[:, 1] < lat_thresholds[val_fold+1]))

			# Train loc is all indices except val/test
			train_loc = np.setdiff1d(np.arange(0, n_datapoints), test_loc)
			train_loc = np.setdiff1d(train_loc, val_loc)
		elif args.split == "vertical":
			# Compute longitude thresholds separating folds. If we have 10 folds, we have 11 boundary thresholds.
			sorted_lons = np.sort(current_data_c[:, 0])   # current_data_c[:, 1] contains longitudes
			indices = np.linspace(0, n_datapoints, num=args.n_folds, endpoint=False)  # Fold start/stop indices: [0, n_datapoints*0.1, n_datapoints*0.2, ... n_datapoints*0.9]
			lon_thresholds = [sorted_lons[int(i)] for i in indices]  # Lon boundaries between folds
			lon_thresholds.append(sorted_lons[-1] + 1)  # Add final threshold above all datapoints

			# Recall cross_val_idx is one-based. Subtract one to make it zero-based. The test fold
			# is given by cross_val_idx-1, and validation fold is one larger (cross_val_idx % n_folds)
			test_loc = np.flatnonzero((current_data_c[:, 0] >= lon_thresholds[test_fold]) & (current_data_c[:, 0] < lon_thresholds[test_fold+1]))
			val_loc = np.flatnonzero((current_data_c[:, 0] >= lon_thresholds[val_fold]) & (current_data_c[:, 0] < lon_thresholds[val_fold+1]))

			# Train loc is all indices except val/test
			train_loc = np.setdiff1d(np.arange(0, n_datapoints), test_loc)
			train_loc = np.setdiff1d(train_loc, val_loc)
		elif args.split == "grid2":
			GRID_SIZE = 2  # in degrees longitude/latitude

			# For each site, compute its coordinates in a GRID_SIZE*GRID_SIZE grid
			min_lon, max_lon = current_data_c[:, 0].min(), current_data_c[:, 0].max()

			# Nearest multiple of GRID_SIZE below min_lon (https://stackoverflow.com/questions/2272149/round-to-5-or-other-number-in-python)
			min_lon_rounded = GRID_SIZE * np.floor(min_lon / GRID_SIZE)
			max_lon_rounded = GRID_SIZE * np.ceil(max_lon / GRID_SIZE)
			col_idx = np.floor((current_data_c[:, 0] - min_lon_rounded) / GRID_SIZE)
			n_cols = round((max_lon_rounded - min_lon_rounded) / GRID_SIZE)

			# Repeat for lat
			min_lat, max_lat = current_data_c[:, 1].min(), current_data_c[:, 1].max()
			min_lat_rounded = GRID_SIZE * np.floor(min_lat / GRID_SIZE)
			max_lat_rounded = GRID_SIZE * np.ceil(max_lat / GRID_SIZE)
			row_idx = np.floor((current_data_c[:, 1] - min_lat_rounded) / GRID_SIZE)
			n_rows = round((max_lat_rounded - min_lat_rounded) / GRID_SIZE)

			# Compute a "grid cell ID"
			cell_id = (row_idx * n_cols + col_idx).astype(int)

			# Split cells into folds. See https://stackoverflow.com/questions/33398017/to-generate-a-split-indices-for-n-fold
			s = np.arange(n_rows * n_cols)
			random.Random(args.data_seed).shuffle(s)
			val_cells = s[val_fold::args.n_folds]
			test_cells = s[test_fold::args.n_folds]

			# Split sites
			test_loc = np.flatnonzero(np.isin(cell_id, test_cells))
			val_loc = np.flatnonzero(np.isin(cell_id, val_cells))
			train_loc = np.setdiff1d(np.setdiff1d(np.arange(0, n_datapoints), test_loc), val_loc)
		else:
			raise ValueError("Invalid value of --split")

else:
	# If we are resuming, load the same train/val/test indices.
	train_loc = checkpoint_main['train_indices']
	val_loc = checkpoint_main['val_indices']
	test_loc = checkpoint_main['test_indices']

# Construct the train/val/test splits
train_y = torch.tensor(current_data_y[train_loc, :], dtype=torch.float32)
val_y = torch.tensor(current_data_y[val_loc, :], dtype=torch.float32)
test_y = torch.tensor(current_data_y[test_loc, :], dtype=torch.float32)

train_z = torch.tensor(current_data_z[train_loc, :], dtype=torch.float32)
val_z = torch.tensor(current_data_z[val_loc, :], dtype=torch.float32)
test_z = torch.tensor(current_data_z[test_loc, :], dtype=torch.float32)

train_c = torch.tensor(current_data_c[train_loc, :], dtype=torch.float32)
val_c = torch.tensor(current_data_c[val_loc, :], dtype=torch.float32)
test_c = torch.tensor(current_data_c[test_loc, :], dtype=torch.float32)

train_x = torch.tensor(current_data_x[train_loc, :, :, :], dtype=torch.float32)
val_x = torch.tensor(current_data_x[val_loc, :, :, :], dtype=torch.float32)
test_x = torch.tensor(current_data_x[test_loc, :, :, :], dtype=torch.float32)

train_profile_id = torch.tensor(current_data_profile_id[train_loc])
val_profile_id = torch.tensor(current_data_profile_id[val_loc])
test_profile_id = torch.tensor(current_data_profile_id[test_loc])

train_proda_para = torch.tensor(np.ones((len(train_loc), 1)), dtype=torch.float32)
val_proda_para = torch.tensor(np.ones((len(val_loc), 1)), dtype=torch.float32)
test_proda_para = torch.tensor(np.ones((len(test_loc), 1)), dtype=torch.float32)

print("Shape of train data", train_x.shape)
print("Shape of val data", val_x.shape)
print("Shape of test data", test_x.shape)

print(datetime.now(), '------------nn data prepared------------')


# Helper function to combine the training data into a single tensor
class MergeDataset(Dataset):
	def __init__(self, data_x, data_y, data_z, data_c, profile_id, proda_para):
		self.data_x = data_x
		self.data_y = data_y
		self.data_z = data_z
		self.data_c = data_c
		self.profile_id = profile_id
		self.proda_para = proda_para

	def __len__(self):
		return len(self.data_x)

	def __getitem__(self, idx):
		return self.data_x[idx], self.data_y[idx], self.data_z[idx], self.data_c[idx], self.profile_id[idx], self.proda_para[idx]



def create_output_folders(args):
	"""
	Creates and returns a unique job id, based on timestamp, note, PBS job id
	(if exists), seed, cross_val_idx.

	Creates all output folders using this jobID.
	Should only be called once (before spawning processes).

	TODO: This could be instead based on a hash of the hyperparameters.
	This way we would not need to manually specify previous job ID to resume.
	"""

	# @joshuafan: Create a "job id" using timestamp, note, and PBS jobid
	if args.whether_resume == 0:
		# If not resuming, create a new job id
		job_id = time.strftime("%Y%m%d-%H%M%S")  # Convert datetime to string: https://stackoverflow.com/questions/10607688/how-to-create-a-file-name-with-the-current-date-time-in-python
		if args.note != "":
			job_id += ("_" + args.note)
		pbs_job_id = os.environ.get('PBS_JOBID')
		if pbs_job_id is not None:
			pbs_job_id = pbs_job_id.split('.')[0]
			job_id += ("_" + pbs_job_id)
		job_id += ("_lr={:.0e}".format(args.lr))
		job_id += ("_fold=" + str(args.cross_val_idx))
		job_id += ("_seed=" + str(args.seed))
	else:
		# If resuming, use the same job id as before
		# (Note: if PREVIOUS_JOB_ID environment variable was set, this was already
		# loaded into args.previous_job_id above.)
		job_id = args.previous_job_id

	# Create all necessary output folders
	os.makedirs(os.path.join(data_dir_output, "neural_network"), exist_ok=True)
	os.makedirs(os.path.join(data_dir_output, "neural_network", job_id), exist_ok=True)
	os.makedirs(os.path.join(data_dir_output, "neural_network", job_id, "model_parameters"), exist_ok=True)
	os.makedirs(os.path.join(data_dir_output, "neural_network", job_id, "model_training_history"), exist_ok=True)
	os.makedirs(os.path.join(data_dir_output, 'neural_network', job_id, 'visualizations'), exist_ok=True)
	os.makedirs(os.path.join(data_dir_output, "neural_network", job_id, "Prediction"), exist_ok=True)
	os.makedirs(os.path.join(data_dir_output, "neural_network", job_id, "Bulk_Simulation"), exist_ok=True)
	return job_id


def ddp_setup(rank, world_size, port):
	"""
	Setup environment parameters and devices.
	Source: https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py

	Args:
		rank: Unique identifier of each process
		world_size: Total number of processes
	"""
	# Print number of threads/CPUs
	cpu_count = multiprocessing.cpu_count()
	thread_count = torch.get_num_threads()
	print(datetime.now(), f"========= Setting up DDP. Worker, rank {rank} of {world_size} ==========")
	# print("Number of CPUs: ", torch.cpu.device_count())  # No idea why it is not working on NCAR server
	print("Number of Cores: ", cpu_count)
	print("Number of threads: ", thread_count)
	if "CUDA_VISIBLE_DEVICES" in os.environ:
		print("CUDA_VISIBLE_DEVICES", os.environ["CUDA_VISIBLE_DEVICES"])

	# Set number of threads *per worker*. Should equal floor(CPUs/processes)
	torch.set_num_threads(math.floor(torch.get_num_threads() / world_size))

	# Environment variables
	os.environ['RANK'] = str(rank)
	os.environ['WORLD_SIZE'] = str(world_size)
	os.environ["MASTER_ADDR"] = "localhost"
	os.environ["MASTER_PORT"] = port
	# Set device to CPU
	device = torch.device("cpu")

	# Initialize process group (for CPU, use gloo backend)
	dist.init_process_group(backend="gloo", rank=rank, world_size=world_size, timeout=timedelta(hours=2))

	return device


# Start training
def worker(rank, world_size, job_id, port):

	# Filename to store loss records and visualizations
	avg_loss_filename = 'avg_loss_' + job_id + '.txt'
	avg_l1_loss_filename = 'avg_l1_loss_' + job_id + '.txt'
	avg_NSE_filename = 'avg_NSE_' + job_id + '.txt'
	PLOT_DIR = os.path.join(data_dir_output, 'neural_network', job_id, 'visualizations')
	os.makedirs(PLOT_DIR, exist_ok=True)  # Note: this should already exist from create_output_folders

	# Save printed output to file. Does not seem to work on Slurm.
	# sys.stdout = misc_utils.Logger(os.path.join(data_dir_output, "neural_network", job_id, "output.txt"))

	# Set up distributed environment
	# if args.use_ddp == 1:
	# 	device = ddp_setup(rank, world_size, port)
	# else:
	# 	device = "cuda:" + str(os.environ["CUDA_VISIBLE_DEVICES"].split(',')[0]) if torch.cuda.is_available() else "cpu"
	# print(f"Finished DDP setup. Rank {rank} of {world_size}. Device {device}. JobID {job_id}.", flush=True)
	# Initialize the process group
	os.environ['RANK'] = str(rank)
	os.environ['WORLD_SIZE'] = str(world_size)
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = str(port)
	device = torch.device("cpu")

	# Initialize distributed environment
	dist.init_process_group('gloo', rank=rank, world_size=world_size, timeout=timedelta(hours=4))

	# Create embeddings for categorical variables (each int maps to a different category)
	# If using PyTorch DDP, I think this has to be done inside worker(). Each worker
	# maintains its own copy of the Embedding weights, but they are initialized the same way.
	var_idx_to_emb = OrderedDict()  # Column index (before expanding categorical vars) to embedding layer to use
	for group in categorical_vars:
		# Note that within a 'group', variables share embeddings. For example,
		# for 'Texture_USDA_0cm' and 'Texture_USDA_30cm', the embedding of each
		# category is the same.
		if group[0] not in var4nn:
			continue
		n_categories = var_to_categories[group[0]]
		if args.categorical == "embedding":
			emb = nn.Embedding(num_embeddings=n_categories, embedding_dim=args.embed_dim).to(device)
		elif args.categorical == "one_hot":
			emb = n_categories  # Just store the number of categories for one-hot encoding
		else:
			raise ValueError("Invalid value for args.categorical")
		for var in group:
			idx = var4nn.index(var)
			var_idx_to_emb[idx] = emb

	# TODO Not sure if "global model" is correct
	# global model
	if args.bias_only_epochs >= 1 and (args.whether_resume == 0 or checkpoint_main["epoch"] < args.bias_only_epochs):
		# If training bias only, temporarily set the model to ConstantParameters
		# and then switch to the real model after that many epochs.
		# Exception: If we are resuming and the epoch is already past bias_only_epochs,
		# go to the else branch (don't create the ConstantParameters model, create the full model)
		model_class = ConstantParameters
		model_kwargs = {
			"num_params": len(para_names),
			"vertical_mixing": args.vertical_mixing,
			"param_constraint": args.param_constraint,
			"min_temp": args.min_temp,
			"max_temp": args.max_temp
		}
	else:
		model_class, model_kwargs = misc_utils.get_model(args, var4nn, var_idx_to_emb, device,
										 				 para_index, train_x, train_y)

	if args.whether_resume == 1:
		# Load the model from the checkpoint, and overwrite model_kwargs if saved
		checkpoint_worker = torch.load(checkpoint_path, map_location=device, weights_only=False)
		model_kwargs = checkpoint_worker["model_kwargs"]

	if args.loss_weighting not in ["manual", "two_stage", "relobralo"]:
		raise ValueError("You selected an advanced loss_weighting method that depends on the LibMTL library. This is not implemented yet.")

		# # Specialized multi-task loss weighting method
		# weighting = weighting_method.__dict__[args.loss_weighting]

		# # Attempt to match the MTLmodel API. Create a class that derives from
		# # both our architecture class (customized for BINN) and our weighting method
		# # class (provided by LibMTL)
		# class MTLmodel(model_class, weighting):
		# 	def __init__(self, **kwargs):
		# 		super(MTLmodel, self).__init__(**kwargs)
		# 		self.init_param()

		# model = MTLmodel(**model_kwargs).to(device)
	else:
		# Create model
		model = model_class(**model_kwargs).to(device)

	# Create distributed version of the model
	if args.use_ddp == 1:
		if torch.cuda.is_available():
			model = DDP(model, device_ids=[device])
		else:  # CPU only
			model = DDP(model)  #, find_unused_parameters=True)
		model_without_ddp = model.module
	else:
		model_without_ddp = model

	# Optimizer and scheduler
	optimizer, scheduler = misc_utils.get_optimizer_and_scheduler(model, args)
	if args.use_swa:
		# Stochastic Weight Averaging. TODO - not tested fully.
		swa_model = AveragedModel(model)
		swa_start = 5
		swa_scheduler = SWALR(optimizer, swa_lr=0.05)

	if args.whether_resume == 1:
		# Load the model from the checkpoint
		state_dict = checkpoint_worker['model_state_dict']
		model.load_state_dict(state_dict)
		optimizer.load_state_dict(checkpoint_worker['optimizer_state_dict'])
		if args.use_swa:
			swa_model.load_state_dict(checkpoint_worker['swa_model_state_dict'])

	# Loss function
	fun_loss = binns_loss

	# Initialize datasets
	# Push to GPU if it fits in the GPU memory. If it doesn't, remove .to(device) below.
	train_dataset = MergeDataset(train_x.to(device), train_y.to(device), train_z.to(device), train_c.to(device), train_profile_id.to(device), train_proda_para.to(device))
	val_dataset = MergeDataset(val_x.to(device), val_y.to(device), val_z.to(device), val_c.to(device), val_profile_id.to(device), val_proda_para.to(device))
	# train_dataset = MergeDataset(train_x, train_y, train_z, train_c, train_profile_id, train_proda_para)
	# val_dataset = MergeDataset(val_x, val_y, val_z, val_c, val_profile_id, val_proda_para)

	# Use DistributedSampler for distributed training
	if args.use_ddp == 1:
		train_sampler = DistributedSampler(train_dataset)
		val_sampler = DistributedSampler(val_dataset)
	else:
		train_sampler, val_sampler = None, None

	# Data loaders with DistributedSampler
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)  #, num_workers=4, persistent_workers=True)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler)  # , num_workers=4, persistent_workers=True)

	# training and validation loop
	num_epoch = args.n_epochs

	if args.whether_resume == 0:
		# record the loss history
		train_loss_history = np.ones((num_epoch, len(args.losses)))*np.nan
		train_loss_history_soc = np.ones((num_epoch, 1))*np.nan
		train_loss_history_POM = np.ones((num_epoch, 1))*np.nan
		train_loss_history_MAOM = np.ones((num_epoch, 1))*np.nan
		val_loss_history = np.ones((num_epoch, len(args.losses)))*np.nan
		val_loss_history_soc = np.ones((num_epoch, 1))*np.nan
		val_loss_history_POM = np.ones((num_epoch, 1))*np.nan
		val_loss_history_MAOM = np.ones((num_epoch, 1))*np.nan
		train_metrics_history = np.ones((num_epoch, 1))*np.nan  # Columns are [MSE, MAE, NSE]
		train_metrics_history_POM = np.ones((num_epoch, 1))*np.nan
		train_metrics_history_MAOM = np.ones((num_epoch, 1))*np.nan
		val_metrics_history = np.ones((num_epoch, 1))*np.nan
		val_metrics_history_POM = np.ones((num_epoch, 1))*np.nan
		val_metrics_history_MAOM = np.ones((num_epoch, 1))*np.nan
		lr_history = np.ones((num_epoch))*np.nan
		best_model_epoch = torch.tensor(0) # epoch with the best model so far

		# Early stopping parameters
		best_val_loss_soc = float('inf')
		best_val_loss_pom_maom = float('inf')
		best_val_NSE = float('inf')
		best_val_NSE_soc = float('inf') 
		best_val_NSE_POM = float('inf')
		best_val_NSE_MAOM = float('inf') 
		patience = args.patience
		epochs_without_improvement = 0

		# Save observations
		binn_obs_soc = np.ones((combined_profile_num, 200))*np.nan
		binn_obs_soc[current_data_profile_id, :] = current_data_y[:, 0, :]
		binn_obs_POM = np.ones((combined_profile_num, 200))*np.nan
		binn_obs_POM[current_data_profile_id, :] = current_data_y[:, 1, :]
		binn_obs_MAOM = np.ones((combined_profile_num, 200))*np.nan
		binn_obs_MAOM[current_data_profile_id, :] = current_data_y[:, 2, :]


		# Define starting epoch
		start_epoch = 0

		if rank == 0:
			# Test whether the observations contains any negative values
			print("Negative values in SOC observations: ", np.any(binn_obs_soc < 0))
			print("Negative values in POM observations: ", np.any(binn_obs_POM < 0))
			print("Negative values in MAOM observations: ", np.any(binn_obs_MAOM < 0))
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/nn_obs_soc_' + job_id + '.csv', binn_obs_soc, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/nn_obs_POM_' + job_id + '.csv', binn_obs_POM, delimiter = ',')
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/nn_obs_MAOM_' + job_id + '.csv', binn_obs_MAOM, delimiter = ',')
			# print the model structure
			print(model, flush=True)


	else:
		# If resuming from a checkpoint, load loss history
		train_loss_history = checkpoint_worker['train_loss_history']
		train_loss_history_soc = checkpoint_worker['train_loss_history_soc']
		train_loss_history_POM = checkpoint_worker['train_loss_history_POM']
		train_loss_history_MAOM = checkpoint_worker['train_loss_history_MAOM']
		val_loss_history = checkpoint_worker['val_loss_history']
		val_loss_history_soc = checkpoint_worker['val_loss_history_soc']
		val_loss_history_POM = checkpoint_worker['val_loss_history_POM']
		val_loss_history_MAOM = checkpoint_worker['val_loss_history_MAOM']
		train_metrics_history = checkpoint_worker['train_metrics_history']
		train_metrics_history_POM = checkpoint_worker['train_metrics_history_POM']
		train_metrics_history_MAOM = checkpoint_worker['train_metrics_history_MAOM']
		val_metrics_history = checkpoint_worker['val_metrics_history']
		val_metrics_history_POM = checkpoint_worker['val_metrics_history_POM']
		val_metrics_history_MAOM = checkpoint_worker['val_metrics_history_MAOM']
		lr_history = checkpoint_worker['lr_history']
		best_model_epoch = checkpoint_worker['best_model_epoch']

		# Early stopping parameters
		best_val_loss_soc = checkpoint_worker['best_val_loss_soc']
		best_val_loss_pom_maom = checkpoint_worker['best_val_loss_pom_maom']
		best_val_NSE = checkpoint_worker['best_val_NSE'] 
		patience = args.patience
		epochs_without_improvement = checkpoint_worker['epochs_without_improvement']

		# Save observations
		binn_obs_soc = np.ones((wosis_profile_info.shape[0], 200))*np.nan
		binn_obs_soc[current_data_profile_id, :] = current_data_y

		# Define starting epoch
		start_epoch = checkpoint_worker['epoch']

		if rank == 1:
			# print the model structure
			print(model)

	# record start time
	start_time = time.time()
	time_limit_exceeded = False
	whether_break = torch.tensor(0).to(device)

	for iepoch in range(start_epoch, num_epoch):
		epoch_start = time.time()

		# Initialize the break flag for this epoch
		whether_break = torch.tensor(0).to(device)

		# clear gradients
		optimizer.zero_grad()
		model.zero_grad()

		# If using the scheduler, record the learning rate for each epoch
		if scheduler is not None and rank == 0:
			if iepoch == 0:
				# If scheduler has never been stepped, get_last_lr does not work
				# (https://discuss.pytorch.org/t/how-to-retrieve-learning-rate-from-reducelronplateau-scheduler/54234/3).
				# Instead just read the lr from the commandline arg.
				curr_lr = args.lr
			else:
				print("Output of get_last_lr", scheduler.get_last_lr())
				curr_lr = scheduler.get_last_lr()[0]
			print(f"Epoch {iepoch}: lr = {curr_lr}")
			lr_history[iepoch] = curr_lr

		# -------------------------------------training
		total_loss_record_train = list()  # List of Tensors. Each Tensor contains losses in the order of args.losses.
		l1_loss_SOC_record_train = list()
		l1_loss_POM_record_train = list()
		l1_loss_MAOM_record_train = list()
		metrics_record_train_SOC = list()  # List of Tensors (one per batch). Each Tensor contains 3 values: [MSE, MAE, NSE]
		metrics_record_train_POM = list()
		metrics_record_train_MAOM = list()
		ibatch = 0
		model.train()
		if args.use_ddp == 1:
			train_loader.sampler.set_epoch(iepoch)  # Set sampler's epoch number, so we use a different order per epoch
		
		# print(f"Rank {rank} training epoch {iepoch}", flush=True)
		# torch.autograd.set_detect_anomaly(True)   # <- helps debug gradient anomalies but is VERY SLOW
		for batch_info in train_loader:
			batch_x, batch_y, batch_z, batch_c, batch_profile_id, batch_proda_para = batch_info
			if batch_x.shape[0] == 1 and args.use_bn:  # Batch size of 1 during training does not work with BatchNorm
				continue

			ibatch = ibatch + 1
			batch_x = batch_x.to(device)
			batch_y = batch_y.to(device)
			batch_z = batch_z.to(device)
			batch_c = batch_c.to(device)
			batch_proda_para = batch_proda_para.to(device)
			batch_profile_id = batch_profile_id.to(device)

			# KAN: update grid
			if args.model == "kan" and ibatch == 1 and iepoch < 10 and args.kan_update_grid == 1:
				with torch.no_grad():
					model_without_ddp.update_grid(batch_x, batch_z, batch_c, whether_predict=0, one_param_only=args.one_param_only, PRODA_para=batch_proda_para)
					# local_min = batch_x.min(dim=0).values
					# local_max = batch_x.max(dim=0).values
					# global_min = local_min.clone()
					# global_max = local_max.clone()
					# dist.all_reduce(global_min, op=dist.ReduceOp.MIN)
					# dist.all_reduce(global_max, op=dist.ReduceOp.MAX)
					# # Create a dummy tensor to hold the global min/max values
					# dummy_x = torch.empty_like(batch_x)
					# dummy_x[0] = global_min
					# dummy_x[1] = global_max
					# dummy_x[2:] = batch_x[2:].clone()  # Copy the rest of the batch

					# model_without_ddp.update_grid(dummy_x, batch_z, batch_c, whether_predict=0, one_param_only=args.one_param_only, PRODA_para=batch_proda_para)

				dist.barrier()

				# Synchronize the updated grid/coef parameters across all ranks
				for i in range(len(model_without_ddp.mlp.act_fun)):
					dist.all_reduce(model_without_ddp.mlp.act_fun[i].grid, op=dist.ReduceOp.SUM)  # ReduceOp.AVG is more concise but not supported with gloo backend
					model_without_ddp.mlp.act_fun[i].grid.data /= world_size
					dist.all_reduce(model_without_ddp.mlp.act_fun[i].coef, op=dist.ReduceOp.SUM)
					model_without_ddp.mlp.act_fun[i].coef.data /= world_size


				dist.barrier()  # MAKE SURE THIS DOES NOT CAUSE ISSUES. (Old run - this was every batch outside the if statement)
				# print(f"Rank {rank} updated grid/coef parameters for KAN model", flush=True)


			#------------ 1 forward
			# train_nn_start = time.time()
			if args.model in ['gnn', 'spatial']:
				# GNN/spatial models return extra information about spatial smoothness that might be used in loss function
				plot_dir = PLOT_DIR if (ibatch==1 and iepoch%5==0) else None
				batch_y_hat, batch_pred_para, spatial_emb, laplacian = model(batch_x, batch_z, batch_c, whether_predict=0, return_extra=True, plot_dir=plot_dir, one_param_only=args.one_param_only)
			elif args.model == 'binn_hybrid':
				# BINN hybrid model also returns "residual", which could be penalized in loss function
				batch_y_hat, batch_pred_para, residual = model(batch_x, batch_z, batch_c, whether_predict=0, return_residual=True, one_param_only=args.one_param_only)
			else:
				# Normal models just return predicted (1) SOC, (2) parameters
				# print(f"Rank {rank} forward pass for batch {ibatch} started", flush=True)
				batch_y_hat, batch_pred_para = model(batch_x, batch_z, batch_c, whether_predict=0, one_param_only=args.one_param_only, PRODA_para=batch_proda_para)
				# print(f"Rank {rank} forward pass for batch {ibatch} done", flush=True)

			# Check if batch_pred_para is nan or inf
			if torch.isnan(batch_pred_para).any() or torch.isinf(batch_pred_para).any():
				whether_break = torch.tensor(1).to(device)
				for ipara in range(batch_pred_para.shape[0]):
					if torch.isnan(batch_pred_para[ipara]).any() or torch.isinf(batch_pred_para[ipara]).any():
						print(f"Epoch {iepoch} batch {ibatch} parameter {ipara} is {batch_pred_para[ipara]}")

			# Check for extreme para values
			# if args.model != 'nn_only' and (torch.any(batch_pred_para < 0.00001) or torch.any(batch_pred_para > 0.99999)) and ibatch == 1:
			# 	print("Extreme param values")
			# 	print(batch_pred_para[torch.any(((batch_pred_para < 0.00001) | (batch_pred_para > 0.99999)), dim=1), :])
			if rank == 0 and ibatch == 1 and iepoch % 10 == 0:
				print("Predicted para", batch_pred_para)
				if args.model == "nam_joint2":
					print("Predicted f_out", model.module.mlp.f_out.shape, model.module.mlp.f_out[0:5])

			# If KAN, plot activation statistics
			if rank == 0 and args.model == "kan" and (ibatch == 1 and iepoch % 50 == 0):
				import pykan_josh
				plt.switch_backend('Agg')
				model_without_ddp.mlp.plot_activation_statistics(os.path.join(PLOT_DIR, f"epoch{iepoch}_KAN_activation_stats.png"))

			#------------ 2 compute the objective function
			smooth_l1_loss, NSE_SOC, NSE_POM, NSE_MAOM, l1_loss_SOC, l1_loss_POM, l1_loss_MAOM, param_reg_loss = fun_loss(batch_y_hat, batch_y, batch_pred_para)

			# Compute additional losses if using. If we are not using them, set them to nan
			l1_loss = l1_loss_SOC
			l2_loss = l1_loss_SOC
			param_violation_loss = np.nan
			unconstrained_param_loss = np.nan
			param_matching_loss = np.nan
			jacobian_loss = np.nan
			jacobian_sparsity = np.nan
			lipmlp_loss = np.nan
			spectral_loss = np.nan
			# c_reg_loss = np.nan
			cure_loss = np.nan
			nam_l2_loss = np.nan
			nam_entropy_loss = np.nan
			kan_l1_loss = np.nan
			kan_entropy_loss = np.nan
			kan_coef_loss = np.nan
			kan_coefdiff_loss = np.nan
			senn_robustness_loss = np.nan
			senn_l1_loss = np.nan
			senn_sparsity = np.nan
			spatial_error_loss = np.nan
			spatial_smoothness_loss = np.nan
			param_smoothness_loss = np.nan
			residual_loss = np.nan

			# Parameter losses
			if "param_violation" in args.losses:  # Penalty if unconstrained params are outside [-3, 3]. Only used with hardsigmoid.
				param_violation_loss = compute_param_violation_loss(model_without_ddp.unconstrained_params)
			if "unconstrained_param" in args.losses:
				unconstrained_param_loss = compute_unconstrained_param_loss(model_without_ddp.unconstrained_params)
			if "param_matching" in args.losses:  # Penalize extreme parameter values that are far from 0.5
				param_matching_loss = compute_param_matching_loss(batch_pred_para, batch_proda_para)

			# EXPERIMENTAL: Spatial losses
			if "spatial_error" in args.losses:
				assert args.pos_enc != 'none'
				spatial_error_loss = torch.mean(spatial_emb ** 2)
			if "spatial_emb_smoothness" in args.losses:
				assert args.pos_enc != "none"
				spatial_smoothness_loss = 0
				for i in para_index:
					spatial_smoothness_loss += ((spatial_emb[:, i] @ laplacian @ spatial_emb[:, i]) / torch.sum(spatial_emb[:, i]**2))
			if "param_smoothness" in args.losses:
				param_smoothness_loss = 0
				for i in para_index:
					param_smoothness_loss += ((batch_pred_para[:, i] @ laplacian @ batch_pred_para[:, i]) / torch.sum(batch_pred_para[:, i]**2))

			# EXPERIMENTAL: If using BINN_Hybrid, also force the PBM output to be close to groundtruth
			if args.model == "binn_hybrid":
				assert "residual" in args.losses
				residual_loss = torch.mean(torch.abs(residual))

			# EXPERIMENTAL: Lipschitz loss if using
			if args.model == "lipmlp" and "lipmlp" in args.losses:
				lipmlp_loss, cs, scalings = model_without_ddp.mlp.get_lipschitz_loss()
				if ibatch == 1:
					print("Lipschitz c", cs, "Scalings", scalings)
					
			# 	# Compute the Lipschitz constant
			# 	c_reg_loss = reg.lip_constant(conf, model, u_reg, v_reg, mean=conf.reg_all)
			elif "spectral" in args.losses:
				assert args.model == "new_mlp", "Spectral norm regularization only works with --model new_mlp"

				# Compute the spectral norm of the model's layers, and add this as a loss
				spectral_loss = model_without_ddp.mlp.spectral_norm_parallel(device)

			jacobian = None
			if "jacobian" in args.losses:
				# Jacobian L1 loss, which is intended to encourage sparsity in the Jacobian
				# (each parameter should only depend on a few features). Doesn't achieve that
				# very well yet.
				model.eval()
				jacobian = model_without_ddp.get_jacobian()  # [batch, n_params, n_inputs]
				jacobian_sparsity = (jacobian.abs() < 1e-6).float().mean()
				jacobian_loss = jacobian.abs().mean(0).sum()
				model.train()

				# Plot a few jacobians
				if iepoch % 50 == 0 and ibatch == 1 and args.plot:
					print("Plotting Jacobian")
					for example_idx in [0, 1, 2]:
						visualization_utils.plot_matrix(jacobian[example_idx, :, :], row_labels=para_names, col_labels=var4nn,
														filename=os.path.join(PLOT_DIR, f"epoch{iepoch}_jacobian_{example_idx}.png"),
														title=f"Parameter-Feature Jacobian: Epoch {iepoch}, Example {example_idx}")

				# TODO: Try more efficient variations (https://arxiv.org/pdf/1908.02729, https://arxiv.org/pdf/1905.11468)
			# if "jacobian_entropy" in args.losses:
				# TODO

			# if "input_gradient" in args.losses:
			# 	# TODO: Simple input gradient regularization. 
			# 	model_without_ddp.new_input.retain_grad()
			# 	batch_pred_para.backward(torch.ones_like(batch_pred_para), retain_graph=True)
			# 	print("Grad", model_without_ddp.new_input.grad.shape)
			# 	jacobian = model_without_ddp.new_input.grad  # dParam/dInput: [batch, num_param, num_inputs]
			# 	avg_jacobian = jacobian.mean(dim=0)
			# 	jacobian_loss = avg_jacobian.abs().mean()

			if "cure" in args.losses:
				model.eval()
				cure_h = 1  # in the paper, they linearly increase for first 5 epochs
				if jacobian is None:
					jacobian = model_without_ddp.get_jacobian()  #  [batch, n_params, n_inputs]

				# Find direction to perturb
				sum_dParam_dInput = jacobian.sum(1)  # [batch, n_inputs]
				z = torch.sign(sum_dParam_dInput)
				z = z / torch.linalg.vector_norm(z, dim=1, keepdim=True)
				jacobian_perturbed = model_without_ddp.get_jacobian(model_without_ddp.new_input + z * cure_h)  # [batch, n_params, n_inputs]
				# print("Jacobian perturbed", jacobian_perturbed.shape, jacobian_perturbed[0, 0:10, 0:10])
				# print("Jacobian original", jacobian.shape, jacobian[0, 0:10, 0:10])
				cure_loss = (jacobian_perturbed - jacobian).square().sum()			
				model.train()
				# Curvature regularization. TODO Not tested yet
				# cure_loss, grad_norm = misc_utils.regularizer(batch_x, batch_y, batch_z, model, binns_loss_simple)

			if "nam_l2" in args.losses:
				assert args.model in ["nam", "nam_joint", "nam_joint2", "nag"]
				f_out = model_without_ddp.mlp.f_out  # [batch, n_features, n_outputs]
				nam_l2_loss = (f_out ** 2).sum() / (f_out.shape[1] * f_out.shape[2])
			if "nam_entropy" in args.losses:
				assert args.model in ["nam", "nam_joint", "nam_joint2"]
				f_out = model_without_ddp.mlp.f_out  # [batch, n_features, n_outputs]
				variance_explained = torch.var(f_out, dim=0)  # [n_features, n_outputs]
				frac_variance_explained = variance_explained / variance_explained.sum(dim=0, keepdim=True)  # [n_features, n_outputs]. For each output, feature fractions sum to 1
				p_log_p = -frac_variance_explained * torch.log(frac_variance_explained)  # p(x) log p(x) elementwise
				nam_entropy_loss = p_log_p.sum(dim=0).mean()
			if {"kan_l1", "kan_entropy", "kan_coef", "kan_coefdiff"} & set(args.losses):
				assert args.model == "kan"

				# NOTE: the lamb values passed are completely unused, as we direclty obtain the individual loss components and weight them later.
				# For default weights see https://github.com/KindXiaoming/pykan/blob/master/kan/MultKAN.py#L1411
				kan_l1_loss, kan_entropy_loss, kan_coef_loss, kan_coefdiff_loss, kan_entropy_output, kan_entropy_input, kan_output_var = model_without_ddp.mlp.reg(reg_metric='edge_backward', lamb_l1=1., lamb_entropy=1., lamb_coef=1., lamb_coefdiff=1., return_indiv=True)
					# model_without_ddp.mlp.get_reg(reg_metric='node_influence_on_output', lamb_l1=0., lamb_entropy=1., lamb_coef=0., lamb_coefdiff=0.)

			if "senn_robustness" in args.losses:
				model.eval()
				senn_robustness_loss = model_without_ddp.senn_robustness_loss()
				model.train()
			if "senn_l1" in args.losses:
				senn_l1_loss = model_without_ddp.senn_l1_loss()
			if "senn_sparsity" in args.losses:
				senn_sparsity = model_without_ddp.senn_sparsity()

			#------------ 3 cleaning gradients
			optimizer.zero_grad()

			#------------ 4 accumulate partical derivatives of objective respect to parameters
			loss_dict = {"l1_loss": l1_loss,
						"smooth_l1": smooth_l1_loss,
						"l2_loss": l2_loss,
						"Smooth_l1_loss_SOC": l1_loss_SOC,
						"Smooth_l1_loss_POM": l1_loss_POM,
						"Smooth_l1_loss_MAOM": l1_loss_MAOM,
						"param_reg": param_reg_loss,
						"param_violation": param_violation_loss,
						"unconstrained_param": unconstrained_param_loss,
						"param_matching": param_matching_loss,
						"jacobian": jacobian_loss,
						"jacobian_sparsity": jacobian_sparsity,
						"lipmlp": lipmlp_loss,
						# "c_reg": c_reg_loss,
						"spectral": spectral_loss,
						"cure": cure_loss,
						"nam_l2": nam_l2_loss,
						"nam_entropy": nam_entropy_loss,
						"kan_l1": kan_l1_loss,
						"kan_entropy": kan_entropy_loss, # How "uniform" the distribution over edges (connections) is
						"kan_coef": kan_coef_loss,
						"kan_coefdiff": kan_coefdiff_loss, # Linear coefficients
						"senn_robustness": senn_robustness_loss,
						"senn_l1": senn_l1_loss,
						"senn_sparsity": senn_sparsity,
						"spatial_error": spatial_error_loss,
						"spatial_emb_smoothness": spatial_smoothness_loss,
						"param_smoothness": param_smoothness_loss,
						"residual": residual_loss, 
						"kan_entropy_output": kan_entropy_output,
						"kan_entropy_input": kan_entropy_input,
						"kan_output_var": kan_output_var,
						}

			# Store losses in a tensor, in the order of args.losses

			if iepoch <= 0:
				# Seperate training by epochs
				smooth_l1_loss_POM_idx = args.losses.index("Smooth_l1_loss_POM")
				smooth_l1_loss_MAOM_idx = args.losses.index("Smooth_l1_loss_MAOM")
				# Remove the loss from the total loss
				seperate_train_loss = args.losses.copy()
				seperate_train_loss.remove("Smooth_l1_loss_POM")
				seperate_train_loss.remove("Smooth_l1_loss_MAOM")
				seperate_train_losses = torch.stack([loss_dict[loss] for loss in seperate_train_loss]).to(device)
				# remove the weights from the total weights for dot product
				seperate_train_weights = args.lambdas.to(device).clone()
				mask_seperate_train_weights = torch.ones_like(seperate_train_weights, dtype=torch.bool)
				mask_seperate_train_weights[smooth_l1_loss_POM_idx] = False
				mask_seperate_train_weights[smooth_l1_loss_MAOM_idx] = False
				seperate_train_weights = seperate_train_weights[mask_seperate_train_weights]
				total_loss = torch.dot(seperate_train_losses, seperate_train_weights)
				total_loss.backward()
			else:
				train_losses = torch.stack([loss_dict[loss] for loss in args.losses]).to(device)
				if args.loss_weighting in ["manual", "two_stage", "relobralo"]:
					# If loss weights are explicitly set: compute weighted total loss, backpropagate
					total_loss = torch.dot(train_losses, args.lambdas.to(device))
					total_loss.backward()
				else:
					# Specialized multi-task loss balancing methods
					alphas = model.backward(train_losses)
					if ibatch == 1:
						print("Alphas", alphas)
			

			

			# clip gradients. TODO Not tested.
			if args.clip_value != -1:
				# torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)
				torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=args.clip_value)

			# Check gradients
			# print(rank, "GRAD WRT NAM", model.module.mlp.feature_nns[1].layers[0].weight.grad, flush=True)
			# print(rank, "DATA WRT NAM", model.module.mlp.feature_nns[1].layers[0].weight.data, flush=True)

			#------------ 5 step in the opposite direction of the gradient
			# with torch.no_grad(): para = para - eta*para.grad # eta is learning rate
			optimizer.step()
			# print(rank, "after optimizer step", flush=True)
			# print(rank, "DATA WRT NAM AFTER STEP", model.module.mlp.feature_nns[1].layers[0].weight.data, flush=True)

			# Noise
			if args.noise_std > 0:
				misc_utils.inject_noise(model, args.noise_std)

			# Record losses
			total_loss_record_train.append(total_loss) 
			l1_loss_SOC_record_train.append(l1_loss_SOC)
			l1_loss_POM_record_train.append(l1_loss_POM)
			l1_loss_MAOM_record_train.append(l1_loss_MAOM)
			metrics_record_train_SOC.append(NSE_SOC)
			metrics_record_train_POM.append(NSE_POM)
			metrics_record_train_MAOM.append(NSE_MAOM)


			# flush all printed output
			# sys.stdout.flush()
		# end for batch_info in train_loader:

		# Ensure all processes reach this point to synchronize
		# Use all_reduce to check if any process has encountered NaN
		dist.all_reduce(whether_break, op=dist.ReduceOp.MAX)

		# Check if batch_pred_para is nan or inf
		if whether_break.item() == 1:
			print(f"Process {rank} breaking after epoch {iepoch}")
			break  # Break out of the epoch loop if NaN detected in any process

		# training time
		train_time = time.time() - epoch_start

		# Ensure all processes reach this point before proceeding
		dist.barrier()

		# -------------------------------------validation
		total_loss_record_val = list()  # List of Tensors (one per batch). Each Tensor contains losses in the order of args.losses.
		l1_loss_SOC_record_val = list()
		l1_loss_POM_record_val = list()
		l1_loss_MAOM_record_val = list()
		metrics_record_val_SOC = list()  # List of Tensors (one per batch). Each Tensor contains [MSE, MAE, NSE]
		metrics_record_val_POM = list()
		metrics_record_val_MAOM = list()
		ibatch = 0
		model.eval()
		with torch.no_grad():
			for batch_info in val_loader:
				batch_x, batch_y, batch_z, batch_c, batch_profile_id, batch_proda_para = batch_info
				ibatch = ibatch + 1
				batch_x = batch_x.to(device)
				batch_y = batch_y.to(device)
				batch_z = batch_z.to(device)
				batch_c = batch_c.to(device)
				batch_proda_para = batch_proda_para.to(device)
				batch_profile_id = batch_profile_id.to(device)

				# 1 forward
				if args.model in ['gnn', 'spatial']:
					plot_dir = PLOT_DIR if (ibatch==1 and iepoch%5==0) else None
					batch_y_hat, batch_pred_para, spatial_emb, laplacian = model(batch_x, batch_z, batch_c, whether_predict=0, return_extra=True)
				elif args.model == 'binn_hybrid':
					batch_y_hat, batch_pred_para, residual = model(batch_x, batch_z, batch_c, whether_predict=0, return_residual=True)
				else:
					batch_y_hat, batch_pred_para = model(batch_x, batch_z, batch_c, whether_predict=0, PRODA_para=batch_proda_para)

				# 2 compute the objective function
				smooth_l1_loss, NSE_SOC, NSE_POM, NSE_MAOM, l1_loss_SOC, l1_loss_POM, l1_loss_MAOM, param_reg_loss = fun_loss(batch_y_hat, batch_y, batch_pred_para)

				# Compute additional losses if using. Not strictly necessary but this helps us see if there
				# is a difference between the losses for train/validation sets
				# If we are not using them, set them to nan
				l1_loss = l1_loss_SOC
				l2_loss = l1_loss_SOC
				param_violation_loss = np.nan
				unconstrained_param_loss = np.nan
				param_matching_loss = np.nan
				jacobian_loss = np.nan
				lipmlp_loss = np.nan
				spectral_loss = np.nan
				# c_reg_loss = np.nan
				nam_l2_loss = np.nan
				nam_entropy_loss = np.nan
				kan_l1_loss = np.nan
				kan_entropy_loss = np.nan
				kan_coef_loss = np.nan
				kan_coefdiff_loss = np.nan
				senn_robustness_loss = np.nan
				senn_l1_loss = np.nan
				senn_sparsity = np.nan
				cure_loss = np.nan
				spatial_error_loss = np.nan
				spatial_smoothness_loss = np.nan
				param_smoothness_loss = np.nan
				residual_loss = np.nan

				# Spatial losses
				if "spatial_error" in args.losses:
					assert args.pos_enc != 'none'
					spatial_error_loss = torch.mean(spatial_emb ** 2)
				if "spatial_emb_smoothness" in args.losses:
					assert args.pos_enc != "none"
					spatial_smoothness_loss = 0
					for i in para_index:
						spatial_smoothness_loss += ((spatial_emb[:, i] @ laplacian @ spatial_emb[:, i]) / torch.sum(spatial_emb[:, i]**2))
				if "param_smoothness" in args.losses:
					param_smoothness_loss = 0
					for i in para_index:
						param_smoothness_loss += ((batch_pred_para[:, i] @ laplacian @ batch_pred_para[:, i]) / torch.sum(batch_pred_para[:, i]**2))
				if args.model == "binn_hybrid":
					residual_loss = torch.mean(torch.abs(residual))
				if "param_violation" in args.losses:
					param_violation_loss = compute_param_violation_loss(model_without_ddp.unconstrained_params)
				if "unconstrained_param" in args.losses:
					unconstrained_param_loss = compute_unconstrained_param_loss(model_without_ddp.unconstrained_params)
				if "param_matching" in args.losses:
					param_matching_loss = compute_param_matching_loss(batch_pred_para, batch_proda_para)

				if args.model == "lipmlp" and "lipmlp" in args.losses:
					lipmlp_loss, cs, scalings = model_without_ddp.mlp.get_lipschitz_loss()
					if ibatch == 1:
						print("Lipschitz c", cs, "Scalings", scalings)
				# elif args.model == "clip":
				# 	# ---------------------------------------------------------------------
				# 	# Adverserial update
				# 	# ---------------------------------------------------------------------
				# 	# get initialization for Lipschitz Training set
				# 	if ((cache['counter'] % conf.reg_incremental) == 0) or (not ('init' in cache)):
				# 		if verbosity > 0:
				# 			print('The Lipschitz set was reset')
				# 		cache['init'] = reg.u_v_init(conf, lip_cycle, cache)
				# 		cache['counter'] = 1
				# 	else:
				# 		cache['counter'] += 1

				# 	# adverserial update on the Lipschitz set
				# 	u, v = reg.search_u_v(conf, model, cache)
				# 	# ---------------------------------------------------------------------

				# 	# Use either all tuples or only one tuple for regularization
				# 	if conf.reg_all:
				# 		u_reg, v_reg = u, v
				# 	else:
				# 		# Use idx:idx+1 to keep shape
				# 		u_reg = u[cache["idx"]:cache["idx"] + 1].detach()
				# 		v_reg = v[cache["idx"]:cache["idx"] + 1].detach()

				# 	# Compute the Lipschitz constant
				# 	c_reg_loss = reg.lip_constant(conf, model, u_reg, v_reg, mean=conf.reg_all)
				elif args.model == "new_mlp" and "spectral" in args.losses:
					# Compute the spectral norm of the model's layers, and add this as a loss
					spectral_loss = model_without_ddp.mlp.spectral_norm_parallel(device)

				if "jacobian" in args.losses:
					# # Use jacobian L1
					jacobian = model_without_ddp.get_jacobian()  # [batch, n_params, n_inputs]
					jacobian_sparsity = (jacobian.abs() < 1e-6).float().mean()
					jacobian_loss = jacobian.abs().mean(0).sum()

				if "cure" in args.losses:
					cure_h = 1  # in the paper, they linearly increase for first 5 epochs
					if "jacobian" not in args.losses:
						jacobian = model_without_ddp.get_jacobian()  # [batch, n_params, n_inputs]

					# Find direction to perturb
					sum_dParam_dInput = jacobian.sum(1)  # [batch, n_inputs], summed over params
					z = torch.sign(sum_dParam_dInput)
					z = z / torch.linalg.vector_norm(z, dim=1, keepdim=True)
					jacobian_perturbed = model_without_ddp.get_jacobian(model_without_ddp.new_input + z * cure_h)  # [batch, n_params, n_inputs]
					cure_loss = (jacobian_perturbed - jacobian).square().sum()			

				if "nam_l2" in args.losses:
					assert args.model in ["nam", "nam_joint", "nam_joint2", "nag"]
					f_out = model_without_ddp.mlp.f_out  # [batch, n_features, n_outputs]
					nam_l2_loss = (f_out ** 2).sum() / (f_out.shape[1] * f_out.shape[2])
				if "nam_entropy" in args.losses:
					assert args.model in ["nam", "nam_joint", "nam_joint2"]
					f_out = model_without_ddp.mlp.f_out  # [batch, n_features, n_outputs]
					variance_explained = torch.var(f_out, dim=0)  # [n_features, n_outputs]
					frac_variance_explained = variance_explained / variance_explained.sum(dim=0, keepdim=True)  # [n_features, n_outputs]. For each output, feature fractions sum to 1
					p_log_p = -frac_variance_explained * torch.log(frac_variance_explained)  # p(x) log p(x) elementwise
					nam_entropy_loss = p_log_p.sum(dim=0).mean()

				if {"kan_l1", "kan_entropy", "kan_coef", "kan_coefdiff"} & set(args.losses):
					assert args.model == "kan"

					# NOTE: the lamb values passed are completely unused, as we direclty obtain the individual loss components and weight them later.
					kan_l1_loss, kan_entropy_loss, kan_coef_loss, kan_coefdiff_loss, kan_entropy_output, kan_entropy_input, kan_output_var = model_without_ddp.mlp.reg(reg_metric='edge_backward', lamb_l1=1., lamb_entropy=1., lamb_coef=1., lamb_coefdiff=1., return_indiv=True)

				if "senn_robustness" in args.losses:
					senn_robustness_loss = model_without_ddp.senn_robustness_loss()
				if "senn_l1" in args.losses:
					senn_l1_loss = model_without_ddp.senn_l1_loss()
				if "senn_sparsity" in args.losses:
					senn_sparsity = model_without_ddp.senn_sparsity()

				loss_dict = {"l1_loss": l1_loss,
				 			"smooth_l1": smooth_l1_loss,
							"l2_loss": l2_loss,
							"Smooth_l1_loss_SOC": l1_loss_SOC,
							"Smooth_l1_loss_POM": l1_loss_POM,
							"Smooth_l1_loss_MAOM": l1_loss_MAOM,
							"param_reg": param_reg_loss,
							"param_violation": param_violation_loss,
							"unconstrained_param": unconstrained_param_loss,
							"param_matching": param_matching_loss,
							"jacobian": jacobian_loss,
							"jacobian_sparsity": jacobian_sparsity,
							"lipmlp": lipmlp_loss,
							# "c_reg": c_reg_loss,
							"spectral": spectral_loss,
							"cure": cure_loss,
							"nam_l2": nam_l2_loss,
							"nam_entropy": nam_entropy_loss,
							"kan_l1": kan_l1_loss,
							"kan_entropy": kan_entropy_loss,
							"kan_coef": kan_coef_loss,
							"kan_coefdiff": kan_coefdiff_loss,
							"senn_robustness": senn_robustness_loss,
							"senn_l1": senn_l1_loss,
							"senn_sparsity": senn_sparsity,
							"spatial_error": spatial_error_loss,
							"spatial_emb_smoothness": spatial_smoothness_loss,
							"param_smoothness": param_smoothness_loss,
							"residual": residual_loss, 
							"kan_entropy_output": kan_entropy_output,
							"kan_entropy_input": kan_entropy_input,
							"kan_output_var": kan_output_var,
							}

				# Record losses
				# Store losses in a tensor, in the order of args.losses
				val_losses = torch.stack([loss_dict[loss] for loss in args.losses]).to(device)
				val_losses = torch.dot(val_losses, args.lambdas.to(device))
				total_loss_record_val.append(val_losses)
				l1_loss_SOC_record_val.append(l1_loss_SOC)
				l1_loss_POM_record_val.append(l1_loss_POM)
				l1_loss_MAOM_record_val.append(l1_loss_MAOM)
				metrics_record_val_SOC.append(NSE_SOC)
				metrics_record_val_POM.append(NSE_POM)
				metrics_record_val_MAOM.append(NSE_MAOM)


				# print(f'{datetime.now()}, Epoch {iepoch + 1}, Rank {rank}, batch {ibatch}, validation loss: {obj.item():.2f}')
				# record prediction
				# Check the shape of the predictions
				# if batch_profile_id.shape[0] != args.batch_size:
				# 	print(f"ibatch {ibatch} Rank {rank} batch_y_hat shape {batch_y_hat.shape}")
				# 	# Add nan values to the end of the tensor
				# 	batch_y_hat = torch.cat((batch_y_hat, torch.ones(args.batch_size - batch_profile_id.shape[0], 200)*np.nan), dim=0)
				# 	batch_pred_para = torch.cat((batch_pred_para, torch.ones(args.batch_size - batch_profile_id.shape[0], len(para_names))*np.nan), dim=0)
				# 	batch_profile_id = torch.cat((batch_profile_id, torch.ones(args.batch_size - batch_profile_id.shape[0])*np.nan), dim=0)
				# val_pred_soc_rank[ibatch, :, :] = batch_y_hat
				# val_pred_para_rank[ibatch, :, :] = batch_pred_para
				# val_profile_id_rank[ibatch, :] = batch_profile_id

				ibatch = ibatch + 1
		# end for batch_info in val_loader: 
			
		# # record the time
		hist_time = time.time() - start_time

		# Gather losses from all processes
		all_train_losses = [torch.tensor(0.0, device=device) for _ in range(world_size)]
		all_l1_train_losses = [torch.tensor(0.0, device=device) for _ in range(world_size)]
		all_l1_POM_train_losses = [torch.tensor(0.0, device=device) for _ in range(world_size)]
		all_l1_MAOM_train_losses = [torch.tensor(0.0, device=device) for _ in range(world_size)]

		all_val_losses = [torch.tensor(0.0, device=device) for _ in range(world_size)]
		all_l1_val_losses = [torch.tensor(0.0, device=device) for _ in range(world_size)]
		all_l1_POM_val_losses = [torch.tensor(0.0, device=device) for _ in range(world_size)]
		all_l1_MAOM_val_losses = [torch.tensor(0.0, device=device) for _ in range(world_size)]

		all_train_times = [torch.tensor(0.0, device=device) for _ in range(world_size)]
		all_train_NSE_soc = [torch.tensor(0.0, device=device) for _ in range(world_size)]
		all_train_NSE_POM = [torch.tensor(0.0, device=device) for _ in range(world_size)]
		all_train_NSE_MAOM = [torch.tensor(0.0, device=device) for _ in range(world_size)]

		all_val_NSE_soc = [torch.tensor(0.0, device=device) for _ in range(world_size)]
		all_val_NSE_POM = [torch.tensor(0.0, device=device) for _ in range(world_size)]
		all_val_NSE_MAOM = [torch.tensor(0.0, device=device) for _ in range(world_size)]
		all_hist_times = [torch.tensor(0.0, device=device) for _ in range(world_size)]

		dist.all_gather(all_train_losses, torch.tensor(total_loss_record_train, device=device).mean())
		dist.all_gather(all_l1_train_losses, torch.tensor(l1_loss_SOC_record_train, device=device).mean())
		dist.all_gather(all_l1_POM_train_losses, torch.tensor(l1_loss_POM_record_train, device=device).mean())
		dist.all_gather(all_l1_MAOM_train_losses, torch.tensor(l1_loss_MAOM_record_train, device=device).mean())

		dist.all_gather(all_val_losses, torch.tensor(total_loss_record_val, device=device).mean())
		dist.all_gather(all_l1_val_losses, torch.tensor(l1_loss_SOC_record_val, device=device).mean())
		dist.all_gather(all_l1_POM_val_losses, torch.tensor(l1_loss_POM_record_val, device=device).mean())
		dist.all_gather(all_l1_MAOM_val_losses, torch.tensor(l1_loss_MAOM_record_val, device=device).mean())

		dist.all_gather(all_train_times, torch.tensor(train_time, device=device))
		dist.all_gather(all_train_NSE_soc, torch.tensor(metrics_record_train_SOC, device=device).mean())
		dist.all_gather(all_train_NSE_POM, torch.tensor(metrics_record_train_POM, device=device).mean())
		dist.all_gather(all_train_NSE_MAOM, torch.tensor(metrics_record_train_MAOM, device=device).mean())

		dist.all_gather(all_val_NSE_soc, torch.tensor(metrics_record_val_SOC, device=device).mean())
		dist.all_gather(all_val_NSE_POM, torch.tensor(metrics_record_val_POM, device=device).mean())
		dist.all_gather(all_val_NSE_MAOM, torch.tensor(metrics_record_val_MAOM, device=device).mean())
		dist.all_gather(all_hist_times, torch.tensor(hist_time, device=device))


		# record the loss history
		train_loss_history[iepoch, :] = torch.stack(all_train_losses).mean().detach().cpu().numpy()
		train_loss_history_soc[iepoch, :] = torch.stack(all_l1_train_losses).mean().detach().cpu().numpy()
		train_loss_history_POM[iepoch, :] = torch.stack(all_l1_POM_train_losses).mean().detach().cpu().numpy()
		train_loss_history_MAOM[iepoch, :] = torch.stack(all_l1_MAOM_train_losses).mean().detach().cpu().numpy()
		val_loss_history[iepoch, :] = torch.stack(all_val_losses).mean().detach().cpu().numpy()
		val_loss_history_soc[iepoch, :] = torch.stack(all_l1_val_losses).mean().detach().cpu().numpy()
		val_loss_history_POM[iepoch, :] = torch.stack(all_l1_POM_val_losses).mean().detach().cpu().numpy()
		val_loss_history_MAOM[iepoch, :] = torch.stack(all_l1_MAOM_val_losses).mean().detach().cpu().numpy()
		
		train_metrics_history[iepoch, :] = torch.stack(all_train_NSE_soc).mean().detach().cpu().numpy()
		train_metrics_history_POM[iepoch, :] = torch.stack(all_train_NSE_POM).mean().detach().cpu().numpy()
		train_metrics_history_MAOM[iepoch, :] = torch.stack(all_train_NSE_MAOM).mean().detach().cpu().numpy()
		val_metrics_history[iepoch, :] = torch.stack(all_val_NSE_soc).mean().detach().cpu().numpy()
		val_metrics_history_POM[iepoch, :] = torch.stack(all_val_NSE_POM).mean().detach().cpu().numpy()
		val_metrics_history_MAOM[iepoch, :] = torch.stack(all_val_NSE_MAOM).mean().detach().cpu().numpy()



		# Save loss history
		if rank == 0:
			print(f'{datetime.now()} - Epoch {iepoch} Rank {rank} - Train losses: {torch.stack(all_train_losses).mean():.4f}, Train SOC losses: {torch.stack(all_l1_train_losses).mean():.4f}, Train POM losses: {torch.stack(all_l1_POM_train_losses).mean():.4f}, Train MAOM losses: {torch.stack(all_l1_MAOM_train_losses).mean():.4f}', flush=True)
			print(f'{datetime.now()} - Epoch {iepoch} Rank {rank} - Val losses: {torch.stack(all_val_losses).mean():.4f}, Val SOC losses: {torch.stack(all_l1_val_losses).mean():.4f}, Val POM losses: {torch.stack(all_l1_POM_val_losses).mean():.4f}, Val MAOM losses: {torch.stack(all_l1_MAOM_val_losses).mean():.4f}', flush=True)
			print(f'{datetime.now()} - Epoch {iepoch} Rank {rank} - Train SOC NSE: {torch.stack(all_train_NSE_soc).mean():.2f}, Train POM NSE: {torch.stack(all_train_NSE_POM).mean():.2f}, Train MAOM NSE: {torch.stack(all_train_NSE_MAOM).mean():.2f}, time: {torch.stack(all_train_times).mean():.2f}', flush=True)
			print(f'{datetime.now()} - Epoch {iepoch} Rank {rank} - Val SOC NSE: {torch.stack(all_val_NSE_soc).mean():.2f}, Val POM NSE: {torch.stack(all_val_NSE_POM).mean():.2f}, Val MAOM NSE: {torch.stack(all_val_NSE_MAOM).mean():.2f}, time: {torch.stack(all_hist_times).mean():.2f}', flush=True)
			sys.stdout.flush()

			# Relobralo update
			if args.loss_weighting == "relobralo" and iepoch >= 1:
				with torch.no_grad():
					loss_curr = torch.tensor([train_loss_history[iepoch, loss_idx] for loss_idx in range(len(args.losses))])
					loss_prev = torch.tensor([train_loss_history[iepoch-1, loss_idx] for loss_idx in range(len(args.losses))])
					loss_init = torch.tensor([train_loss_history[0, loss_idx] for loss_idx in range(len(args.losses))])
					lambda_bal_prev = F.softmax(loss_curr / (args.relobralo_temp * loss_prev), dim=0)  # Based on ratio of current loss & prev epoch loss
					print("Lambda bal prev", lambda_bal_prev)
					lambda_bal_init = F.softmax(loss_curr / (args.relobralo_temp * loss_init), dim=0)  # Based on ratio of current loss & epoch 0 loss
					lambda_hist = args.relobralo_saudade * args.lambdas + (1-args.relobralo_saudade) * lambda_bal_init
					args.lambdas = args.relobralo_alpha * lambda_hist + (1-args.relobralo_alpha) * lambda_bal_prev
					args.lambdas = args.lambdas / args.lambdas.sum()
					print("lambdas", args.lambdas)

					# Broadcast lambda update to all ranks. TODO Not tested yet
					dist.broadcast(args.lambdas, src=0)

			# If this model is the best so far, save the checkpoint into 'opt_nn_{job_id}.pt'
			# if val_metrics_history[iepoch, :] <= best_val_NSE: 
			# if val_loss_history_soc[iepoch, :] + val_loss_history_POM[iepoch, :] + val_loss_history_MAOM[iepoch, :] < best_val_loss_soc + best_val_loss_pom_maom:
			if val_loss_history_soc[iepoch, :] < best_val_loss_soc and val_loss_history_POM[iepoch, :] + val_loss_history_MAOM[iepoch, :] < best_val_loss_pom_maom:
				print(f'Best model updated at epoch {iepoch}')
				best_model_epoch = torch.tensor(iepoch, device=device)

				checkpoint_best_model = {
					'epoch': iepoch,
					'model_state_dict': model.state_dict(),
					'model_kwargs': model_kwargs,  # Save kwargs used to construct the model
					'optimizer_state_dict': optimizer.state_dict(),
					'best_val_loss_soc': best_val_loss_soc,
					'best_val_loss_POM_MAOM': best_val_loss_pom_maom,
					'best_val_NSE': best_val_NSE,
					'best_model_epoch': best_model_epoch,
					'train_loss_history': train_loss_history,
					'train_loss_history_soc': train_loss_history_soc,
					'train_loss_history_POM': train_loss_history_POM,
					'train_loss_history_MAOM': train_loss_history_MAOM,
					'val_loss_history': val_loss_history,
					'val_loss_history_soc': val_loss_history_soc,
					'val_loss_history_POM': val_loss_history_POM,
					'val_loss_history_MAOM': val_loss_history_MAOM,
					'train_metrics_history': train_metrics_history,
					'train_metrics_history_POM': train_metrics_history_POM,
					'train_metrics_history_MAOM': train_metrics_history_MAOM,
					'val_metrics_history': val_metrics_history,
					'val_metrics_history_POM': val_metrics_history_POM,
					'val_metrics_history_MAOM': val_metrics_history_MAOM,
					'lr_history': lr_history,
					'train_indices': train_loc,
					'val_indices': val_loc,
					'test_indices': test_loc,
					'epochs_without_improvement': epochs_without_improvement,
					'args': args,  # Commandline args
				}
				
				best_model_path = data_dir_output + 'neural_network/' + job_id + '/opt_nn_' + job_id  + '.pt'
				torch.save(checkpoint_best_model, best_model_path)

				# # Test with the best model
				# best_guess_test_y_hat, best_guess_test_pred_para = model_without_ddp(test_x.to(device), test_z.to(device), test_c.to(device),
				# 																	whether_predict=0, PRODA_para=test_proda_para.to(device))
				# test_smooth_l1_loss, test_NSE_SOC, test_NSE_POM, test_NSE_MAOM, test_l1_loss_SOC, test_l1_loss_POM, test_l1_loss_MAOM, test_param_reg_loss = fun_loss(best_guess_test_y_hat, test_y.to(device), best_guess_test_pred_para)
				# print(f'Test - smooth_l1_loss: {test_smooth_l1_loss.item():.2f}, NSE_SOC: {test_NSE_SOC.item():.2f}, NSE_POM: {test_NSE_POM.item():.2f}, NSE_MAOM: {test_NSE_MAOM.item():.2f}, l1_loss_SOC: {test_l1_loss_SOC.item():.2f}, l1_loss_POM: {test_l1_loss_POM.item():.2f}, l1_loss_MAOM: {test_l1_loss_MAOM.item():.2f}')

				# # Save the test results
				# with open(os.path.join(data_dir_output, "neural_network", job_id, "test_results.txt"), 'a') as f:
				# 	f.write(f"Epoch {iepoch}, Test smooth_l1_loss: {test_smooth_l1_loss.item():.2f}, NSE_SOC: {test_NSE_SOC.item():.2f}, NSE_POM: {test_NSE_POM.item():.2f}, NSE_MAOM: {test_NSE_MAOM.item():.2f}, l1_loss_SOC: {test_l1_loss_SOC.item():.2f}, l1_loss_POM: {test_l1_loss_POM.item():.2f}, l1_loss_MAOM: {test_l1_loss_MAOM.item():.2f}, epoch: {iepoch}\n")

				# KAN-specific visualizations
				if args.model == "kan":
					# Produce edge/node importance scores
					model_without_ddp.mlp.attribute()
					model_without_ddp.mlp.node_attribute()

					# dump scores to txt
					txt_dir = os.path.join(PLOT_DIR, "importance_txt")
					os.makedirs(txt_dir, exist_ok=True)

					for l, s in enumerate(model_without_ddp.mlp.edge_scores):
						# convert to NumPy on CPU
						arr = s.detach().cpu().numpy()
						# prepend header row with input names for readability
						header = "\t".join([""] + list(var4nn))
						np.savetxt(
							f"{txt_dir}/edge_scores_layer{l}_epoch{iepoch}.txt",
							arr,
							delimiter="\t",
							header=header,
							comments=""
						)

					for l, s in enumerate(model_without_ddp.mlp.node_scores):
						arr = s.detach().cpu().numpy()
						np.savetxt(
							f"{txt_dir}/node_scores_layer{l}_epoch{iepoch}.txt",
							arr,
							fmt="%.6f",
							newline="\n"
						)        

					# Plot the unpruned model
					model_without_ddp.mlp.plot(folder=os.path.join(PLOT_DIR, str(iepoch), "splines"), in_vars=var4nn, out_vars=para_names, scale=5, varscale=0.1)
					plt.savefig(os.path.join(PLOT_DIR, f"epoch{iepoch}_kan_plot.png"))
					plt.close()

					# Plot the pruned model
					pruned_model = model_without_ddp.mlp.prune(node_th=0.01, edge_th=0.01)
					pruned_model.plot(folder=os.path.join(PLOT_DIR, str(iepoch), "splines"), in_vars=var4nn, out_vars=para_names, scale=5, varscale=0.1)
					plt.savefig(os.path.join(PLOT_DIR, f"epoch{iepoch}_kan_plot_pruned.png"))
					plt.close()

			
				
		
		# Add a early stopping condition
		# if val_loss_history_soc[iepoch, :] + val_loss_history_POM[iepoch, :] + val_loss_history_MAOM[iepoch, :] < best_val_loss_soc + best_val_loss_pom_maom:
		if val_loss_history_soc[iepoch, :] < best_val_loss_soc and val_loss_history_POM[iepoch, :] + val_loss_history_MAOM[iepoch, :] < best_val_loss_pom_maom:
			best_val_NSE = (val_metrics_history[iepoch, :] + val_metrics_history_POM[iepoch, :] + val_metrics_history_MAOM[iepoch, :]) / 3
			best_val_loss_soc = val_loss_history_soc[iepoch, :]
			best_val_loss_pom_maom = val_loss_history_POM[iepoch, :] + val_loss_history_MAOM[iepoch, :]
			epochs_without_improvement = 0
			best_model_epoch = torch.tensor(iepoch, device=device)
			# Optionally save the model here if it's the best one so far
		else:
			epochs_without_improvement += 1

		dist.all_reduce(best_model_epoch, op=dist.ReduceOp.MAX)
		dist.barrier()

		if rank == 0:
			# Save the training and validation loss history into txt files
			with open(os.path.join(data_dir_output, "neural_network", job_id, avg_loss_filename), 'a') as f:
				f.write(f"{iepoch}, {torch.stack(all_train_losses).mean():.4f}, {torch.stack(all_val_losses).mean():.4f}, {torch.stack(all_train_times).mean():.2f}, {torch.stack(all_hist_times).mean():.2f}, {best_model_epoch.item()}\n")
			with open(os.path.join(data_dir_output, "neural_network", job_id, avg_l1_loss_filename), 'a') as f:
				f.write(f"{iepoch}, {torch.stack(all_l1_train_losses).mean():.4f}, {torch.stack(all_l1_POM_train_losses).mean():.4f}, {torch.stack(all_l1_MAOM_train_losses).mean():.4f}, {torch.stack(all_l1_val_losses).mean():.4f}, {torch.stack(all_l1_POM_val_losses).mean():.4f}, {torch.stack(all_l1_MAOM_val_losses).mean():.4f}, {torch.stack(all_train_times).mean():.2f}, {torch.stack(all_hist_times).mean():.2f}, {best_model_epoch.item()}\n")
			with open(os.path.join(data_dir_output, "neural_network", job_id, avg_NSE_filename), 'a') as f:
				f.write(f"{iepoch}, {torch.stack(all_train_NSE_soc).mean():.2f}, {torch.stack(all_train_NSE_POM).mean():.2f}, {torch.stack(all_train_NSE_MAOM).mean():.2f}, {torch.stack(all_val_NSE_soc).mean():.2f}, {torch.stack(all_val_NSE_POM).mean():.2f}, {torch.stack(all_val_NSE_MAOM).mean():.2f}, {torch.stack(all_train_times).mean():.2f}, {torch.stack(all_hist_times).mean():.2f}, {best_model_epoch.item()}\n")


		# # Add a learning rate scheduler
		if iepoch >= args.bias_only_epochs:
			if args.use_swa and all_val_NSE_soc.item() < 0.5 and iepoch > swa_start:
				swa_model.update_parameters(model)
				swa_scheduler.step()
			elif args.scheduler == "reduce_on_plateau":
				old_lr = scheduler.get_last_lr()[0] if iepoch > 0 else args.lr
				scheduler.step(all_val_NSE_soc)
				new_lr = scheduler.get_last_lr()[0]
				if new_lr != old_lr:
					# Revert back to the best model so far
					model_ckpt = torch.load(data_dir_output + 'neural_network/' + job_id + '/opt_nn_' + job_id + '.pt', map_location=device, weights_only=False)
					print(f"Reducing LR {old_lr} to {new_lr}. Reverting to model at epoch {model_ckpt['epoch']}")
					model.load_state_dict(model_ckpt['model_state_dict'])

			elif scheduler is not None:
				scheduler.step()
			if rank == 0 and scheduler is not None:
				print("New learning rate =", scheduler.get_last_lr())

		# if args.loss_weighting == "two_stage" and iepoch > args.second_start:
		# 	args.lambdas = args.second_lambdas

		dist.barrier()  # Wait for all ranks to finish the epoch
		# Early stopping condition
		if epochs_without_improvement == patience:
			print("Rank {}: Early stopping due to no improvement after {} epochs.".format(rank, patience))
			break  # exit the epoch loop

		# If runtimes are over 11.30 hours, save checkpoint and exit
		whether_checkpoint = False
		if time.time() - job_begin_time > 41400:
			if rank == 0:
				print("Rank {}: Runtime exceeded, saving checkpoint and exiting.".format(rank))
				# break # at this point, no longer pass the time limit
				whether_checkpoint = True
				checkpoint = {
					'epoch': iepoch,
					'model_state_dict': model.state_dict(),
					'model_kwargs': model_kwargs,  # Save kwargs used to construct the model
					'optimizer_state_dict': optimizer.state_dict(),
					'best_val_loss_soc': best_val_loss_soc,
					'best_val_loss_POM_MAOM': best_val_loss_pom_maom,
					'best_val_NSE': best_val_NSE,
					'best_model_epoch': best_model_epoch,
					'train_loss_history': train_loss_history,
					'train_loss_history_soc': train_loss_history_soc,
					'train_loss_history_POM': train_loss_history_POM,
					'train_loss_history_MAOM': train_loss_history_MAOM,
					'val_loss_history': val_loss_history,
					'val_loss_history_soc': val_loss_history_soc,
					'val_loss_history_POM': val_loss_history_POM,
					'val_loss_history_MAOM': val_loss_history_MAOM,
					'train_metrics_history': train_metrics_history,
					'train_metrics_history_POM': train_metrics_history_POM,
					'train_metrics_history_MAOM': train_metrics_history_MAOM,
					'val_metrics_history': val_metrics_history,
					'val_metrics_history_POM': val_metrics_history_POM,
					'val_metrics_history_MAOM': val_metrics_history_MAOM,
					'lr_history': lr_history,
					'train_indices': train_loc,
					'val_indices': val_loc,
					'test_indices': test_loc,
					'epochs_without_improvement': epochs_without_improvement,
					'args': args,  # Commandline args
				}
				if args.use_swa:
					checkpoint['swa_model_state_dict'] = swa_model.state_dict()
			
				torch.save(checkpoint, data_dir_output + 'neural_network/' + job_id + '/checkpoint_' + job_id + '.pt')

				# Create a file to submit the job again
				with open(job_submit_path + 'Resume' + job_id + '.submit', 'w') as f:
					f.write(f'#!/bin/bash\n')
					f.write(f'#PBS -A UCOR0092\n')
					f.write(f'#PBS -N DDP_BINN_POM_MAOM_Resume\n')
					f.write(f'#PBS -q main\n')
					f.write(f'#PBS -l walltime=12:00:00\n')
					f.write(f'#PBS -l select=1:ncpus=128\n\n')
					f.write(f'# Use scratch for temporary files to avoid space limits in /tmp\n')
					f.write(f'export TMPDIR=/glade/scratch/$USER/temp\n')
					f.write(f'mkdir -p $TMPDIR\n\n')
					f.write(f'# Load modules to match compile-time environment\n')
					f.write(f'module purge\n')
					f.write(f'module load conda\n')
					f.write(f'module load cuda\n\n')
					f.write(f'# Activate environment in conda\n')
					f.write(f'conda activate BINN_310_CPU\n\n')
					f.write(f'# Start the Python Code\n')
					f.write(f'python -u /glade/u/home/haodixu/BINN/Server_Script/BINN_COMPAS/binns_DDP.py {" ".join(sys.argv)} --whether_resume 1\n')

				# submit the job again
				submit_command = ['qsub', 
					'-v', f"PREVIOUS_JOB_ID={job_id}",
					job_submit_path + 'Resume' + job_id + '.submit']
				# Submit the job and get the new job ID
				try:
					submit_output = subprocess.check_output(submit_command, universal_newlines=True)
					new_job_id = submit_output.strip()
					print(f"New job submitted. New Job ID is {new_job_id}")
				except subprocess.CalledProcessError as e:
					print(f"Failed to submit job: {e.output}")
			break


	print(f"Rank {rank} finished processing data.")

	if time_limit_exceeded:
		print(f"Rank {rank}: Exiting after saving checkpoint.")
		dist.destroy_process_group()
		return
	if whether_break.item() == 1:
		print(f"Rank {rank}: Exiting after training due to NaN encountered in any process.")
		dist.destroy_process_group()
		return

	dist.barrier()
				



	##################################################
	# Done training. Load best model for analysis
	##################################################
	# TODO: weights_only=False is not recommended. Should modify code to only save tensors.
	new_checkpoint = torch.load(data_dir_output + 'neural_network/' + job_id + '/opt_nn_' + job_id + '.pt', map_location=device, weights_only=False)
	# new_checkpoint = torch.load('D:/BINN/OUTPUT_DATA/neural_network/20250418-142035_BINN_Global_COMPAS_with_KAN/opt_nn_20250418-142035_BINN_Global_COMPAS_with_KAN.pt', map_location=device, weights_only=False)
	if args.use_swa:
		# Update batchnorm stats of averaged model (required when using SWA)
		misc_utils.update_bn_custom(train_loader, swa_model, device)
		best_guess_model = swa_model
		best_guess_model.load_state_dict(new_checkpoint['swa_model_state_dict'])
	else:
		best_guess_model = model  # Do not need to create a new model
		best_guess_model.load_state_dict(new_checkpoint['model_state_dict'])
	print("Loaded model for rank: {}".format(rank))

	dist.barrier()


	#######################################################
	# Get best model's predictions on train/val/test sets
	#######################################################
	# TODO: This takes a long time and should be distributed.
	if rank == 0:
		print("Rank 0 beginning prediction at time {}".format(datetime.now()))
		best_guess_model.eval()
		# best_guess_model.module.save_act = True 
		with torch.no_grad():
			# Get predictions for train examples, compute loss & plot
			best_guess_train_y_hat, best_guess_train_pred_para = best_guess_model(train_x.to(device), train_z.to(device), train_c.to(device),
																					whether_predict=0, PRODA_para=train_proda_para.to(device))
			train_smooth_l1_loss, train_NSE_SOC, train_NSE_POM, train_NSE_MAOM, train_l1_loss_SOC, train_l1_loss_POM, train_l1_loss_MAOM, train_param_reg_loss = fun_loss(best_guess_train_y_hat, train_y.to(device), best_guess_train_pred_para)
			print(f'Train - smooth_l1_loss: {train_smooth_l1_loss.item():.2f}, NSE_SOC: {train_NSE_SOC.item():.2f}, NSE_POM: {train_NSE_POM.item():.2f}, NSE_MAOM: {train_NSE_MAOM.item():.2f}, l1_loss_SOC: {train_l1_loss_SOC.item():.2f}, l1_loss_POM: {train_l1_loss_POM.item():.2f}, l1_loss_MAOM: {train_l1_loss_MAOM.item():.2f}')

			# Get predictions for val examples, compute loss & plot
			best_guess_val_y_hat, best_guess_val_pred_para = best_guess_model(val_x.to(device), val_z.to(device), val_c.to(device),
																				whether_predict=0, PRODA_para=val_proda_para.to(device))
			val_smooth_l1_loss, val_NSE_SOC, val_NSE_POM, val_NSE_MAOM, val_l1_loss_SOC, val_l1_loss_POM, val_l1_loss_MAOM, val_param_reg_loss = fun_loss(best_guess_val_y_hat, val_y.to(device), best_guess_val_pred_para)
			print(f'Val - smooth_l1_loss: {val_smooth_l1_loss.item():.2f}, NSE_SOC: {val_NSE_SOC.item():.2f}, NSE_POM: {val_NSE_POM.item():.2f}, NSE_MAOM: {val_NSE_MAOM.item():.2f}, l1_loss_SOC: {val_l1_loss_SOC.item():.2f}, l1_loss_POM: {val_l1_loss_POM.item():.2f}, l1_loss_MAOM: {val_l1_loss_MAOM.item():.2f}')

			if test_split_ratio != 0:
				# Get predictions for test examples, compute loss & plot
				best_guess_test_y_hat, best_guess_test_pred_para = best_guess_model(test_x.to(device), test_z.to(device), test_c.to(device),
																					whether_predict=0, PRODA_para=test_proda_para.to(device))
				test_smooth_l1_loss, test_NSE_SOC, test_NSE_POM, test_NSE_MAOM, test_l1_loss_SOC, test_l1_loss_POM, test_l1_loss_MAOM, test_param_reg_loss = fun_loss(best_guess_test_y_hat, test_y.to(device), best_guess_test_pred_para)
				print(f'Test - smooth_l1_loss: {test_smooth_l1_loss.item():.2f}, NSE_SOC: {test_NSE_SOC.item():.2f}, NSE_POM: {test_NSE_POM.item():.2f}, NSE_MAOM: {test_NSE_MAOM.item():.2f}, l1_loss_SOC: {test_l1_loss_SOC.item():.2f}, l1_loss_POM: {test_l1_loss_POM.item():.2f}, l1_loss_MAOM: {test_l1_loss_MAOM.item():.2f}')
				

			# Also generate PREDICTED SOC for EACH SOIL LAYER (20)
			train_simu_all_layers, _ = best_guess_model(train_x.to(device), train_z.to(device), train_c.to(device), whether_predict=1, PRODA_para=train_proda_para.to(device))
			val_simu_all_layers, _ = best_guess_model(val_x.to(device), val_z.to(device), val_c.to(device), whether_predict=1, PRODA_para=val_proda_para.to(device))
			test_simu_all_layers, _ = best_guess_model(test_x.to(device), test_z.to(device), test_c.to(device), whether_predict=1, PRODA_para=test_proda_para.to(device))

		# Summary csv file of all results. Create this if it doesn't exist
		results_summary_file = os.path.join(data_dir_output, f"neural_network/results_summary_{args.note}.csv")
		if not os.path.isfile(results_summary_file):
			with open(results_summary_file, mode='w') as f:
				csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
				csv_writer.writerow(['job_id', 'command', 'data_string', 'lr', 'weight_decay', 'seed', 'model_path', 'loss_lambda', 'val_NSE_soc', 'val_NSE_POM', 'val_NSE_MAOM', 'test_NSE_soc', 'test_NSE_POM', 'test_NSE_MAOM'])
		command_string = " ".join(sys.argv)
		data_string = f"Fold {args.cross_val_idx} {args.split} (data_seed = {args.data_seed}, n_datapoints = {args.n_datapoints})"

		# Add a row to the summary csv file
		with open(results_summary_file, mode='a+') as f:
			csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			best_model_path = data_dir_output + 'neural_network/' + job_id + '/opt_nn_' + job_id  + '.pt'
			csv_writer.writerow([job_id, command_string, data_string, args.lr, args.weight_decay, args.seed, best_model_path, args.lambdas, 
								val_NSE_SOC.item(), val_NSE_POM.item(), val_NSE_MAOM.item(), 
								test_NSE_SOC.item(), test_NSE_POM.item(), test_NSE_MAOM.item()])

		# Save the predictions
		# create folder for the results
		os.makedirs(data_dir_output + 'neural_network/' + job_id + '/Validation', exist_ok=True)
		os.makedirs(data_dir_output + 'neural_network/' + job_id + '/Train', exist_ok=True)
		if test_split_ratio != 0:
			os.makedirs(data_dir_output + 'neural_network/' + job_id + '/Test', exist_ok=True)

		#############
		# Test Data #
		#############

		## predictions and parameters for the test profiles ##

		best_simu_soc = torch.tensor(np.ones((combined_profile_num, 200))*np.nan, dtype = torch.float32, device=device)
		best_simu_POM = torch.tensor(np.ones((combined_profile_num, 200))*np.nan, dtype = torch.float32, device=device)
		best_simu_MAOM = torch.tensor(np.ones((combined_profile_num, 200))*np.nan, dtype = torch.float32, device=device)
		best_pred_para = torch.tensor(np.ones((combined_profile_num, len(para_names)))*np.nan, dtype = torch.float32, device=device)
		upper_depth_all = np.ones((combined_profile_num, 200))*np.nan
		lower_depth_all = np.ones((combined_profile_num, 200))*np.nan

		
		upper_depth_all[current_data_profile_id, :] = obs_upper_depth_matrix
		lower_depth_all[current_data_profile_id, :] = obs_lower_depth_matrix

		best_simu_soc[test_profile_id, :] = best_guess_test_y_hat[:, 0, :]
		best_simu_POM[test_profile_id, :] = best_guess_test_y_hat[:, 1, :]
		best_simu_MAOM[test_profile_id, :] = best_guess_test_y_hat[:, 2, :]
		best_pred_para[test_profile_id, :] = best_guess_test_pred_para


		# save data
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_best_simu_soc_' + job_id + '.csv', best_simu_soc.detach().cpu().numpy(), delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_best_simu_POM_' + job_id + '.csv', best_simu_POM.detach().cpu().numpy(), delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_best_simu_MAOM_' + job_id + '.csv', best_simu_MAOM.detach().cpu().numpy(), delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_best_pred_para_' + job_id + '.csv', best_pred_para.detach().cpu().numpy(), delimiter = ',')

		# Save pool data
		best_simu_soc_pool = torch.tensor(np.ones((combined_profile_num, 200))*np.nan, dtype = torch.float32, device=device)
		best_simu_POM_pool = torch.tensor(np.ones((combined_profile_num, 200))*np.nan, dtype = torch.float32, device=device)
		best_simu_MAOM_pool = torch.tensor(np.ones((combined_profile_num, 200))*np.nan, dtype = torch.float32, device=device)
		best_simu_DOC_pool = torch.tensor(np.ones((combined_profile_num, 200))*np.nan, dtype = torch.float32, device=device)
		best_simu_MIC_pool = torch.tensor(np.ones((combined_profile_num, 200))*np.nan, dtype = torch.float32, device=device)

		best_simu_soc_pool[test_profile_id, :] = test_simu_all_layers[:, 0, :]
		best_simu_POM_pool[test_profile_id, :] = test_simu_all_layers[:, 1, :]
		best_simu_MAOM_pool[test_profile_id, :] = test_simu_all_layers[:, 2, :]
		best_simu_DOC_pool[test_profile_id, :] = test_simu_all_layers[:, 3, :]
		best_simu_MIC_pool[test_profile_id, :] = test_simu_all_layers[:, 4, :]

		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_best_simu_soc_pool_' + job_id + '.csv', best_simu_soc_pool.detach().cpu().numpy(), delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_best_simu_POM_pool_' + job_id + '.csv', best_simu_POM_pool.detach().cpu().numpy(), delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_best_simu_MAOM_pool_' + job_id + '.csv', best_simu_MAOM_pool.detach().cpu().numpy(), delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_best_simu_DOC_pool_' + job_id + '.csv', best_simu_DOC_pool.detach().cpu().numpy(), delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_best_simu_MIC_pool_' + job_id + '.csv', best_simu_MIC_pool.detach().cpu().numpy(), delimiter = ',')


		# best_simu_sMAOM_pool = torch.tensor(np.ones((combined_profile_num, 200))*np.nan, dtype = torch.float32, device=device)
		# best_simu_sMAOM_pool[test_profile_id, :] = best_guess_test_y_pool[:, 5, :]
		# np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_best_simu_sMAOM_pool_' + job_id + '.csv', best_simu_sMAOM_pool.detach().cpu().numpy(), delimiter = ',')

		############
		# Val Data #
		############

		## predictions and parameters for the validation profiles ##

		# initializz a seperate array to store the prediction results for the validation profiles
		val_simu_soc = torch.tensor(np.ones((combined_profile_num, 200))*np.nan, dtype = torch.float32, device=device)
		val_simu_POM = torch.tensor(np.ones((combined_profile_num, 200))*np.nan, dtype = torch.float32, device=device)
		val_simu_MAOM = torch.tensor(np.ones((combined_profile_num, 200))*np.nan, dtype = torch.float32, device=device)
		val_pred_para = torch.tensor(np.ones((combined_profile_num, len(para_names)))*np.nan, dtype = torch.float32, device=device)

		val_simu_soc[val_profile_id, :] = best_guess_val_y_hat[:, 0, :]
		val_simu_POM[val_profile_id, :] = best_guess_val_y_hat[:, 1, :]
		val_simu_MAOM[val_profile_id, :] = best_guess_val_y_hat[:, 2, :]
		val_pred_para[val_profile_id, :] = best_guess_val_pred_para

		# save data
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_best_simu_soc_' + job_id + '.csv', val_simu_soc.detach().cpu().numpy(), delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_best_simu_POM_' + job_id + '.csv', val_simu_POM.detach().cpu().numpy(), delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_best_simu_MAOM_' + job_id + '.csv', val_simu_MAOM.detach().cpu().numpy(), delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_best_pred_para_' + job_id + '.csv', val_pred_para.detach().cpu().numpy(), delimiter = ',')

		# save pool data
		val_simu_soc_pool = torch.tensor(np.ones((combined_profile_num, 200))*np.nan, dtype = torch.float32, device=device)
		val_simu_POM_pool = torch.tensor(np.ones((combined_profile_num, 200))*np.nan, dtype = torch.float32, device=device)
		val_simu_MAOM_pool = torch.tensor(np.ones((combined_profile_num, 200))*np.nan, dtype = torch.float32, device=device)
		val_simu_DOC_pool = torch.tensor(np.ones((combined_profile_num, 200))*np.nan, dtype = torch.float32, device=device)
		val_simu_MIC_pool = torch.tensor(np.ones((combined_profile_num, 200))*np.nan, dtype = torch.float32, device=device)


		val_simu_soc_pool[val_profile_id, :] = val_simu_all_layers[:, 0, :]
		val_simu_POM_pool[val_profile_id, :] = val_simu_all_layers[:, 1, :]
		val_simu_MAOM_pool[val_profile_id, :] = val_simu_all_layers[:, 2, :]
		val_simu_DOC_pool[val_profile_id, :] = val_simu_all_layers[:, 3, :]
		val_simu_MIC_pool[val_profile_id, :] = val_simu_all_layers[:, 4, :]

		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_best_simu_soc_pool_' + job_id + '.csv', val_simu_soc_pool.detach().cpu().numpy(), delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_best_simu_POM_pool_' + job_id + '.csv', val_simu_POM_pool.detach().cpu().numpy(), delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_best_simu_MAOM_pool_' + job_id + '.csv', val_simu_MAOM_pool.detach().cpu().numpy(), delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_best_simu_DOC_pool_' + job_id + '.csv', val_simu_DOC_pool.detach().cpu().numpy(), delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_best_simu_MIC_pool_' + job_id + '.csv', val_simu_MIC_pool.detach().cpu().numpy(), delimiter = ',')


		# val_simu_sMAOM_pool = torch.tensor(np.ones((combined_profile_num, 200))*np.nan, dtype = torch.float32, device=device)
		# val_simu_sMAOM_pool[val_profile_id, :] = best_guess_val_y_pool[:, 5, :]
		# np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_best_simu_sMAOM_pool_' + job_id + '.csv', val_simu_sMAOM_pool.detach().cpu().numpy(), delimiter = ',')

		##############
		# Train Data #
		##############

		## predictions and parameters for the training profiles ##

		# initializz a seperate array to store the prediction results for the training profiles
		train_simu_soc = torch.tensor(np.ones((combined_profile_num, 200))*np.nan, dtype = torch.float32, device=device)
		train_simu_POM = torch.tensor(np.ones((combined_profile_num, 200))*np.nan, dtype = torch.float32, device=device)
		train_simu_MAOM = torch.tensor(np.ones((combined_profile_num, 200))*np.nan, dtype = torch.float32, device=device)
		train_pred_para = torch.tensor(np.ones((combined_profile_num, len(para_names)))*np.nan, dtype = torch.float32, device=device)

		train_simu_soc[train_profile_id, :] = best_guess_train_y_hat[:, 0, :]
		train_simu_POM[train_profile_id, :] = best_guess_train_y_hat[:, 1, :]
		train_simu_MAOM[train_profile_id, :] = best_guess_train_y_hat[:, 2, :]
		train_pred_para[train_profile_id, :] = best_guess_train_pred_para

		# save data
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_best_simu_soc_' + job_id + '.csv', train_simu_soc.detach().cpu().numpy(), delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_best_simu_POM_' + job_id + '.csv', train_simu_POM.detach().cpu().numpy(), delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_best_simu_MAOM_' + job_id + '.csv', train_simu_MAOM.detach().cpu().numpy(), delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_best_pred_para_' + job_id + '.csv', train_pred_para.detach().cpu().numpy(), delimiter = ',')


		# save pool data
		train_simu_soc_pool = torch.tensor(np.ones((combined_profile_num, 200))*np.nan, dtype = torch.float32, device=device)
		train_simu_POM_pool = torch.tensor(np.ones((combined_profile_num, 200))*np.nan, dtype = torch.float32, device=device)
		train_simu_MAOM_pool = torch.tensor(np.ones((combined_profile_num, 200))*np.nan, dtype = torch.float32, device=device)
		train_simu_DOC_pool = torch.tensor(np.ones((combined_profile_num, 200))*np.nan, dtype = torch.float32, device=device)
		train_simu_MIC_pool = torch.tensor(np.ones((combined_profile_num, 200))*np.nan, dtype = torch.float32, device=device)

		train_simu_soc_pool[train_profile_id, :] = train_simu_all_layers[:, 0, :]
		train_simu_POM_pool[train_profile_id, :] = train_simu_all_layers[:, 1, :]
		train_simu_MAOM_pool[train_profile_id, :] = train_simu_all_layers[:, 2, :]
		train_simu_DOC_pool[train_profile_id, :] = train_simu_all_layers[:, 3, :]
		train_simu_MIC_pool[train_profile_id, :] = train_simu_all_layers[:, 4, :]

		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_best_simu_soc_pool_' + job_id + '.csv', train_simu_soc_pool.detach().cpu().numpy(), delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_best_simu_POM_pool_' + job_id + '.csv', train_simu_POM_pool.detach().cpu().numpy(), delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_best_simu_MAOM_pool_' + job_id + '.csv', train_simu_MAOM_pool.detach().cpu().numpy(), delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_best_simu_DOC_pool_' + job_id + '.csv', train_simu_DOC_pool.detach().cpu().numpy(), delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_best_simu_MIC_pool_' + job_id + '.csv', train_simu_MIC_pool.detach().cpu().numpy(), delimiter = ',')


		# train_simu_sMAOM_pool = torch.tensor(np.ones((combined_profile_num, 200))*np.nan, dtype = torch.float32, device=device)
		# train_simu_sMAOM_pool[train_profile_id, :] = best_guess_train_y_pool[:, 5, :]
		# np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_best_simu_sMAOM_pool_' + job_id + '.csv', train_simu_sMAOM_pool.detach().cpu().numpy(), delimiter = ',')
		
		#############
		# Test Maps #
		#############

		# get the latitudes and longitudes of the test profiles by matching ProfileID in env_info with the test_profile_id
		test_lons = np.ones((combined_profile_num))*np.nan
		test_lats = np.ones((combined_profile_num))*np.nan
		test_profile_id_all = np.ones((combined_profile_num))*np.nan
		test_profile_id_all[test_profile_id] = test_profile_id
		test_profile_id_num = test_profile_id.numpy().astype(int)
		# check shape
		print("test_profile_id_num.shape: ", test_profile_id_num.shape)
		# print the range of test_profile_id_num
		print("test_profile_id_num.min(): ", test_profile_id_num.min())
		print("test_profile_id_num.max(): ", test_profile_id_num.max())
		print("env_info.shape: ", env_info.shape)
		test_lons[test_profile_id_num] = np.array(env_info.loc[test_profile_id_num, "original_lon"])
		test_lats[test_profile_id_num] = np.array(env_info.loc[test_profile_id_num, "original_lat"])
		print("Finished getting lat/lon data")
		# print the range of test_lons and test_lats
		print("test_lons.min(): ", test_lons.min())
		print("test_lons.max(): ", test_lons.max())
		print("test_lats.min(): ", test_lats.min())
		print("test_lats.max(): ", test_lats.max())

		# get the upper and lower depth of the test profiles
		test_upper_depth = np.ones((combined_profile_num, 200))*np.nan
		test_lower_depth = np.ones((combined_profile_num, 200))*np.nan
		# check shape
		print("test_upper_depth.shape: ", test_upper_depth.shape)
		print("obs_upper_depth_matrix.shape: ", upper_depth_all.shape)
		test_upper_depth[test_profile_id_num, :] = np.array(upper_depth_all[test_profile_id_num])
		test_lower_depth[test_profile_id_num, :] = np.array(lower_depth_all[test_profile_id_num])
		print("Finished getting depth data")

		# save location data for test profiles
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_profile_id_' + job_id + '.csv', test_profile_id_all, delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_lons_' + job_id + '.csv', test_lons, delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_lats_' + job_id + '.csv', test_lats, delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_upper_depth_' + job_id + '.csv', test_upper_depth, delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_lower_depth_' + job_id + '.csv', test_lower_depth, delimiter = ',')

		# initialize the scaled difference
		scaled_diff = np.ones((combined_profile_num))*np.nan

		# for each location, calculate the difference between the predicted and observed SOC values
		for i in range(binn_obs_soc.shape[0]):
			if np.isnan(binn_obs_soc[i, :]).all() or torch.isnan(best_simu_soc[i, :]).all():
				continue
			else: 
				# Get the predicted and observed SOC values for this profile
				obs_soc = binn_obs_soc[i, :]
				simu_soc = best_simu_soc[i, :]
				lower_depth = test_lower_depth[i]
				upper_depth = test_upper_depth[i]
				temp_simu_sum = 0
				temp_obs_sum = 0
				for j in range(len(simu_soc)):
					if np.isnan(obs_soc[j]) or torch.isnan(simu_soc[j]):
						continue
					else:
						if j >= 25:
							# print('outlier: ', test_profile_id_all[i], j, obs_soc[j], simu_soc[j])
							continue
						# Calculate the scaled difference
						temp_simu_sum += simu_soc[j] * (upper_depth[j] - lower_depth[j])
						temp_obs_sum += obs_soc[j] * (upper_depth[j] - lower_depth[j])
				scaled_diff[i] = temp_obs_sum/temp_simu_sum
				# # print outlier
				# if scaled_diff[i] > 2:
					# print('outlier: ', test_profile_id_all[i], scaled_diff[i])


		############
		# Val Maps #
		############

		
		# get the latitudes and longitudes of the validation profiles by matching ProfileID in env_info with the val_profile_id
		val_lons = np.ones((combined_profile_num))*np.nan
		val_lats = np.ones((combined_profile_num))*np.nan
		val_profile_id_all = np.ones((combined_profile_num))*np.nan
		val_profile_id_all[val_profile_id] = val_profile_id
		val_profile_id_num = val_profile_id.numpy().astype(int)
		val_lons[val_profile_id_num] = np.array(env_info.loc[val_profile_id_num, "original_lon"])
		val_lats[val_profile_id_num] = np.array(env_info.loc[val_profile_id_num, "original_lat"])
		# print the range of val_lons and val_lats
		print("val_lons.min(): ", val_lons.min())
		print("val_lons.max(): ", val_lons.max())
		print("val_lats.min(): ", val_lats.min())
		print("val_lats.max(): ", val_lats.max())

		# get the upper and lower depth of the validation profiles
		val_upper_depth = np.ones((combined_profile_num, 200))*np.nan
		val_lower_depth = np.ones((combined_profile_num, 200))*np.nan
		val_upper_depth[val_profile_id_num, :] = np.array(upper_depth_all[val_profile_id_num])
		val_lower_depth[val_profile_id_num, :] = np.array(lower_depth_all[val_profile_id_num])

		# save location data for validation profiles
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_profile_id_' + job_id + '.csv', val_profile_id_all, delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_lons_' + job_id + '.csv', val_lons, delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_lats_' + job_id + '.csv', val_lats, delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_upper_depth_' + job_id + '.csv', val_upper_depth, delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_lower_depth_' + job_id + '.csv', val_lower_depth, delimiter = ',')

		# Plot maps to show the scaled difference between the predicted and observed SOC values for validation profiles
		# convert nan to 0
		# binn_obs_soc[np.isnan(binn_obs_soc)] = 0
		# best_simu_soc[torch.isnan(best_simu_soc)] = 0
		# initialize the scaled difference
		scaled_diff = np.ones((combined_profile_num))*np.nan

		# for each location, calculate the difference between the predicted and observed SOC values
		for i in range(val_simu_soc.shape[0]):
			if np.isnan(binn_obs_soc[i, :]).all() or torch.isnan(val_simu_soc[i, :]).all():
				continue
			else: 
				# Get the predicted and observed SOC values for this profile
				obs_soc = binn_obs_soc[i, :]
				# print(obs_soc)
				# print(obs_soc.dtype)
				simu_soc = val_simu_soc[i, :]
				lower_depth = val_lower_depth[i]
				upper_depth = val_upper_depth[i]
				temp_simu = 0
				temp_obs_sum = 0
				for j in range(len(simu_soc)):
					if np.isnan(obs_soc[j]) or torch.isnan(simu_soc[j]):
						continue
					else:
						if j >= 25:
							# print('outlier: ', val_profile_id_all[i], j, obs_soc[j], simu_soc[j])
							continue
						# Calculate the scaled difference
						temp_simu += simu_soc[j] * (upper_depth[j] - lower_depth[j])
						temp_obs_sum += obs_soc[j] * (upper_depth[j] - lower_depth[j])
				scaled_diff[i] = temp_obs_sum/temp_simu
				# print outlier
				# if scaled_diff[i] > 2:
				# 	print('outlier: ', val_profile_id_all[i], scaled_diff[i])



		##############
		# Train Maps #
		##############

		# get the latitudes and longitudes of the training profiles by matching ProfileID in env_info with the train_profile_id
		train_lons = np.ones((combined_profile_num))*np.nan
		train_lats = np.ones((combined_profile_num))*np.nan
		train_profile_id_all = np.ones((combined_profile_num))*np.nan
		train_profile_id_all[train_profile_id] = train_profile_id
		train_profile_id_num = train_profile_id.numpy().astype(int)
		train_lons[train_profile_id_num] = np.array(env_info.loc[train_profile_id_num, "original_lon"])
		train_lats[train_profile_id_num] = np.array(env_info.loc[train_profile_id_num, "original_lat"])
		# print the range of lon and lat in the env_info
		print("env_info.lon.min(): ", env_info.loc[train_profile_id_num, "original_lon"].max())
		print("env_info.lon.max(): ", env_info.loc[train_profile_id_num, "original_lon"].min())
		print("env_info.lat.min(): ", env_info.loc[train_profile_id_num, "original_lat"].max())
		print("env_info.lat.max(): ", env_info.loc[train_profile_id_num, "original_lat"].min())
		

		# get the upper and lower depth of the training profiles
		train_upper_depth = np.ones((combined_profile_num, 200))*np.nan
		train_lower_depth = np.ones((combined_profile_num, 200))*np.nan
		train_upper_depth[train_profile_id_num, :] = np.array(upper_depth_all[train_profile_id_num])
		train_lower_depth[train_profile_id_num, :] = np.array(lower_depth_all[train_profile_id_num])

		# save location data for training profiles
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_profile_id_' + job_id + '.csv', train_profile_id_all, delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_lons_' + job_id + '.csv', train_lons, delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_lats_' + job_id + '.csv', train_lats, delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_upper_depth_' + job_id + '.csv', train_upper_depth, delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_lower_depth_' + job_id + '.csv', train_lower_depth, delimiter = ',')


		# Plot maps to show the scaled difference between the predicted and observed SOC values for training profiles
		scaled_diff = np.ones((combined_profile_num))*np.nan

		# for each location, calculate the difference between the predicted and observed SOC values
		for i in range(train_simu_soc.shape[0]):
			if np.isnan(binn_obs_soc[i, :]).all() or torch.isnan(train_simu_soc[i, :]).all():
				continue
			else: 
				# Get the predicted and observed SOC values for this profile
				obs_soc = binn_obs_soc[i, :]
				simu_soc = train_simu_soc[i, :]
				lower_depth = train_lower_depth[i]
				upper_depth = train_upper_depth[i]
				temp_simu = 0
				temp_obs_sum = 0
				for j in range(len(simu_soc)):
					if np.isnan(obs_soc[j]) or torch.isnan(simu_soc[j]):
						continue
					else:
						if j >= 25:
							# print('outlier: ', train_profile_id_all[i], j, obs_soc[j], simu_soc[j])
							continue
						# Calculate the scaled difference
						temp_simu += simu_soc[j] * (upper_depth[j] - lower_depth[j])
						temp_obs_sum += obs_soc[j] * (upper_depth[j] - lower_depth[j])
				scaled_diff[i] = temp_obs_sum/temp_simu



		# create folder for the results
		os.makedirs(data_dir_output + 'neural_network/' + job_id + '/Validation', exist_ok=True)
		os.makedirs(data_dir_output + 'neural_network/' + job_id + '/Train', exist_ok=True)
		if test_split_ratio != 0:
			os.makedirs(data_dir_output + 'neural_network/' + job_id + '/Test', exist_ok=True)


		# Output the symbolic formula
		best_guess_model.module.mlp.get_act()
		lib = ['x', 'x^2', 'x^3', 'exp', 'log', 'sqrt', '0', '1/x', '1/x^2']
		best_guess_model.module.mlp.auto_symbolic(lib = lib,
													a_range = (-20, 20),
        											b_range = (-20, 20),
													weight_simple = 0, r2_threshold = 0, verbose = 2)
		formula, variables = best_guess_model.module.mlp.symbolic_formula()
		# Save the formula to a text file
		with open(os.path.join(PLOT_DIR, "formula.txt"), "w") as f:
			for line in formula:
				f.write(str(line) + "\n" + "\n")
		# Save the variables to a text file
		with open(os.path.join(PLOT_DIR, "variables.txt"), "w") as f:
			for line in variables:
				f.write(str(line) + "\n")

		print("----------------- Finished train/val/test prediction " + str(datetime.now()) + "-----------------")
	
	# end if rank == 0:
	else:
		if whether_break.item() == 1:
			print("Rank {} finished".format(rank))
			dist.destroy_process_group()
			return

		
	# Pause to allow rank 0 to finish writing the summary file
	dist.barrier()
	print("Rank {} finished".format(rank))
	dist.destroy_process_group()





if __name__ == '__main__':

	# Number of CPUs requester
	world_size = args.num_CPU
	processes = []
	port='12355'

	# Create job ID
	job_id = create_output_folders(args)
	print("MAIN, JOB ID", job_id)
	print("Command:", " ".join(sys.argv))

	for rank in range(world_size):
		p = Process(target=worker, args=(rank, world_size, job_id, port))
		p.start()
		processes.append(p)

	for p in processes:
		p.join()

	
