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
import torch
from collections import OrderedDict
import misc_utils
from sklearn.model_selection import KFold
from mlp import ConstantParameters
from torch.optim.swa_utils import AveragedModel, SWALR
from losses import binns_loss, compute_param_matching_loss, compute_param_violation_loss, compute_unconstrained_param_loss
import visualization_utils
import glob

# Set HDF5_DISABLE_VERSION_CHECK to suppress version mismatch error
import os
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2'

from datetime import datetime, timedelta
import pandas as pd
from pandas import DataFrame as df
import numpy as np
from scipy.interpolate import pchip_interpolate

print("Start binns_DDP")

# NOTE: previously we set default dtype to float64 to avoid underflow in process-based model.
# Now checking float32 with fixed process-based model.
torch.set_default_dtype(torch.float32)

# Temporary hack to avoid printing np.float64(...) when printing out numpy scalars.
# TODO fix this
np.set_printoptions(legacy="1.21")

import os
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

###################################
# Import CLM5 process-based model #
###################################
# fun_model_simu predicts at user-specified depths. fun_model_prediction predicts at 20 default layers.
from fun_matrix_clm5_experimental import fun_model_simu, fun_model_prediction

# fun_bulk_simu returns additional components (quantities describing physical processes)
from fun_matrix_clm5_vectorized_bulk_converge import fun_bulk_simu

################################################
# Command-line arguments
################################################
parser = argparse.ArgumentParser()

# Model architecture
parser.add_argument("--note", type=str, default="", help="Optional name to give to the model")
parser.add_argument("--model", type=str, default="new_mlp", choices=['new_mlp', 'kan', 'nn_only'], help="Model type")
parser.add_argument("--width", type=int, default=128, help="Size of hidden layers (new_mlp or nn_only)")
parser.add_argument("--num_layers", type=int, default=3, help="Size of hidden layers (new_mlp or nn_only)")
parser.add_argument("--residual", action='store_true', help="Whether to add residual connections in neural network portion (MLP)")
parser.add_argument("--categorical", type=str, default="embedding", choices=["embedding", "one_hot"], help="How to embed categorical variables")
parser.add_argument("--embed_dim", type=int, default=5, help="Embedding dim for each categorical variable (if using embeddings)")
parser.add_argument("--use_bn", action='store_true', help="Whether to use batchnorm")
parser.add_argument("--dropout_prob", default=0., type=float, help="Dropout prob")
parser.add_argument("--activation", type=str, choices=['relu', 'leaky_relu', 'tanh'], default='relu', help="Activation function inside neural network")
parser.add_argument("--param_constraint", type=str, choices=['sigmoid', 'hardsigmoid', 'none'], default='sigmoid', help="Activation function used to constrain parameter predictions. If sigmoid, we suggest using param_reg loss. If hardsigmoid, use param_violation loss")

# KAN specific
parser.add_argument("--kan_grid", type=int, default=3, help="Number of grid intervals in KAN")
parser.add_argument("--kan_update_grid", type=int, default=1, help="Whether to update grids for KAN every epoch (default true)")
parser.add_argument("--kan_grid_margin", type=float, default=1.0, help="How much margin to use (in units of input range) when creating grids for KAN. Only used if kan_update_grid is 1.")
parser.add_argument("--kan_noise", type=float, default=0.3, help="Noise scale for KAN")
parser.add_argument("--kan_base_fun", type=str, default="silu", choices=["silu", "identity", "silu_identity", "zero"], help="Base function for KAN")
parser.add_argument("--kan_affine_trainable", action='store_true')
parser.add_argument("--kan_absolute_deviation", action='store_true')
parser.add_argument("--kan_flat_entropy", type=int, default=1)

# DropKAN
parser.add_argument("--kan_drop_rate", type=float, default=0.0, help="Drop rate for DropKAN")
parser.add_argument("--kan_drop_mode", type=str, choices=['postspline', 'postact', 'postact_input', 'dropout'], default='postact', help="Drop mode: 'postspline' the drop mask is applied to the layer's postsplines, 'postact' the drop mask is applied to the layer's postacts, 'dropout' applies a standard dropout layer to the inputs.")
parser.add_argument("--kan_drop_scale", action='store_true', help='If true, the retained postsplines/postacts are scaled by a factor of 1/(1-drop_rate)')

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
parser.add_argument("--init", type=str, default="default", choices=["default", "xavier_uniform", "kaiming_uniform"], help="Initialization for weights. For xavier_uniform/kaiming_uniform, biases are initialized to zero.")

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
parser.add_argument("--labels", type=str, default='real', choices=['real', 'synthetic_proda', 'synthetic_function'], help="Whether to use real labels, synthetic SOC labels generated from PRODA parameters, or synthetic SOC labels generated from synthetic parameters, which are generated using prescribed functional relationships.")
parser.add_argument("--label_noise_std", type=float, default=0., help="epsilon, where we multiply SOC labels by N(1, epsilon). Only used if args.labels is synthetic_proda or synthetic_function.")

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
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_epochs", type=int, default=200)
parser.add_argument("--bias_only_epochs", type=int, default=0, help="Number of epochs where we train ONLY FINAL-LAYER BIAS. This helps initialize params to a good value globally.")
parser.add_argument("--patience", type=int, default=100)
parser.add_argument("--save_freq", type=int, default=5, help="How often (epochs) to save the latest checkpoint, in case the job crashes")

# Regularization
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--use_swa", action='store_true', help="Whether to use Stochastic Weight Averaging")
parser.add_argument("--clip_value", type=float, default=-1, help="Clip value for gradient clipping. -1 for no clipping.")

# Losses and loss weights
# TO DELETE: "spectral", "lipmlp", "cure", "senn_robustness", "senn_l1", "senn_sparsity", "nam_l2", "nam_entropy", "spatial_error", "spatial_emb_smoothness", "param_smoothness", "residual"
parser.add_argument("--losses", nargs="+", choices=["l1", "smooth_l1", "l2", "param_reg", "param_violation", "unconstrained_param", "param_matching", "jacobian",
													"jacobian_sparsity", "kan_l1", "kan_entropy", "kan_coef", "kan_coefdiff", "kan_coefdiff2", "kan_diversity"], default=["smooth_l1", "param_reg"],
					help="Losses to use (can list any number). Note jacobian_sparsity cannot be optimized (non-differentiable): it is just something we track.")
parser.add_argument("--loss_weighting", default="manual", choices=["manual", "relobralo", "two_stage"])
parser.add_argument("--lambdas", nargs="+", type=float, default=[1.0, 10.0], help="If loss_weighting is manual, provide weights in the same order as `args.losses`")
parser.add_argument("--second_start", type=int, default=30, help="If loss_weighting is two_stage, epoch the second phase starts")
parser.add_argument("--second_lambdas", nargs="+", type=float, default=[1.0, 10.0], help="If loss_weighting is two_stage, weights for the second stage - in the same order as `args.losses`")

# Relobralo specific hyperparams (specific method of loss balancing: only used if you set `--loss_weighting relobralo`)
parser.add_argument("--relobralo_alpha", type=float, default=0.9, help="Exponential decay rate for Relobralo")
parser.add_argument("--relobralo_temp", type=float, default=0.1, help="Softmax temperature for Relobralo")
parser.add_argument("--relobralo_saudade", type=float, default=0.999, help="Saudade (1 minus probability of looking back to epoch 0)")

# Positional encoding
parser.add_argument("--features", type=str, choices=["all", "all_including_lonlat", "ten", "eight"], default="all", help="Which features to use. Default `all` includes all features except lon/lat. To include lon/lat explicitly, use `all_including_lonlat`. `ten` is 10 handcrafted features. `eight` is same as ten but excluding vegetation (biological) features.")
parser.add_argument("--pos_enc", type=str, default='none', choices=['none', 'early', 'late'],
					help="How lon/lat features are encoded. 'none' means not used. 'early' means that positional encoding is concatenated with other features. 'late' means that it is only used as an error term for the latent parameters.")

# Computational environment
parser.add_argument("--use_ddp", type=int, default=1, help="Whether to use DDP")
parser.add_argument("--num_CPU", type=int, default=32, help="Number of processes for Torch DDP")
parser.add_argument("--job_scheduler", type=str, default="pbs", choices=["pbs", "slurm"], help="Job scheduler system. Job scheduler. PBS for NCAR computers, slurm for AIDA server.")
parser.add_argument("--whether_resume", type=int, default=0, help="Whether to resume training from a previous model")
parser.add_argument("--previous_job_id", type=str, default="", help="Previous job id to resume from (otherwise, resumes using envir variable PREVIOUS_JOB_ID)")
parser.add_argument("--time_limit", type=float, default=11.5, help="Time limit for this job in HOURS. If trainng is not finished yet, start another job to continue.")
parser.add_argument("--plot", action='store_true', help="Whether to produce plots.")

args = parser.parse_args()

# Assert the number of lambdas provided equals the number of losses
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
		torch.cuda.manual_seed_all(args.seed)
	torch.backends.cudnn.deterministic = True

	# Below lines help reproducibility but may harm performance
	# https://docs.pytorch.org/docs/stable/notes/randomness.html
	os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Needed to run lstsq (in pykan/spline.py) deterministically
	torch.backends.cudnn.benchmark = False
	torch.use_deterministic_algorithms(True)


# Set seeds for reproducibility
set_seeds(args.seed)
if args.data_seed == -1:
	args.data_seed = args.seed

# Keep track of when the job started
job_begin_time = time.time()


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
# job_submit_path = '/glade/u/home/haodixu/BINN/PBS_Submit/Bulk_Converge/'
# data_dir_input = '/glade/u/home/haodixu/BINN/ENSEMBLE/INPUT_DATA/'
# data_dir_output = '/glade/work/haodixu/BINN/BINNS/OUTPUT_DATA/'
data_dir_input = '../ENSEMBLE/INPUT_DATA/'
data_dir_output = '../OUTPUT_DATA/'
job_submit_path = './resume_jobs/'
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

#--------------------------------
# Load PRODA Predicted Parameters
#--------------------------------
for i in range(1, 10):
	# Loop over all runs (folds) of PRODA. Construct a dataframe where each row
	# is a site. The first column is the profile ID. The next 21 columns are
	# parameters from the first run, next 21 are parameters from the second one, etc.

	# contains one column of profile id
	nn_site_loc_temp = pd.read_csv(data_dir_input + 'PRODA_Results/nn_site_loc_full_cesm2_clm5_cen_vr_v2_whole_time_exp_pc_cesm2_23_cross_valid_0_' + str(i) + '.csv', header=None)
	# contains the predicted parameters (21) for each profile
	nn_site_para_temp = pd.read_csv(data_dir_input + 'PRODA_Results/nn_para_result_full_cesm2_clm5_cen_vr_v2_whole_time_exp_pc_cesm2_23_cross_valid_0_' + str(i) + '.csv', header=None)

	if i == 1:
		# Initialize the dataframe with just profile_id
		PRODA_para = pd.DataFrame(nn_site_loc_temp)
		PRODA_para.columns = ['profile_id']

		# Concatenate parameters on the right
		PRODA_para = pd.concat([PRODA_para, nn_site_para_temp], axis = 1)
	else:
		# Concatenate this run's parameters on the right
		PRODA_para = pd.concat([PRODA_para, nn_site_para_temp], axis = 1)
# end
# For each parameter at each site, take the average across all runs
for i in range(1,22):
	PRODA_para['mean_' + str(i)] = PRODA_para.iloc[:, i:21*10:21].mean(axis = 1)
# end
# Drop the original columns
PRODA_para = PRODA_para.drop(PRODA_para.columns[1:21*9], axis = 1)

# Convert profile ID to zero-based, to match how WOSIS data is processed below
PRODA_para['profile_id'] = PRODA_para['profile_id'] - 1
print("PRODA parameters:")
print(PRODA_para.head())


#-------------------------------
# CLM5 constants
#-------------------------------
# Parameter names
if args.vertical_mixing == 'original':
	para_names = ['diffus', 'cryo', 'q10', 'efolding', 'taucwd', 'taul1', 'taul2', 'tau4s1', 'tau4s2', 'tau4s3', 'fl1s1', 'fl2s1', 'fl3s2', 'fs1s2', 'fs1s3', 'fs2s1', 'fs2s3', 'fs3s1', 'fcwdl2', 'w-scaling', 'beta']
else:
	# If using the simpler vertical mixing parameterization, replace diffus/cryo with slope/intercept.
	para_names = ['slope', 'intercept', 'q10', 'efolding', 'taucwd', 'taul1', 'taul2', 'tau4s1', 'tau4s2', 'tau4s3', 'fl1s1', 'fl2s1', 'fl3s2', 'fs1s2', 'fs1s3', 'fs2s1', 'fs2s3', 'fs3s1', 'fcwdl2', 'w-scaling', 'beta']
	if args.vertical_mixing == 'simple_two_intercepts':
		para_names.append('intercept_leach')

# Parameter indices that the neural network predicts. Usually we predict all the parameters, but for
# the retrieval test we may prescribe some and only predict 4 or 15 most sensitive parameters. 
if args.para_to_predict == "all":  # All parameters
	para_index = np.arange(0, len(para_names), dtype=int)
elif args.para_to_predict == "four":
	assert len(para_names) == 21
	para_index = np.array([3, 9, 14, 19], dtype=int)
elif args.para_to_predict == "fifteen":
	assert len(para_names) == 21
	para_index = np.array([0, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20], dtype=int)
else:
	raise ValueError("Invalid value of --para_to_predict")
predicted_para_names = np.array(para_names)[para_index].tolist()

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
sample_profile_id = loadmat(data_dir_input + 'wosis_2019_snap_shot/wosis_2019_snapshot_hugelius_mishra_representative_profiles.mat')

# NOTE: Not sure why "sample_profile_id" shape is [100, 50] before flattening?
sample_profile_id = sample_profile_id['sample_profile_id'].flatten()

# convert the number to be starting from 0 in python world
sample_profile_id = sample_profile_id - 1

# Choose the profile id with lat and lon within the range of the United States
profile_collection = np.where(
	(wosis_profile_info[:, 2] == 156) & 
	(wosis_profile_info[:, 3] >= -124.763068) & 
	(wosis_profile_info[:, 3] <= -66.949895) & 
	(wosis_profile_info[:, 4] >= 24.5) & 
	(wosis_profile_info[:, 4] <= 49.384358)
)[0]

##################################
# If using same dataset as PRODA #
##################################
# load mat file
para_gr = loadmat(data_dir_input + 'wosis_2019_snap_shot/cesm2_clm5_cen_vr_v2_para_gr.mat')
stat_r2 = loadmat(data_dir_input + 'wosis_2019_snap_shot/cesm2_clm5_cen_vr_v2_stat_r2.mat')
eligible_profile = loadmat(data_dir_input + 'wosis_2019_snap_shot/eligible_profile_loc_0_cesm2_clm5_cen_vr_v2_whole_time.mat')
para_gr = para_gr['para_gr']
stat_r2 = stat_r2['stat_r2']
eligible_profile = eligible_profile['eligible_loc_0']
# convert the number to be starting from 0 in python world
eligible_profile = eligible_profile - 1
# calculate average value per row in para_gr, and choose those profiles with average value less than 1.05
# calculate average value per row in stat_r2, and choose those profiles with average value larger than 0
# choose profile that listed in eligible_profile
PRODA_collection = np.where((np.mean(para_gr, axis = 1) < 1.05) & 
							(np.mean(stat_r2, axis = 1) > 0) & 
							(np.isin(np.arange(0, wosis_profile_info.shape[0]), eligible_profile) == True) & 
							# Also in the column profile_id of the dataframe PRODA_para
							(np.isin(np.arange(0, wosis_profile_info.shape[0]), PRODA_para['profile_id']) == True)
							)[0]
# Choose overlap between profile_collection and PRODA_collection
profile_collection = np.intersect1d(profile_collection, PRODA_collection)
print("Profile collection after intersect1d", profile_collection.shape)  # np.sort(profile_collection)[0:10])

if args.representative_sample:
	# Restrict to only "representative profiles" (1018)
	profile_collection = np.intersect1d(profile_collection, sample_profile_id)
	print("Profile collection after sample profile ID", profile_collection.shape)

if args.n_datapoints != -1:
	# Choose random subset of profiles for training.
	rng = np.random.default_rng(seed=args.data_seed)
	profile_collection = rng.choice(profile_collection, args.n_datapoints, replace=False)
	print("Profile collection after choice", profile_collection.shape)

profile_collection = np.reshape(profile_collection, [profile_collection.shape[0], 1])
profile_range = np.arange(0, len(profile_collection))
print('number of profiles: ', len(profile_collection))
print(datetime.now(), '------------all input data loaded------------')

#---------------------------------------------------
# wrap up soc data for NN
#---------------------------------------------------
obs_soc_matrix = np.ones([len(profile_collection), 200])*np.nan  # Each row is a profile. Each non-nan column is an SOC observation
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
'R_Squared']

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
	var4nn = ["BIO1", "BIO12", "Clay_Content_avg", "Sand_Content_avg", "Bulk_Density_avg", "SWC_v_Wilting_Point_avg", "pH_Water_avg", "CEC_avg", "cesm2_npp", "cesm2_vegc"]
elif args.features == "eight":
	var4nn = ["BIO1", "BIO12", "Clay_Content_avg", "Sand_Content_avg", "Bulk_Density_avg", "SWC_v_Wilting_Point_avg", "pH_Water_avg", "CEC_avg"]
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
env_info = loadmat(data_dir_input + 'wosis_2019_snap_shot/wosis_2019_snapshot_hugelius_mishra_env_info.mat')
env_info = env_info['EnvInfo']
original_lons = env_info[:, 3].copy()  # Save the original (unscaled) lon/lat
original_lats = env_info[:, 4].copy()
env_info = df(env_info)
env_info.columns = env_info_names

# Min/max for each feature
col_max_min = loadmat(data_dir_input + 'wosis_2019_snap_shot/world_grid_envinfo_present_cesm2_clm5_cen_vr_v2_whole_time_col_max_min.mat')
col_max_min = col_max_min['col_max_min']

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
		columns = env_info.filter(regex=(f"{var_prefix}*"))
		env_info[v] = columns.mean(axis=1)
		indices = [env_info_names.index(col) for col in columns.columns]
		new_max_min = np.mean(col_max_min[indices, :], axis=0, keepdims=True)  # keep shape [1, 2]
		all_col_max_mins.append(new_max_min)
col_max_min = np.concatenate(all_col_max_mins, axis=0)  # shape [num_columns_new, 2]

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
current_data_y = obs_soc_matrix

# Depth of each observation in current_data_y (same shape)
current_data_z = obs_depth_matrix

# Geographic coordinates
lons = np.array(env_info.loc[profile_collection[:, 0], "original_lon"])
lats = np.array(env_info.loc[profile_collection[:, 0], "original_lat"])
current_data_c = np.stack([lons, lats], axis=1)  # [profile, 2]: lon/lat of each site

# Remove sites with missing features or forcing variables
nan_loc = np.nanmean(current_data_y, axis = 1) + \
			np.sum(current_data_x[:, 0:len(var4nn), 0, 0], axis = 1) + \
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
valid_profile_loc = np.where(np.isnan(nan_loc) == False)[0] ### Why change the shape from 26915 to 26934??? ###  joshuafan: some sites may have missing forcing data or covariates

current_data_y = current_data_y[valid_profile_loc, :]
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


# Select PRODA parameters so that the Profile_IDs match the current data
PRODA_para = PRODA_para.loc[PRODA_para['profile_id'].isin(current_data_profile_id)]
PRODA_para = PRODA_para.sort_values(by='profile_id')

# Store the PRODA_para into numpy array (mean_1 to mean_21)
current_PRODA_para = PRODA_para[['mean_1', 'mean_2', 'mean_3', 'mean_4', 'mean_5', 'mean_6', 'mean_7', 'mean_8', 'mean_9', 'mean_10', 'mean_11', \
								 'mean_12', 'mean_13', 'mean_14', 'mean_15', 'mean_16', 'mean_17', 'mean_18', 'mean_19', 'mean_20', 'mean_21']].to_numpy()
print("Shape of PRODA para", current_PRODA_para.shape)

# Clamp to [0, 1]
current_PRODA_para = np.clip(current_PRODA_para, a_min=0, a_max=1)

#############################
# PRODA soc simulation data #
#############################
true_relationships = None

# Fetch directory where cached labels are saved
if args.labels != "real":
	# If using synthetic datasets, save the labels to a directory
	label_dir = os.path.join(data_dir_input, f"labels_{args.labels}_para={args.para_to_predict}_seed={args.seed}")
	if args.representative_sample:
		label_dir += "_representative"
	elif args.n_datapoints != -1:
		label_dir += "_datapoints={args.n_datapoints}"
	os.makedirs(label_dir, exist_ok=True)

if args.labels == "synthetic_proda":
	synthetic_label_path = os.path.join(label_dir, "synthetic_soc.npy")
	if os.path.exists(synthetic_label_path):  # If synthetic labels available, load them
		PRODA_soc_simu = np.load(synthetic_label_path)
		current_data_y = PRODA_soc_simu
	else:   # Otherwise compute synthetic labels from the PRODA parameters
		PRODA_soc_simu = np.ones((len(current_data_profile_id), 200))*np.nan

		start_time = time.time()
		for i in range(len(current_data_profile_id)):
			# Get the current profile's data
			current_data_x_simu = current_data_x[i, :, :, :]
			current_data_z_simu = current_data_z[i, :]
			current_PRODA_para_simu = current_PRODA_para[i, :]

			# Convert the data to tensor, reshape to shape [1, 60, 12, 13] and [1, 21]
			current_data_x_simu = torch.tensor(current_data_x_simu, dtype=torch.float32).unsqueeze(0)
			current_data_z_simu = torch.tensor(current_data_z_simu, dtype=torch.float32).unsqueeze(0)
			current_PRODA_para_simu = torch.tensor(current_PRODA_para_simu, dtype=torch.float32).unsqueeze(0)

			# Run the simulation
			PRODA_soc_simu[i, :] = fun_model_simu(current_PRODA_para_simu, current_data_x_simu, current_data_z_simu, args.vertical_mixing, args.vectorized)

			# If any simulation is over 1,000,000 gC/m2, set it to nan
			if np.any(PRODA_soc_simu[i, :] > 1000000):
				print(">>>>>>>>>>>>>>>>>>>>>>>> Extreme simulated SOC. Coordinates", current_data_c[i, :])
				print("PRODA params", current_PRODA_para[i, :])
				valid_loc = ~np.isnan(current_data_z[i, :])
				print("Depths", current_data_z[i, valid_loc])
				print("SOC simu", PRODA_soc_simu[i, valid_loc])
				print("SOC obs", current_data_y[i, valid_loc])

		# Drop the profiles with all nan values
		valid_profile_loc = np.where(np.all(np.isnan(PRODA_soc_simu), axis=1) == False)[0]
		current_data_y = current_data_y[valid_profile_loc, :]
		current_data_z = current_data_z[valid_profile_loc, :]
		current_data_c = current_data_c[valid_profile_loc, :]
		current_data_x = current_data_x[valid_profile_loc, :, :, :]
		current_data_profile_id = current_data_profile_id[valid_profile_loc]
		current_PRODA_para = current_PRODA_para[valid_profile_loc, :]
		PRODA_soc_simu = PRODA_soc_simu[valid_profile_loc, :]
		obs_upper_depth_matrix = obs_upper_depth_matrix[valid_profile_loc, :]
		obs_lower_depth_matrix = obs_lower_depth_matrix[valid_profile_loc, :]

		print("Time taken to run PRODA soc simu", time.time() - start_time)
		print("Shape of PRODA soc simu", PRODA_soc_simu.shape)
		print("Shape of current data x", current_data_x.shape)

		# If using synthetic labels, treat the simulated SOC as the true labels
		current_data_y = PRODA_soc_simu
		np.save(synthetic_label_path, PRODA_soc_simu)

elif args.labels == "synthetic_function":
	print("loading synthetic_function", label_dir, flush=True)

	synthetic_label_path = os.path.join(label_dir, "synthetic_soc.npy")
	if os.path.exists(synthetic_label_path) and False:
		PRODA_soc_simu = np.load(synthetic_label_path)
		current_data_y = PRODA_soc_simu
		current_PRODA_para = np.load(os.path.join(label_dir, "synthetic_para.npy"))
		sym_mask = np.load(os.path.join(label_dir, "relationship_types.npy"))
		true_relationships = np.load(os.path.join(label_dir, "true_relationships.npy"))
		# relationship_mask = torch.tensor(sym_mask != 0, dtype=int)  # 1 if relationship exists between input i and output j
	else:
		print("computing synthetic_function", label_dir, flush=True)

		# Construct a symbolic-only "KAN" that prescribes the true relationships between
		# input features and biogeochemical parameters. We mainly use the KAN infrastructure
		# for its plotting functionality.
		# Assume no categorical features for now.
		import kan
		true_kan = kan.KAN(width=[len(var4nn), len(para_index)], device="cpu",
					  	   input_size=len(var4nn), base_fun="identity")
		with torch.no_grad():

			# Set act_fun[0].mask to 0 to completely ignore the spline/learnable portion and only 
			# use the symbolic portion.
			true_kan.act_fun[0].mask = torch.zeros_like(true_kan.act_fun[0].mask)
			true_kan.save_acts = True

			# Make each parameter (column) depend on two inputs.
			sym_mask = np.zeros((len(var4nn), len(predicted_para_names)), dtype=int)
			rng = np.random.default_rng(seed=args.data_seed)
			for col_idx in range(len(predicted_para_names)):
				# Choose 2 random inputs to depend on
				row_indices = rng.permutation(len(var4nn))[:2]

				# For these 2 inputs, assign a random relationship type (between 1 and 5 inclusive)
				# 1=linear, 2=quadratic, 3=exp, 4=log, 5=relu.
				sym_mask[row_indices, col_idx] = rng.integers(low=1, high=6, size=row_indices.shape)
			relationship_mask = torch.tensor(sym_mask != 0, dtype=int)  # 1 if relationship exists between input i and output j

			# Explicitly defining functions since lambda functions cannot be pickled
			def zero_fn(x):
				return x*0
			def linear_fn(x):
				return x
			def quadratic_fn(x):
				return x**2
			def log_fn(x):
				return torch.log2(x + 2.5)
			def exp_fn(x):
				return 2**x
			def abs_fn(x):
				return torch.abs(x)
			functions = [zero_fn, linear_fn, quadratic_fn, log_fn, exp_fn, abs_fn]
			for i in range(sym_mask.shape[0]):
				for j in range(sym_mask.shape[1]):
					true_kan.fix_symbolic(0, i, j, fun_name=functions[sym_mask[i, j]], random=True, fit_params_bool=False, verbose=False)
			true_kan.symbolic_fun[0].mask = relationship_mask.T  # Transpose because Symbolic_KANLayer's mask is [out_dim, in_dim] 

			# Obtain parameters based on prescribed functional relationships.
			input_features = torch.tensor(current_data_x[:, 0:len(var4nn), 0, 0], dtype=torch.float32)  # This is the way to extract input features from current_data_x
			prescribed_para = true_kan(input_features)

			# Rescale so that each function's output is mean 0, std 1
			postacts = true_kan.spline_postacts[0]  # [batch, out_dim, in_dim]
			postacts_mean = postacts.mean(dim=0)
			postacts_std = postacts.std(dim=0)

			# rationale: if (cx+d) has mean mu, std sigma, then (cx+d - mu) / sigma has mean 0, std 1
			# thus c becomes c/sigma, and d becomes (d-mu)/sigma
			true_kan.symbolic_fun[0].affine[:, :, 2] = true_kan.symbolic_fun[0].affine[:, :, 2] / postacts_std
			true_kan.symbolic_fun[0].affine[:, :, 2] = torch.nan_to_num(true_kan.symbolic_fun[0].affine[:, :, 2], nan=1.0, posinf=1.0, neginf=1.0)
			true_kan.symbolic_fun[0].affine[:, :, 3] = (true_kan.symbolic_fun[0].affine[:, :, 3] - postacts_mean) / postacts_std
			true_kan.symbolic_fun[0].affine[:, :, 3] = torch.nan_to_num(true_kan.symbolic_fun[0].affine[:, :, 3], nan=0.0, posinf=0.0, neginf=0.0)

			# get the parameters based on rescale KAN
			prescribed_para = true_kan(input_features)

			# # verify that the postacts are mean 0, std 1
			# scaled_postacts = true_kan.spline_postacts[0]
			# print("Scaled postacts", scaled_postacts.mean(dim=0), scaled_postacts.std(dim=0))

			# Further scale/shift affine coefficients so all outputs are in range [0.2, 0.8].
			scale_by = 0.6 / (prescribed_para.max(dim=0).values - prescribed_para.min(dim=0).values)
			total_shift = 0.2 - 0.6 * prescribed_para.min(dim=0).values / (prescribed_para.max(dim=0).values - prescribed_para.min(dim=0).values)
			shift_by = total_shift / relationship_mask.sum(dim=0)
			constant_para = (relationship_mask.sum(dim=0) == 0)  # parameters with no functional relationships (constant). should not contain anything now.
			true_kan.symbolic_fun[0].affine[:, :, 2] = true_kan.symbolic_fun[0].affine[:, :, 2] * scale_by[:, None]
			true_kan.symbolic_fun[0].affine[constant_para, :, 2] = 1.0  # torch.nan_to_num(true_kan.symbolic_fun[0].affine[:, :, 2], nan=1.0, posinf=1.0, neginf=1.0)
			true_kan.symbolic_fun[0].affine[:, :, 3] = true_kan.symbolic_fun[0].affine[:, :, 3] * scale_by[:, None] + shift_by[:, None]
			true_kan.symbolic_fun[0].affine[constant_para, :, 3] = 0.0  # torch.nan_to_num(true_kan.symbolic_fun[0].affine[:, :, 3], nan=0.0, posinf=0.0, neginf=0.0)

			# Now get the prescribed scaled parameters
			prescribed_para = true_kan(input_features)

			# for parameters with no functional relationships, set them to 0.5
			prescribed_para[:, constant_para] = 0.5
			current_PRODA_para = torch.ones((len(current_data_profile_id), len(para_names))) * 0.5
			current_PRODA_para[:, para_index] = prescribed_para

			# Plot true functional relationships
			true_kan.attribute()
			true_kan.node_attribute()
			if not os.path.exists(os.path.join(label_dir, "TRUE_kan_plot.png")):
				true_kan.plot(folder=os.path.join(label_dir, "splines"), in_vars=var4nn, out_vars=predicted_para_names, scale=5, varscale=0.13)
				plt.savefig(os.path.join(label_dir, "TRUE_kan_plot.png"))
				plt.close()

			# Save picture of functional relationships. Source: https://stackoverflow.com/questions/69986007/matplotlib-imshow-with-1-color-for-each-discrete-value
			fig, ax = plt.subplots()
			cmap = plt.get_cmap('Set2', 6)
			im = ax.imshow(sym_mask, vmin=-0.5, vmax=5.5, cmap=cmap, interpolation="none")
			ax.set_xticks(np.arange(len(predicted_para_names)))
			ax.set_yticks(np.arange(len(var4nn)))
			ax.set_xticklabels(predicted_para_names, rotation='vertical')
			ax.set_yticklabels(var4nn)
			cbar = fig.colorbar(im, ticks=np.arange(0, 6), orientation="horizontal")
			cbar.ax.set_xticklabels(['None', 'Linear', 'Quadratic', 'Log', 'Exponential', 'Abs'])
			# fig.legend()
			plt.tight_layout()
			plt.savefig(os.path.join(label_dir, "functional_relationships.png"))
			plt.close()

			true_relationships = true_kan.edge_scores[0].permute(1, 0).detach().cpu().numpy()  # [n_input, n_output]
			true_relationships /= true_relationships.sum(axis=0, keepdims=True)

			# Use these synthetic parameters to generate SOC
			PRODA_soc_simu = np.ones((len(current_data_profile_id), 200))*np.nan
			start_time = time.time()
			for i in range(len(current_data_profile_id)):
				# Get the current profile's data
				current_data_x_simu = current_data_x[i, :, :, :]
				current_data_z_simu = current_data_z[i, :]
				current_PRODA_para_simu = current_PRODA_para[i, :]

				# Convert the data to tensor, reshape to shape [1, 60, 12, 13] and [1, 21]
				current_data_x_simu = torch.tensor(current_data_x_simu, dtype=torch.float32).unsqueeze(0)
				current_data_z_simu = torch.tensor(current_data_z_simu, dtype=torch.float32).unsqueeze(0)
				current_PRODA_para_simu = torch.tensor(current_PRODA_para_simu, dtype=torch.float32).unsqueeze(0)

				# Run the simulation
				PRODA_soc_simu[i, :] = fun_model_simu(current_PRODA_para_simu, current_data_x_simu, current_data_z_simu, args.vertical_mixing, args.vectorized)

				# If any simulation is over 1,000,000 gC/m2, set it to nan
				if np.any(PRODA_soc_simu[i, :] > 1000000):
					print(">>>>>>>>>>>>>>>>>>>>>>>> Extreme simulated SOC. Coordinates", current_data_c[i, :])
					print("PRODA params", current_PRODA_para[i, :])
					valid_loc = ~np.isnan(current_data_z[i, :])
					print("Depths", current_data_z[i, valid_loc])
					print("SOC simu", PRODA_soc_simu[i, valid_loc])
					print("SOC obs", current_data_y[i, valid_loc])

			# Drop the profiles with all nan values
			valid_profile_loc = np.where(np.all(np.isnan(PRODA_soc_simu), axis=1) == False)[0]
			current_data_y = current_data_y[valid_profile_loc, :]
			current_data_z = current_data_z[valid_profile_loc, :]
			current_data_c = current_data_c[valid_profile_loc, :]
			current_data_x = current_data_x[valid_profile_loc, :, :, :]
			current_data_profile_id = current_data_profile_id[valid_profile_loc]
			current_PRODA_para = current_PRODA_para[valid_profile_loc, :]
			PRODA_soc_simu = PRODA_soc_simu[valid_profile_loc, :]
			obs_upper_depth_matrix = obs_upper_depth_matrix[valid_profile_loc, :]
			obs_lower_depth_matrix = obs_lower_depth_matrix[valid_profile_loc, :]

			print("Time taken to run PRODA soc simu", time.time() - start_time)
			print("Shape of PRODA soc simu", PRODA_soc_simu.shape)
			print("Shape of current data x", current_data_x.shape, current_PRODA_para.shape)
			
			# Label noise
			if args.label_noise_std > 0:
				eps = torch.nn.init.trunc_normal_(torch.empty(PRODA_soc_simu.shape, requires_grad=False), mean=0, std=args.label_noise_std, a=-0.95, b=0.95)
				PRODA_soc_simu = PRODA_soc_simu * (1 + eps).detach().cpu().numpy()

			# Treat the simulated SOC as the true labels
			current_data_y = PRODA_soc_simu
			np.save(synthetic_label_path, PRODA_soc_simu)
			np.save(os.path.join(label_dir, "synthetic_para.npy"), current_PRODA_para)
			np.save(os.path.join(label_dir, "relationship_types.npy"), sym_mask)
			np.save(os.path.join(label_dir, "true_relationships.npy"), true_relationships)
			torch.save(true_kan, os.path.join(label_dir, "true_kan.pth"))
			print("Finished synthetic data generation")



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

# Load functional relationships inferred from previous runs so
# we can explore new relationships.
prev_relationships = []
prev_files = glob.glob(os.path.join(data_dir_output, "neural_network/*" + args.note + "*/visualizations/predicted_relationships.npy"))
print("PREV RELATIONSHIP FILES", prev_files)
for filename in prev_files:
	prev_relationships.append(np.load(filename))

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

train_proda_para = torch.tensor(current_PRODA_para[train_loc, :], dtype=torch.float32)
val_proda_para = torch.tensor(current_PRODA_para[val_loc, :], dtype=torch.float32)
test_proda_para = torch.tensor(current_PRODA_para[test_loc, :], dtype=torch.float32)

print("Shape of train data", train_x.shape)
print("Shape of val data", val_x.shape)
print("Shape of test data", test_x.shape)
print(datetime.now(), '------------nn data prepared------------')

#---------------------------------------------------
# Grid env info for prediction
#---------------------------------------------------
# load grid env info
grid_env_info = loadmat(data_dir_input + 'wosis_2019_snap_shot/world_grid_envinfo_present.mat')
grid_env_info = grid_env_info['EnvInfo']
original_lons_grid = grid_env_info[:, 0].copy()
original_lats_grid = grid_env_info[:, 1].copy()

# column names
# environmental info of global grids 
# Difference: does not include first 3 columns 'ProfileNum', 'ProfileID', 'LayerNum' and the last column 'R_Squared'
# Therefore, we choose to use the original categorical column names 
grid_env_info_names = [\
	'Lon', 'Lat', 'Date', \
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
	'nbedrock']

grid_env_info = df(grid_env_info)
grid_env_info.columns = grid_env_info_names

# Remove the first 3 columns and the R_squared column from the col_max_min matrix
col_max_min_grid = np.delete(col_max_min, env_info_names.index("R_Squared"), axis=0)[3:, :]

# Logic to add columns for "average" variables (e.g. average over layers)
for v in var4nn:
	if v not in env_info_names:
		var_prefix = v.split("_avg")[0]
		columns = grid_env_info.filter(regex=(f"{var_prefix}*"))
		grid_env_info[v] = columns.mean(axis=1)

# Normalize grid env info
for ivar in np.arange(0, len(col_max_min_grid[:, 0])):
	if np.isnan(col_max_min_grid[ivar, :]).any():
		pass
	else:
		grid_env_info.iloc[:, ivar] = (grid_env_info.iloc[:, ivar] - col_max_min_grid[ivar, 0])/(col_max_min_grid[ivar, 1] - col_max_min_grid[ivar, 0])
		grid_env_info.iloc[(grid_env_info.iloc[:, ivar] > 1), ivar] = 1
		grid_env_info.iloc[(grid_env_info.iloc[:, ivar] < 0), ivar] = 0


# Only keep the variables used in training the NN
grid_env_info = grid_env_info[var4nn]
grid_env_info["original_lon"] = original_lons_grid
grid_env_info["original_lat"] = original_lats_grid

# Select the rows with lon and lat values within continental US (and not nan)
grid_US_mask = (grid_env_info["original_lon"] >= -124.763068) \
			& (grid_env_info["original_lon"] <= -66.949895) \
			& (grid_env_info["original_lat"] >= 24.521694) \
			& (grid_env_info["original_lat"] <= 49.384358) \
			& (~grid_env_info.isnull().any(axis=1))  # True if grid cell is within US bounding box and has no nans
grid_US_profiles = np.where(grid_US_mask)[0]  # Indices (zero-based 'grid profile IDs') of grid cells in US, used later
grid_env_info_US = grid_env_info[grid_US_mask]
grid_env_info_num = grid_env_info_US.shape[0]

# Check the max value of categorical variables, if it is larger than the number of categories, then remove the row
for group in categorical_vars:
	if group[0] not in var4nn:
		continue
	mask = grid_env_info_US[group].apply(lambda x: (x > np.max(env_info[group])).any(), axis=1)
	indices_to_remove = grid_env_info_US[mask].index
	grid_env_info_US = grid_env_info_US.drop(indices_to_remove)
grid_env_info_num = grid_env_info_US.shape[0]

# Include forcing data for the grid env info
# Initialize the forcing data for the grid env info to nan and then fill in the values row by row
forcing_var = ['Input_CWD', 'Input_Litter1', 'Input_Litter2', 
			   'Input_Litter3', 'Altmax_Last_Year', 'Altmax_Current', 
			   'Nbedrock', 'Xio', 'Xin', 'Sand_Content', 'Soil_Temperature', 
			   'Soil_Water']

model_force_pred_input_vector_cwd = np.ones([grid_env_info_num, month_num])*np.nan
model_force_pred_input_vector_litter1 = np.ones([grid_env_info_num, month_num])*np.nan
model_force_pred_input_vector_litter2 = np.ones([grid_env_info_num, month_num])*np.nan
model_force_pred_input_vector_litter3 = np.ones([grid_env_info_num, month_num])*np.nan
model_force_pred_altmax_lastyear_profile = np.ones([grid_env_info_num, month_num])*np.nan
model_force_pred_altmax_current_profile = np.ones([grid_env_info_num, month_num])*np.nan
model_force_pred_nbedrock = np.ones([grid_env_info_num, month_num])*np.nan
model_force_pred_xio = np.ones([grid_env_info_num, soil_decom_num, month_num])*np.nan
model_force_pred_xin = np.ones([grid_env_info_num, soil_decom_num, month_num])*np.nan
model_force_pred_sand_vector = np.ones([grid_env_info_num, soil_decom_num, month_num])*np.nan
model_force_pred_soil_temp_profile = np.ones([grid_env_info_num, soil_decom_num, month_num])*np.nan
model_force_pred_soil_water_profile = np.ones([grid_env_info_num, soil_decom_num, month_num])*np.nan

# Fill in the forcing data
for irow in np.arange(0, grid_env_info_num):
	lat_loc = np.where(abs(grid_env_info_US.iloc[irow, :]["original_lat"] - lat_grid) == min(abs(grid_env_info_US.iloc[irow, :]["original_lat"] - lat_grid)))[0][0]
	lon_loc = np.where(abs(grid_env_info_US.iloc[irow, :]["original_lon"] - lon_grid) == min(abs(grid_env_info_US.iloc[irow, :]["original_lon"] - lon_grid)))[0][0]
	model_force_pred_input_vector_cwd[irow, :] = cesm2_simu_input_sum_cwd[lat_loc, lon_loc, :]
	model_force_pred_input_vector_litter1[irow, :] = cesm2_simu_input_sum_litter1[lat_loc, lon_loc, :]
	model_force_pred_input_vector_litter2[irow, :] = cesm2_simu_input_sum_litter2[lat_loc, lon_loc, :]
	model_force_pred_input_vector_litter3[irow, :] = cesm2_simu_input_sum_litter3[lat_loc, lon_loc, :]
	model_force_pred_altmax_lastyear_profile[irow, :] = cesm2_simu_altmax_last_year[lat_loc, lon_loc, :]
	model_force_pred_altmax_current_profile[irow, :] = cesm2_simu_altmax[lat_loc, lon_loc, :]
	model_force_pred_nbedrock[irow, :] = cesm2_simu_nbedrock[lat_loc, lon_loc, :]
	model_force_pred_xio[irow, :, :] = cesm2_simu_o_scalar[lat_loc, lon_loc, 0:soil_decom_num, :]
	model_force_pred_xin[irow, :, :] = cesm2_simu_n_scalar[lat_loc, lon_loc, 0:soil_decom_num, :]
	model_force_pred_sand_vector[irow, :, :] = cesm2_simu_cellsand[lat_loc, lon_loc, 0:soil_decom_num, :]
	model_force_pred_soil_temp_profile[irow, :, :] = cesm2_simu_soil_temperature[lat_loc, lon_loc, 0:soil_decom_num, :]
	model_force_pred_soil_water_profile[irow, :, :] = cesm2_simu_w_scalar[lat_loc, lon_loc, 0:soil_decom_num, :]
# end

# wrapping up for the nn prediction
predict_data_x = np.ones((grid_env_info_num, max(len(var4nn), 20), 12, 13))*np.nan
predict_data_x[:, 0:len(var4nn), 0, 0] = np.array(grid_env_info_US.loc[:, var4nn])
predict_data_x[:, 0:12, 0, 1] = model_force_pred_input_vector_cwd
predict_data_x[:, 0:12, 0, 2] = model_force_pred_input_vector_litter1
predict_data_x[:, 0:12, 0, 3] = model_force_pred_input_vector_litter2
predict_data_x[:, 0:12, 0, 4] = model_force_pred_input_vector_litter3
predict_data_x[:, 0:12, 0, 5] = model_force_pred_altmax_lastyear_profile
predict_data_x[:, 0:12, 0, 6] = model_force_pred_altmax_current_profile
predict_data_x[:, 0:12, 0, 7] = model_force_pred_nbedrock
predict_data_x[:, 0:20, 0:12, 8] = model_force_pred_xio
predict_data_x[:, 0:20, 0:12, 9] = model_force_pred_xin
predict_data_x[:, 0:20, 0:12, 10] = model_force_pred_sand_vector
predict_data_x[:, 0:20, 0:12, 11] = model_force_pred_soil_temp_profile
predict_data_x[:, 0:20, 0:12, 12] = model_force_pred_soil_water_profile

# create dummy z since it is not used in the prediction
predict_data_z = np.ones((grid_env_info_num))*np.nan
predict_data_c = np.stack([grid_env_info_US["original_lon"], grid_env_info_US["original_lat"]], axis=1)

print("Shape of predict data x", predict_data_x.shape)
print("Shape of predict data z", predict_data_z.shape)
print("Shape of grid env info US", grid_env_info_US.shape)
print(datetime.now(), '------------grid env info prepared------------')

#-----------------------------------------------
# Load PRODA Predicted Parameters for grid data
#-----------------------------------------------
for i in range(1, 10):
	# Loop over all runs (folds) of PRODA. Construct a dataframe where each row
	# is a site. The first column is the profile ID. The next 21 columns are
	# parameters from the first run, next 21 are parameters from the second one, etc.

	# contains one column of profile id
	valid_grid_loc = pd.read_csv(data_dir_input + 'PRODA_Results/valid_grid_loc_cesm2_clm5_cen_vr_v2_whole_time_exp_pc_cesm2_23_cross_valid_0_' + str(i) + '.csv', header=None)
	# contains the predicted parameters (21) for each profile
	grid_para = pd.read_csv(data_dir_input + 'PRODA_Results/grid_para_result_cesm2_clm5_cen_vr_v2_whole_time_exp_pc_cesm2_23_cross_valid_0_' + str(i) + '.csv', header=None)

	if i == 1:
		# Initialize the dataframe with just profile_id
		grid_PRODA_para = pd.DataFrame(valid_grid_loc)
		grid_PRODA_para.columns = ['profile_id']

		# Concatenate parameters on the right
		grid_PRODA_para = pd.concat([grid_PRODA_para, grid_para], axis = 1)
	else:
		# Concatenate this run's parameters on the right
		grid_PRODA_para = pd.concat([grid_PRODA_para, grid_para], axis = 1)
# end
# For each parameter at each site, take the average across all runs
for i in range(1,22):
	grid_PRODA_para['mean_' + str(i)] = grid_PRODA_para.iloc[:, i:21*10:21].mean(axis = 1)
# end
# Drop the original columns
grid_PRODA_para = grid_PRODA_para.drop(grid_PRODA_para.columns[1:21*9], axis = 1)

# Convert profile ID to zero-based, to match how WOSIS data is processed below
grid_PRODA_para['profile_id'] = grid_PRODA_para['profile_id'] - 1
grid_PRODA_para['profile_id'] = grid_PRODA_para['profile_id'].astype(int)

# First create an empty dataframe with the profile IDs in the same order as grid_env_info_US.
# Then, we attach the PRODA parameters. NOTE: not all profile IDs have PRODA parameters,
# so there may be nans.
grid_PRODA_para_aligned = pd.DataFrame({'profile_id': grid_US_profiles})
grid_PRODA_para_aligned = grid_PRODA_para_aligned.merge(grid_PRODA_para, how='left', on='profile_id')
grid_PRODA_para = grid_PRODA_para_aligned[['mean_1', 'mean_2', 'mean_3', 'mean_4', 'mean_5', 'mean_6', 'mean_7', 'mean_8', 'mean_9', 'mean_10', 'mean_11', \
											  'mean_12', 'mean_13', 'mean_14', 'mean_15', 'mean_16', 'mean_17', 'mean_18', 'mean_19', 'mean_20', 'mean_21']].to_numpy()
grid_PRODA_para = np.clip(grid_PRODA_para, a_min=0, a_max=1)

# Convert predict data to tensors
predict_data_x = torch.tensor(predict_data_x, dtype=torch.float32)
predict_data_z = torch.tensor(predict_data_z, dtype=torch.float32)
predict_data_c = torch.tensor(predict_data_c, dtype=torch.float32)
if args.labels == "synthetic_function":
	# Get the prescribed parameters
	grid_features = torch.tensor(predict_data_x[:, 0:len(var4nn), 0, 0], dtype=torch.float32)
	prescribed_para = true_kan(grid_features)
	prescribed_para[:, constant_para] = 0.5  # for parameters with no functional relationships, set them to 0.5
	grid_PRODA_para = torch.ones((predict_data_x.shape[0], len(para_names))) * 0.5
	grid_PRODA_para[:, para_index] = prescribed_para
else:
	grid_PRODA_para = torch.tensor(grid_PRODA_para, dtype=torch.float32)

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

	# NOTE: Create a "job id" using timestamp, note, and PBS jobid
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
	print("Number of CPUs: ", torch.cpu.device_count())  # No idea why it is not working on NCAR server
	print("Number of Cores: ", cpu_count)
	print("Number of threads: ", thread_count)
	if "CUDA_VISIBLE_DEVICES" in os.environ:
		print("CUDA_VISIBLE_DEVICES", os.environ["CUDA_VISIBLE_DEVICES"])

	# Set number of threads *per worker*. Should equal floor(CPUs/processes)
	# torch.set_num_threads(math.floor(torch.get_num_threads() / world_size))
	torch.set_num_threads(math.floor(cpu_count / world_size))  # TODO This line works on NCAR

	# Environment variables
	os.environ['RANK'] = str(rank)
	os.environ['WORLD_SIZE'] = str(world_size)
	os.environ["MASTER_ADDR"] = "localhost"
	os.environ["MASTER_PORT"] = port
	if torch.cuda.is_available():
		# Set device to the appropriate GPU
		gpu = int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[rank])
		device = torch.device(f"cuda:{gpu}")

		# Initialize process group (for GPU, use nccl backend)
		dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=timedelta(hours=2))
	else:
		# Set device to CPU
		device = torch.device("cpu")

		# Initialize process group (for CPU, use gloo backend)
		dist.init_process_group(backend="gloo", rank=rank, world_size=world_size, timeout=timedelta(hours=2))

	return device


# Start training
def worker(rank, world_size, job_id, port):

	# Filename to store loss records and visualizations
	nn_training_name = job_id + '_' + model_name
	LOSSES_FILENAME = 'avg_loss_' + nn_training_name + '.csv'
	METRICS_FILENAME = 'avg_metrics_' + nn_training_name + '.csv'
	PLOT_DIR = os.path.join(data_dir_output, 'neural_network', job_id, 'visualizations')
	os.makedirs(PLOT_DIR, exist_ok=True)  # Note: this should already exist from create_output_folders

	# Save printed output to file. Does not seem to work on Slurm.
	# sys.stdout = misc_utils.Logger(os.path.join(data_dir_output, "neural_network", job_id, "output.txt"))

	# Set up distributed environment
	if args.use_ddp == 1:
		device = ddp_setup(rank, world_size, port)
	else:
		device = "cuda:" + str(os.environ["CUDA_VISIBLE_DEVICES"].split(',')[0]) if torch.cuda.is_available() else "cpu"
	print(f"Finished DDP setup. Rank {rank} of {world_size}. Device {device}. JobID {job_id}.", flush=True)

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

	if args.bias_only_epochs >= 1 and (args.whether_resume == 0 or checkpoint_main["epoch"] < args.bias_only_epochs):
		# If training bias only, temporarily set the model to ConstantParameters
		# and then switch to the real model after that many epochs.
		# Exception: If we are resuming and the epoch is already past bias_only_epochs,
		# go to the else branch (don't create the ConstantParameters model, create the full model)
		model_class = ConstantParameters
		model_kwargs = {
			"num_params": len(predicted_para_names),
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
		val_loss_history = np.ones((num_epoch, len(args.losses)))*np.nan
		train_metrics_history = np.ones((num_epoch, 3))*np.nan  # Columns are [MSE, MAE, NSE]
		val_metrics_history = np.ones((num_epoch, 3))*np.nan
		lr_history = np.ones((num_epoch))*np.nan
		best_model_epoch = torch.tensor(0)  # epoch with the best model so far

		# Early stopping parameters
		best_val_loss = float('inf') 
		best_val_NSE = float('inf') 
		patience = args.patience
		epochs_without_improvement = 0

		# Save observations
		binn_obs_soc = np.ones((wosis_profile_info.shape[0], 200))*np.nan
		binn_obs_soc[current_data_profile_id, :] = current_data_y

		# Define starting epoch
		start_epoch = 0

		if rank == 1:
			np.savetxt(data_dir_output + 'neural_network/' + job_id + '/nn_obs_soc_' + job_id + '.csv', binn_obs_soc, delimiter = ',')
			# print the model structure
			print(model)

			# # try to save the predicted parameters before training.
		elif rank == 0:
			# model.eval()  # TODO Can't really use eval mode before model is trained, since batchnorm stats are not there yet
			print("Rank 0 About to val model before training", flush=True)
			with torch.no_grad():
				val_pred_soc = torch.tensor(np.ones((wosis_profile_info.shape[0], 200))*np.nan, device=device, dtype=torch.float32)
				val_pred_para = torch.tensor(np.ones((wosis_profile_info.shape[0], len(para_names)))*np.nan, device=device, dtype=torch.float32)

				# Since this is only run on one rank, it's important to use model.module, otherwise hangs may arise.
				# https://github.com/pytorch/pytorch/issues/54059
				temp_SOC, temp_pred_para = model.module(val_x.to(device), val_z.to(device), val_c.to(device), whether_predict=0, PRODA_para=val_proda_para.to(device))
				print("Rank 0 finished val model before training", flush=True)
				val_pred_soc[val_profile_id, :] = temp_SOC.detach()
				val_pred_para[val_profile_id, :] = temp_pred_para.detach()
				np.savetxt(data_dir_output + 'neural_network/' + job_id + '/model_training_history/nn_val_pred_soc_' + job_id + "_initial" + '.csv', val_pred_soc.detach().cpu().numpy(), delimiter = ',')
				np.savetxt(data_dir_output + 'neural_network/' + job_id + '/model_parameters/nn_val_pred_soc_' + job_id + "_initial" + '.csv', val_pred_para.detach().cpu().numpy(), delimiter = ',')
	else:
		# If resuming from a checkpoint, load loss history
		train_loss_history = checkpoint_worker['train_loss_history']
		val_loss_history = checkpoint_worker['val_loss_history']
		train_metrics_history = checkpoint_worker['train_metrics_history']
		val_metrics_history = checkpoint_worker['val_metrics_history']
		lr_history = checkpoint_worker['lr_history']
		best_model_epoch = checkpoint_worker['best_model_epoch']

		# Early stopping parameters
		best_val_loss = checkpoint_worker['best_val_loss']
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

		# Store predicted para/coords, and predicted/true SOC (for both train and val - for plotting)
		all_train_pred_para = []
		all_train_proda_para = []
		all_train_coords = []
		all_val_pred_para = []
		all_val_proda_para = []
		all_val_coords = []
		all_train_z = []
		all_train_pred_soc = []
		all_train_true_soc = []
		all_val_pred_soc = []
		all_val_true_soc = []
		all_val_z = []

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

		# If we have trained for "bias_only_epochs" epochs and it's
		# time to switch to training the full model, change the model to the full model
		if args.bias_only_epochs >= 1 and iepoch == args.bias_only_epochs:
			old_bias = model_without_ddp.best_params.data  # Get optimal param set from ConstantParameters model
			print("OLD BIAS", old_bias)
			model_class, model_kwargs = misc_utils.get_model(args, var4nn, var_idx_to_emb, device,
															para_index, train_x, train_y)
			model = model_class(**model_kwargs).to(device)
			model.mlp.layer_output.bias.data = old_bias

			# Create distributed version of the model
			if args.use_ddp == 1:
				if torch.cuda.is_available():
					model = DDP(model, device_ids=[device])
				else:
					model = DDP(model)
				model_without_ddp = model.module
			else:
				model_without_ddp = model
			optimizer, scheduler = misc_utils.get_optimizer_and_scheduler(model, args)

		# -------------------------------------training
		loss_record_train = list()  # List of Tensors. Each Tensor contains losses in the order of args.losses.
		metrics_record_train = list()  # List of Tensors (one per batch). Each Tensor contains 3 values: [MSE, MAE, NSE]
		ibatch = 0
		model.train()
		if args.use_ddp == 1:
			train_loader.sampler.set_epoch(iepoch)  # Set sampler's epoch number, so we use a different order per epoch

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
					model_without_ddp.update_grid(batch_x, batch_z, batch_c, whether_predict=0, PRODA_para=batch_proda_para)
					dist.barrier()

					# Synchronize the updated grid/coef parameters across all ranks
					for i in range(len(model_without_ddp.mlp.act_fun)):
						dist.all_reduce(model_without_ddp.mlp.act_fun[i].grid, op=dist.ReduceOp.SUM)  # ReduceOp.AVG is more concise but not supported with gloo backend
						model_without_ddp.mlp.act_fun[i].grid.data /= world_size
						dist.all_reduce(model_without_ddp.mlp.act_fun[i].coef, op=dist.ReduceOp.SUM)
						model_without_ddp.mlp.act_fun[i].coef.data /= world_size
						if model_without_ddp.mlp.base_fun == "silu_identity":
							dist.all_reduce(model_without_ddp.mlp.act_fun[i].silu_input_offset, op=dist.ReduceOp.SUM)
							model_without_ddp.mlp.act_fun[i].silu_input_offset.data /= world_size
					dist.barrier()  # MAKE SURE THIS DOES NOT CAUSE ISSUES. (Old run - this was every batch outside the if statement)


			#------------ 1 forward
			# train_nn_start = time.time()
			# Normal models return predicted (1) SOC, (2) parameters
			batch_y_hat, batch_pred_para = model(batch_x, batch_z, batch_c, whether_predict=0, PRODA_para=batch_proda_para)

			# Check if batch_pred_para is nan or inf
			if batch_y_hat is None or batch_pred_para is None:
				whether_break = torch.tensor(1).to(device)
			elif torch.isnan(batch_pred_para).any() or torch.isinf(batch_pred_para).any():
				whether_break = torch.tensor(1).to(device)
				for ipara in range(batch_pred_para.shape[0]):
					if torch.isnan(batch_pred_para[ipara]).any() or torch.isinf(batch_pred_para[ipara]).any():
						print(f"Epoch {iepoch} batch {ibatch} parameter {ipara} is {batch_pred_para[ipara]}")

			# Check range of parameter values
			if rank == 0 and ibatch == 1 and iepoch % 10 == 0:
				print("Predicted para", batch_pred_para)

			# If KAN, plot activation statistics
			if rank == 0 and args.model == "kan" and (ibatch == 1 and iepoch % 50 == 0):
				import pykan
				model_without_ddp.mlp.plot_activation_statistics(os.path.join(PLOT_DIR, f"epoch{iepoch}_KAN_activation_stats.png"))

			#------------ 2 compute the objective function
			l1_loss, smooth_l1_loss, l2_loss, param_reg_loss, train_NSE, corr = fun_loss(batch_y_hat, batch_y, batch_pred_para)

			# Compute additional losses if using. If we are not using them, set them to nan
			param_violation_loss = np.nan
			unconstrained_param_loss = np.nan
			param_matching_loss = np.nan
			jacobian_loss = np.nan
			jacobian_sparsity = np.nan
			kan_l1_loss = np.nan
			kan_entropy_loss = np.nan
			kan_coef_loss = np.nan
			kan_coefdiff_loss = np.nan
			kan_coefdiff2_loss = np.nan
			kan_diversity_loss = np.nan

			# Parameter losses
			if "param_violation" in args.losses:  # Penalty if unconstrained params are outside [-3, 3]. Only used with hardsigmoid.
				param_violation_loss = compute_param_violation_loss(model_without_ddp.unconstrained_params)
			if "unconstrained_param" in args.losses:
				unconstrained_param_loss = compute_unconstrained_param_loss(model_without_ddp.unconstrained_params)
			if "param_matching" in args.losses:  # Penalize extreme parameter values that are far from 0.5
				param_matching_loss = compute_param_matching_loss(batch_pred_para, batch_proda_para)

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
						visualization_utils.plot_matrix(jacobian[example_idx, :, :], row_labels=predicted_para_names, col_labels=var4nn,
														filename=os.path.join(PLOT_DIR, f"epoch{iepoch}_jacobian_{example_idx}.png"),
														title=f"Parameter-Feature Jacobian: Epoch {iepoch}, Example {example_idx}")

			if {"kan_l1", "kan_entropy", "kan_coef", "kan_coefdiff", "kan_coefdiff2"} & set(args.losses):
				assert args.model == "kan"

				# NOTE: the lamb values passed are completely unused, as we direclty obtain the individual loss components and weight them later.
				# For default weights see https://github.com/KindXiaoming/pykan/blob/master/kan/MultKAN.py#L1411
				kan_l1_loss, kan_entropy_loss, kan_coef_loss, kan_coefdiff_loss, kan_coefdiff2_loss, kan_conn_cost = model_without_ddp.mlp.reg(reg_metric='edge_backward', lamb_l1=1., lamb_entropy=1., lamb_coef=1., lamb_coefdiff=1.,
																																return_indiv=True, flat_entropy=args.kan_flat_entropy)
				# model_without_ddp.mlp.get_reg(reg_metric='node_influence_on_output', lamb_l1=0., lamb_entropy=1., lamb_coef=0., lamb_coefdiff=0.)

			if "kan_diversity" in args.losses:
				# TODO Only works with single layer KAN for now. Check plot_functional_relationships for additional.
				kan_importances = model_without_ddp.mlp.edge_scores[0].permute(1, 0)  # .detach().cpu().numpy()
				kan_importances = kan_importances / kan_importances.sum(dim=0, keepdim=True)
				kan_diversity_loss = torch.tensor(0., device=device)
				# print("CUR relationship", kan_importances)
				for prev_relationship in prev_relationships:
					# print("Prev relationships", prev_relationship)
					# print("KL", misc_utils.kl_divergence(torch.tensor(prev_relationship, device=device), kan_importances))
					kan_diversity_loss -= misc_utils.kl_divergence(torch.tensor(prev_relationship, device=device), kan_importances)

			#------------ 3 cleaning gradients
			optimizer.zero_grad()

			#------------ 4 accumulate partical derivatives of objective respect to parameters
			loss_dict = {"l1": l1_loss,
						"smooth_l1": smooth_l1_loss,
						"l2": l2_loss,
						"param_reg": param_reg_loss,
						"param_violation": param_violation_loss,
						"unconstrained_param": unconstrained_param_loss,
						"param_matching": param_matching_loss,
						"jacobian": jacobian_loss,
						"jacobian_sparsity": jacobian_sparsity,
						"kan_l1": kan_l1_loss,
						"kan_entropy": kan_entropy_loss,
						"kan_coef": kan_coef_loss,
						"kan_coefdiff": kan_coefdiff_loss,
						"kan_coefdiff2": kan_coefdiff2_loss,
						"kan_diversity": kan_diversity_loss}

			# Store losses in a tensor, in the order of args.losses
			train_losses = torch.stack([loss_dict[loss] for loss in args.losses]).to(device)

			# Compute weighted total loss, backpropagate
			total_loss = torch.dot(train_losses, args.lambdas.to(device))
			total_loss.backward()

			# clip gradients. TODO Not tested.
			if args.clip_value != -1:
				# torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)
				torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=args.clip_value)

			#------------ 5 step in the opposite direction of the gradient
			# with torch.no_grad(): para = para - eta*para.grad # eta is learning rate
			optimizer.step()

			# Noise
			if args.noise_std > 0:
				misc_utils.inject_noise(model, args.noise_std)

			# Record losses
			loss_record_train.append(train_losses)
			metrics_record_train.append(torch.tensor([l2_loss.item(), l1_loss.item(), train_NSE.item()], device=device))

			# Record predicted parameters, true/predicted SOC
			all_train_pred_para.append(batch_pred_para)
			all_train_proda_para.append(batch_proda_para)
			all_train_coords.append(batch_c)
			all_train_z.append(batch_z)
			all_train_pred_soc.append(batch_y_hat)
			all_train_true_soc.append(batch_y)

			# flush all printed output
			sys.stdout.flush()
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
		loss_record_val = list()  # List of Tensors (one per batch). Each Tensor contains losses in the order of args.losses.
		metrics_record_val = list()  # List of Tensors (one per batch). Each Tensor contains [MSE, MAE, NSE]
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
				batch_y_hat, batch_pred_para = model(batch_x, batch_z, batch_c, whether_predict=0, PRODA_para=batch_proda_para)

				# 2 compute the objective function
				l1_loss, smooth_l1_loss, l2_loss, param_reg_loss, val_NSE, corr = fun_loss(batch_y_hat, batch_y, batch_pred_para)

				# Compute additional losses if using. Not strictly necessary but this helps us see if there
				# is a difference between the losses for train/validation sets
				# If we are not using them, set them to nan
				param_violation_loss = np.nan
				unconstrained_param_loss = np.nan
				param_matching_loss = np.nan
				jacobian_loss = np.nan
				kan_l1_loss = np.nan
				kan_entropy_loss = np.nan
				kan_coef_loss = np.nan
				kan_coefdiff_loss = np.nan
				kan_coefdiff2_loss = np.nan
				kan_diversity_loss = np.nan

				if "param_violation" in args.losses:
					param_violation_loss = compute_param_violation_loss(model_without_ddp.unconstrained_params)
				if "unconstrained_param" in args.losses:
					unconstrained_param_loss = compute_unconstrained_param_loss(model_without_ddp.unconstrained_params)
				if "param_matching" in args.losses:
					param_matching_loss = compute_param_matching_loss(batch_pred_para, batch_proda_para)

				if "jacobian" in args.losses:
					# # Use jacobian L1
					jacobian = model_without_ddp.get_jacobian()  # [batch, n_params, n_inputs]
					jacobian_sparsity = (jacobian.abs() < 1e-6).float().mean()
					jacobian_loss = jacobian.abs().mean(0).sum()

				if {"kan_l1", "kan_entropy", "kan_coef", "kan_coefdiff", "kan_coefdiff2"} & set(args.losses):
					assert args.model == "kan"

					# NOTE: the lamb values passed are completely unused, as we direclty obtain the individual loss components and weight them later.
					kan_l1_loss, kan_entropy_loss, kan_coef_loss, kan_coefdiff_loss, kan_coefdiff2_loss, kan_conn_cost = model_without_ddp.mlp.reg(reg_metric='edge_backward', lamb_l1=1., lamb_entropy=1., lamb_coef=1., lamb_coefdiff=1.,
																													 				return_indiv=True, flat_entropy=args.kan_flat_entropy)

				if "kan_diversity" in args.losses:
					# TODO Only works with single layer KAN for now. Check plot_functional_relationships for additional.
					kan_importances = model_without_ddp.mlp.edge_scores[0].permute(1, 0)  #.detach().cpu().numpy()
					kan_importances = kan_importances / kan_importances.sum(dim=0, keepdim=True)
					kan_diversity_loss = torch.tensor(0., device=device)
					for prev_relationship in prev_relationships:
						kan_diversity_loss -= misc_utils.kl_divergence(torch.tensor(prev_relationship, device=device), kan_importances)

				loss_dict = {"l1": l1_loss,
				 			"smooth_l1": smooth_l1_loss,
							"l2": l2_loss,
							"param_reg": param_reg_loss,
							"param_violation": param_violation_loss,
							"unconstrained_param": unconstrained_param_loss,
							"param_matching": param_matching_loss,
							"jacobian": jacobian_loss,
							"jacobian_sparsity": jacobian_sparsity,
							"kan_l1": kan_l1_loss,
							"kan_entropy": kan_entropy_loss,
							"kan_coef": kan_coef_loss,
							"kan_coefdiff": kan_coefdiff_loss,
							"kan_coefdiff2": kan_coefdiff2_loss,
							"kan_diversity": kan_diversity_loss}

				# Record losses
				# Store losses in a tensor, in the order of args.losses
				val_losses = torch.stack([loss_dict[loss] for loss in args.losses]).to(device)
				loss_record_val.append(val_losses)
				metrics_record_val.append(torch.tensor([l2_loss.item(), l1_loss.item(), val_NSE.item()], device=device))

				# Record predicted parameters, true/predicted SOC
				all_val_pred_para.append(batch_pred_para)
				all_val_proda_para.append(batch_proda_para)
				all_val_coords.append(batch_c)
				all_val_z.append(batch_z)
				all_val_pred_soc.append(batch_y_hat)
				all_val_true_soc.append(batch_y)
				ibatch = ibatch + 1
		# end for batch_info in val_loader: 
			
		# # record the time
		hist_time = time.time() - start_time

		# Gather losses from all processes
		all_train_losses = [torch.zeros((len(args.losses)), device=device) for _ in range(world_size)]
		all_val_losses = [torch.zeros((len(args.losses)), device=device) for _ in range(world_size)]
		all_train_times = [torch.tensor(0.0, device=device) for _ in range(world_size)]
		all_train_metrics = [torch.zeros((len(metrics_record_train[0])), device=device) for _ in range(world_size)]
		all_val_metrics = [torch.zeros((len(metrics_record_val[0])), device=device) for _ in range(world_size)]
		all_hist_times = [torch.tensor(0.0, device=device) for _ in range(world_size)]

		dist.all_gather(all_train_losses, torch.stack(loss_record_train, dim=0).mean(dim=0))
		dist.all_gather(all_val_losses, torch.stack(loss_record_val, dim=0).mean(dim=0))
		dist.all_gather(all_train_times, torch.tensor(train_time, device=device))
		dist.all_gather(all_train_metrics, torch.stack(metrics_record_train, dim=0).mean(dim=0))
		dist.all_gather(all_val_metrics, torch.stack(metrics_record_val, dim=0).mean(dim=0))
		dist.all_gather(all_hist_times, torch.tensor(hist_time, device=device))

		# record the loss history
		train_loss_history[iepoch, :] = torch.stack(all_train_losses, dim=0).mean(dim=0).detach().cpu().numpy()
		val_loss_history[iepoch, :] = torch.stack(all_val_losses, dim=0).mean(dim=0).detach().cpu().numpy()
		train_metrics_history[iepoch, :] = torch.stack(all_train_metrics, dim=0).mean(dim=0).detach().cpu().numpy()
		val_metrics_history[iepoch, :] = torch.stack(all_val_metrics, dim=0).mean(dim=0).detach().cpu().numpy()


		####################################################
		## Create true vs predicted scatters per 50 epoch ##
		####################################################
		# To produce comprehensive visualizations, for both train/val sets, create tensors of
		# 1) Predicted parameters for each site
		# 2) PRODA parameters for each site
		# 3) Coordinates (longitude/latitude) of each site
		# 4) Depths for each site/observation (a site may have up to 200 observations, usually much less)
		# 5) Predicted SOC for each site/observation
		# 6) True SOC for each site/observation
		# First aggregate for this rank, then combine all ranks.
		# (Note that DistributedSampler contains repeated examples. We do not remove them.)
		all_train_pred_para = torch.cat(all_train_pred_para, dim=0)  # Pred params for this rank
		all_train_proda_para = torch.cat(all_train_proda_para, dim=0)
		all_train_coords = torch.cat(all_train_coords, dim=0)  # Coords for this rank
		all_train_z = torch.cat(all_train_z, dim=0)
		all_train_pred_soc = torch.cat(all_train_pred_soc, dim=0)  # Pred SOC for this rank
		all_train_true_soc = torch.cat(all_train_true_soc, dim=0)
		all_val_pred_para = torch.cat(all_val_pred_para, dim=0)
		all_val_proda_para = torch.cat(all_val_proda_para, dim=0)
		all_val_coords = torch.cat(all_val_coords, dim=0)
		all_val_z = torch.cat(all_val_z, dim=0)
		all_val_pred_soc = torch.cat(all_val_pred_soc, dim=0)
		all_val_true_soc = torch.cat(all_val_true_soc, dim=0)

		# Estimate max examples per rank. Ok for some to be nan
		train_examples_per_rank = len(train_sampler)  # math.ceil(len(train_sampler) / world_size)
		val_examples_per_rank = len(val_sampler)  # math.ceil(len(val_sampler) / world_size)

		# Pad arrays to this length
		def pad_tensor(tensor, new_length, device):
			"""
			Given tensor of shape [L, D], pads it to shape [new_length, D], where the
			extra rows are filled with nan. new_length must be greater than L.
			"""
			padded = torch.full([new_length, tensor.shape[1]], torch.nan, device=device)
			padded[0:tensor.shape[0]] = tensor
			return padded

		all_train_pred_para = pad_tensor(all_train_pred_para, train_examples_per_rank, device)
		all_train_proda_para = pad_tensor(all_train_proda_para, train_examples_per_rank, device)
		all_train_coords = pad_tensor(all_train_coords, train_examples_per_rank, device)
		all_train_z = pad_tensor(all_train_z, train_examples_per_rank, device)
		all_train_pred_soc = pad_tensor(all_train_pred_soc, train_examples_per_rank, device)
		all_train_true_soc = pad_tensor(all_train_true_soc, train_examples_per_rank, device)
		all_val_pred_para = pad_tensor(all_val_pred_para, val_examples_per_rank, device)
		all_val_proda_para = pad_tensor(all_val_proda_para, val_examples_per_rank, device)
		all_val_coords = pad_tensor(all_val_coords, val_examples_per_rank, device)
		all_val_z = pad_tensor(all_val_z, val_examples_per_rank, device)
		all_val_pred_soc = pad_tensor(all_val_pred_soc, val_examples_per_rank, device)
		all_val_true_soc = pad_tensor(all_val_true_soc, val_examples_per_rank, device)

		# Gather SOC/para/coords/depths from all processes
		train_pred_para_list = [torch.full([train_examples_per_rank, len(para_names)], torch.nan, device=device) for _ in range(world_size)]  # Empty list of per-rank pred paras
		train_proda_para_list = [torch.full([train_examples_per_rank, len(para_names)], torch.nan, device=device) for _ in range(world_size)]  # Empty list of per-rank PRODA paras
		train_coords_list = [torch.full([train_examples_per_rank, 2], torch.nan, device=device) for _ in range(world_size)]
		train_z_list = [torch.full([train_examples_per_rank, 200], torch.nan, device=device) for _ in range(world_size)]
		train_pred_soc_list = [torch.full([train_examples_per_rank, 200], torch.nan, device=device) for _ in range(world_size)]  # Empty list of per-rank pred SOCs
		train_true_soc_list = [torch.full([train_examples_per_rank, 200], torch.nan, device=device) for _ in range(world_size)]
		val_pred_para_list = [torch.full([val_examples_per_rank, len(para_names)], torch.nan, device=device) for _ in range(world_size)]
		val_proda_para_list = [torch.full([val_examples_per_rank, len(para_names)], torch.nan, device=device) for _ in range(world_size)]
		val_coords_list = [torch.full([val_examples_per_rank, 2], torch.nan, device=device) for _ in range(world_size)]
		val_z_list = [torch.full([val_examples_per_rank, 200], torch.nan, device=device) for _ in range(world_size)]
		val_pred_soc_list = [torch.full([val_examples_per_rank, 200], torch.nan, device=device) for _ in range(world_size)]
		val_true_soc_list = [torch.full([val_examples_per_rank, 200], torch.nan, device=device) for _ in range(world_size)]
		dist.all_gather(train_pred_para_list, all_train_pred_para)
		dist.all_gather(train_proda_para_list, all_train_proda_para)
		dist.all_gather(train_coords_list, all_train_coords)
		dist.all_gather(train_z_list, all_train_z)
		dist.all_gather(train_pred_soc_list, all_train_pred_soc)
		dist.all_gather(train_true_soc_list, all_train_true_soc)
		dist.all_gather(val_pred_para_list, all_val_pred_para)
		dist.all_gather(val_proda_para_list, all_val_proda_para)
		dist.all_gather(val_coords_list, all_val_coords)
		dist.all_gather(val_z_list, all_val_z)
		dist.all_gather(val_pred_soc_list, all_val_pred_soc)
		dist.all_gather(val_true_soc_list, all_val_true_soc)

		allrank_train_pred_para = torch.cat(train_pred_para_list, dim=0)
		allrank_train_proda_para = torch.cat(train_proda_para_list, dim=0)
		allrank_train_coords = torch.cat(train_coords_list, dim=0)
		allrank_train_z = torch.cat(train_z_list, dim=0)			
		allrank_train_pred_soc = torch.cat(train_pred_soc_list, dim=0)
		allrank_train_true_soc = torch.cat(train_true_soc_list, dim=0)
		allrank_val_pred_para = torch.cat(val_pred_para_list, dim=0)
		allrank_val_proda_para = torch.cat(val_proda_para_list, dim=0)
		allrank_val_coords = torch.cat(val_coords_list, dim=0)
		allrank_val_z = torch.cat(val_z_list, dim=0)
		allrank_val_pred_soc = torch.cat(val_pred_soc_list, dim=0)
		allrank_val_true_soc = torch.cat(val_true_soc_list, dim=0)

		# Compute metrics across all ranks
		allrank_train_mae, _, allrank_train_mse, _, allrank_train_NSE, allrank_train_corr = fun_loss(allrank_train_pred_soc, allrank_train_true_soc, allrank_train_pred_para)
		allrank_val_mae, _, allrank_val_mse, _, allrank_val_NSE, allrank_val_corr = fun_loss(allrank_val_pred_soc, allrank_val_true_soc, allrank_val_pred_para)
		allrank_train_mae, allrank_train_mse, allrank_train_NSE = allrank_train_mae.item(), allrank_train_mse.item(), allrank_train_NSE.item() 
		allrank_val_mae, allrank_val_mse, allrank_val_NSE = allrank_val_mae.item(), allrank_val_mse.item(), allrank_val_NSE.item()

		if args.plot and (iepoch % 50 == 0) and rank == 0:
			print("Creating plots", datetime.now(), flush=True)

			# KAN-specific visualizations
			if args.model == "kan" and not args.residual:  # TODO pruning doesn't work for residual?
				# Produce edge/node importance scores
				model_without_ddp.mlp.attribute()
				model_without_ddp.mlp.node_attribute()

				# Plot the unpruned model
				model_without_ddp.mlp.plot(folder=os.path.join(PLOT_DIR, "splines"), in_vars=var4nn, out_vars=predicted_para_names, scale=5, varscale=0.13)
				plt.savefig(os.path.join(PLOT_DIR, f"epoch{iepoch}_kan_plot.png"))
				plt.close()

			predicted_relationships, default_kl, default_l2 = misc_utils.plot_functional_relationships(args, model_without_ddp, var4nn, predicted_para_names, 
																							  		   PLOT_DIR, f"epoch{iepoch}", true_relationships)


			# # Test functional relationship retrieval
			# if args.labels == "synthetic_function" and args.model != "nn_only":
			# 	# Feature importance by Jacobian
			# 	jacobian = model_without_ddp.get_jacobian()  # [batch, n_params, n_inputs]
			# 	avg_jacobian_magnitude = jacobian.abs().mean(dim=0).detach().cpu().numpy().T  # transpose to [n_inputs, n_params]
			# 	jacobian_importances = avg_jacobian_magnitude / avg_jacobian_magnitude.sum(axis=0, keepdims=True)

			# 	# Special predictor function to use NN (alibi library)
			# 	from alibi.explainers import ALE, PartialDependenceVariance, plot_ale, plot_pd_variance
			# 	@torch.no_grad()
			# 	def predictor(X: np.ndarray) -> np.ndarray:
			# 		assert model_without_ddp.mlp.training == False, "Model must be in eval mode"
			# 		X = torch.as_tensor(X, device=device)
			# 		return model_without_ddp.mlp(X).cpu().numpy()

			# 	# Feature importance: log PDP Variance scores
			# 	with warnings.catch_warnings():  # suppress warnings inside alibi code
			# 		warnings.simplefilter("ignore")
			# 		cached_nn_input = model_without_ddp.new_input
			# 		pd_variance = PartialDependenceVariance(predictor=predictor,
			# 												feature_names=var4nn,
			# 												target_names=predicted_para_names)
			# 		exp_importance = pd_variance.explain(cached_nn_input.detach().cpu().numpy(), method='importance')
			# 		importance_scores = exp_importance.data['feature_importance'].T  # transpose to [n_inputs, n_params]
			# 		pdv_importances = importance_scores / importance_scores.sum(axis=0, keepdims=True)

			# 	# Combine feature importances by different methods
			# 	method_str = "Blackbox-Hybrid" if args.model == "new_mlp" else f"ScIReN {args.num_layers}-layer"
			# 	predicted_importances_all = {f"{method_str} (Jacobian)": jacobian_importances,
			# 								 f"{method_str}\n(Partial Dependence Variance)": pdv_importances}
			# 	if args.model == "kan" and args.num_layers == 1:
			# 		# If using one-layer KAN, read off functional relationships with mask
			# 		kan_importances = model_without_ddp.mlp.edge_scores[0].permute(1, 0).detach().cpu().numpy()
			# 		predicted_importances_all["KAN"] = kan_importances / kan_importances.sum(axis=0, keepdims=True)

			# 	# If functional relationships are known, compare KAN's predicted relationships with ground-truth relationships
			# 	# Save picture of functional relationships. Source: https://stackoverflow.com/questions/69986007/matplotlib-imshow-with-1-color-for-each-discrete-value
			# 	fig, axeslist = plt.subplots(1, len(predicted_importances_all)+1, figsize=(6*(len(predicted_importances_all)+1), 6))
			# 	cmap = plt.get_cmap('Greens')
			# 	for pred_idx, (pred_method, pred_rel) in enumerate(predicted_importances_all.items()):
			# 		relationship_kl = misc_utils.kl_divergence(true_relationships, pred_rel)
			# 		relationship_l2 = math.sqrt(((true_relationships - pred_rel) ** 2).sum())
			# 		im = axeslist[pred_idx].imshow(pred_rel, cmap=cmap, vmin=0, vmax=1)  #, vmin=-0.5, vmax=5.5, cmap=cmap, interpolation="none")
			# 		axeslist[pred_idx].set_xticks(np.arange(len(predicted_para_names)))
			# 		axeslist[pred_idx].set_yticks(np.arange(len(var4nn)))
			# 		axeslist[pred_idx].set_xticklabels(predicted_para_names, rotation='vertical')
			# 		axeslist[pred_idx].set_yticklabels(var4nn)
			# 		axeslist[pred_idx].set_title(f"Predicted by {pred_method}\n(KL: {relationship_kl:.3f}, Euclidean dist: {relationship_l2:.3f})")
			# 	im = axeslist[-1].imshow(true_relationships, cmap=cmap, vmin=0, vmax=1)  #, vmin=-0.5, vmax=5.5, cmap=cmap, interpolation="none")
			# 	axeslist[-1].set_xticks(np.arange(len(predicted_para_names)))
			# 	axeslist[-1].set_yticks(np.arange(len(var4nn)))
			# 	axeslist[-1].set_xticklabels(predicted_para_names, rotation='vertical')
			# 	axeslist[-1].set_yticklabels(var4nn)
			# 	axeslist[-1].set_title("Ground-truth")
			# 	fig.colorbar(im, orientation="vertical", ax=axeslist[-1])
			# 	plt.tight_layout()
			# 	plt.savefig(os.path.join(PLOT_DIR, f"epoch{iepoch}_functional_relationships.png"))
			# 	plt.close()

			# Scatters of true-vs-predicted SOC (grid).
			# Each row represents a layer (or all layers), each column represents a split (train/val)
			titles = ["Train: All Depths", "Val: All Depths"]
			y_hats = [allrank_train_pred_soc.flatten(), allrank_val_pred_soc.flatten()]  # predictions
			ys = [allrank_train_true_soc.flatten(), allrank_val_true_soc.flatten()]  # labels
			LAYER_BOUNDARIES = [0, 0.1, 0.3, 1.0, 50.0]
			for i in range(len(LAYER_BOUNDARIES) - 1):  # Loop through layers
				layer_loc_train = (allrank_train_z >= LAYER_BOUNDARIES[i]) & (allrank_train_z < LAYER_BOUNDARIES[i+1])  # nan considered false, which is good
				layer_loc_val = (allrank_val_z >= LAYER_BOUNDARIES[i]) & (allrank_val_z < LAYER_BOUNDARIES[i+1])  # nan considered false, which is good
				y_hats.extend([allrank_train_pred_soc[layer_loc_train], allrank_val_pred_soc[layer_loc_val]])
				ys.extend([allrank_train_true_soc[layer_loc_train], allrank_val_true_soc[layer_loc_val]])
				layer_str = f'{LAYER_BOUNDARIES[i]}-{LAYER_BOUNDARIES[i+1]}m'
				titles.extend([f'Train: {layer_str}', f'Val: {layer_str}'])
			visualization_utils.plot_true_vs_predicted_multiple(os.path.join(PLOT_DIR, f"epoch{iepoch}_scatters.png"), y_hats, ys, titles, cols=2)

			# Parameter scatters (predicted vs PRODA/prescribed parameters). Each row is a parameter, each column represents a split (train/val)
			if args.model != "nn_only":
				y_hats = []
				ys = []
				titles = []
				for para_idx in para_index:  # Only plot parameters that were predicted by model
					y_hats.extend([allrank_train_pred_para[:, para_idx], allrank_val_pred_para[:, para_idx]])
					ys.extend([allrank_train_proda_para[:, para_idx], allrank_val_proda_para[:, para_idx]])
					para_name = para_names[para_idx]
					titles.extend([f'Train: {para_name}', f'Val: {para_name}'])
				visualization_utils.plot_true_vs_predicted_multiple(os.path.join(PLOT_DIR, f"epoch{iepoch}_para_scatters.png"), y_hats, ys, titles, cols=2)

		if rank == 0:
			# Compute NSE directly on true/predicted values
			train_NSE = round(allrank_train_NSE, 2)
			val_NSE = round(allrank_val_NSE, 2)

			# Save loss history
			train_losses_epoch = {loss: round(train_loss_history[iepoch, loss_idx], 2) for loss_idx, loss in enumerate(args.losses)}
			val_losses_epoch = {loss: round(val_loss_history[iepoch, loss_idx], 2) for loss_idx, loss in enumerate(args.losses)}

			print(f'{datetime.now()} - Epoch {iepoch} Rank {rank} - Train NSE: {train_NSE}, validation NSE: {val_NSE}, time: {train_time:.2f}', flush=True)
			print(f'Train losses ({all_train_pred_soc.shape[0]} examples): {train_losses_epoch}')
			print(f'Validation losses ({all_val_pred_soc.shape[0]} examples): {val_losses_epoch}')
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
			if allrank_val_NSE <= best_val_NSE:  # NOTE: removed the iepoch==0 condition, switched to new way of calculating NSE (on entire dataset)
				print(f'Best model updated at epoch {iepoch}')
				best_model_epoch = torch.tensor(iepoch, device=device)

				checkpoint_best_model = {
					'epoch': iepoch,
					'model_state_dict': model.state_dict(),
					'model_kwargs': model_kwargs,  # Save kwargs used to construct the model
					'optimizer_state_dict': optimizer.state_dict(),
					'best_val_loss': best_val_loss,
					'best_val_NSE': best_val_NSE,
					'best_model_epoch': best_model_epoch,
					'train_loss_history': train_loss_history,
					'val_loss_history': val_loss_history,
					'train_metrics_history': train_metrics_history,
					'val_metrics_history': val_metrics_history,
					'lr_history': lr_history,
					'train_indices': train_loc,
					'val_indices': val_loc,
					'test_indices': test_loc,
					'epochs_without_improvement': epochs_without_improvement,
					'args': args,  # Commandline args
				}
				
				best_model_path = data_dir_output + 'neural_network/' + job_id + '/opt_nn_' + job_id  + '.pt'
				torch.save(checkpoint_best_model, best_model_path)

			# save the training and validation loss history
			train_time = torch.stack(all_train_times).mean().item()
			hist_time = torch.stack(all_hist_times).mean().item()
			loss_file = os.path.join(data_dir_output, "neural_network", job_id, LOSSES_FILENAME)
			if iepoch == 0:  # Write the header if the file doesn't exist yet
				with open(loss_file, mode='w') as f:
					csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
					csv_writer.writerow(['epoch'] + [f'{loss}_loss_train' for loss in args.losses] +
										[f'{loss}_loss_val' for loss in args.losses] +
										['epoch_time', 'cumulative_time', 'best_model_epoch'])
			with open(loss_file, mode='a+') as f:
				csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
				csv_writer.writerow([iepoch] + [train_loss_history[iepoch, loss_idx] for loss_idx, loss in enumerate(args.losses)] +
									[val_loss_history[iepoch, loss_idx] for loss_idx, loss in enumerate(args.losses)] +
									[round(train_time, 2), round(hist_time, 2), best_model_epoch.item()])

			# Metrics file
			nse_file = os.path.join(data_dir_output, "neural_network", job_id, METRICS_FILENAME)
			if iepoch == 0:
				with open(nse_file, mode='w') as f:
					csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
					csv_writer.writerow(['epoch', 'train_MSE', 'train_MAE', 'train_NSE', 'val_MSE', 'val_MAE', 'val_NSE', 'epoch_time', 'cumulative_time', 'best_model_epoch'])
			with open(nse_file, mode='a+') as f:
				csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
				csv_writer.writerow([iepoch, allrank_train_mse, allrank_train_mae, allrank_train_NSE, allrank_val_mse, allrank_val_mae, allrank_val_NSE] +
									[round(train_time, 2), round(hist_time, 2), best_model_epoch.item()])  # NOTE switched to new metrics computed across all ranks

		# Ensure all processes reach this point before proceeding
		dist.barrier()

		# # Add a learning rate scheduler
		if iepoch >= args.bias_only_epochs:
			if args.use_swa and allrank_val_NSE.item() < 0.5 and iepoch > swa_start:
				swa_model.update_parameters(model)
				swa_scheduler.step()
			elif args.scheduler == "reduce_on_plateau":
				old_lr = scheduler.get_last_lr()[0] if iepoch > 0 else args.lr
				scheduler.step(allrank_val_NSE)
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

		if args.loss_weighting == "two_stage" and iepoch > args.second_start:
			args.lambdas = args.second_lambdas

		# Add a early stopping condition
		if allrank_val_NSE <= best_val_NSE:
			best_val_NSE = allrank_val_NSE
			best_val_loss = val_loss_history[iepoch, :]
			epochs_without_improvement = 0
			# Optionally save the model here if it's the best one so far
		else:
			epochs_without_improvement += 1

		# Early stopping condition
		if epochs_without_improvement == patience:
			print("Rank {}: Early stopping due to no improvement after {} epochs.".format(rank, patience))
			break  # exit the epoch loop
		
		# Save checkpoint (1) every save_freq epochs or (2) time limit exceeded 
		time_limit_exceeded = (time.time() - job_begin_time > args.time_limit * 3600)
		if time_limit_exceeded or iepoch % args.save_freq == 0:
			if rank == 0:
				checkpoint = {
					'epoch': iepoch,
					'model_state_dict': model.state_dict(),
					'model_kwargs': model_kwargs,  # Save kwargs used to construct the model
					'optimizer_state_dict': optimizer.state_dict(),
					'best_val_loss': best_val_loss,
					'best_val_NSE': best_val_NSE,
					'best_model_epoch': best_model_epoch,
					'train_loss_history': train_loss_history,
					'val_loss_history': val_loss_history,
					'train_metrics_history': train_metrics_history,
					'val_metrics_history': val_metrics_history,
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

				if time_limit_exceeded:
					print("Rank {}: Runtime exceeded, saving checkpoint and exiting.".format(rank))

					# Create a file to submit the job again
					if args.job_scheduler == 'slurm':
						# Create a file to submit the job again
						with open(job_submit_path + 'Resume' + job_id + '.submit', 'w') as f:
							f.write(f'#!/bin/bash\n')
							f.write(f'#SBATCH -p full\n')
							f.write(f'#SBATCH -J binn_resume\n')
							f.write(f'#SBATCH --gpus {args.num_CPU}\n')
							f.write(f'#SBATCH -c {args.num_CPU*2}\n')
							f.write(f'#SBATCH -N 1 -n 1\n')
							f.write(f'#SBATCH --mem=50GB\n')
							f.write(f'#SBATCH -t 12:00:00\n')
							f.write(f'source ~/.bashrc\n')
							f.write(f'module load cuda\n')
							f.write(f'conda activate binn\n\n')
							f.write(f'python {" ".join(sys.argv)} --time_limit 11.5 --whether_resume 1\n')

						# submit the job again
						submit_command = ['sbatch',
							f'--export=PREVIOUS_JOB_ID={job_id}',
							job_submit_path + 'Resume' + job_id + '.submit']
						# Submit the job and get the new job ID
						try:
							submit_output = subprocess.check_output(submit_command, universal_newlines=True)
							new_job_id = submit_output.strip()
							print(f"New job submitted. New Job ID is {new_job_id}")
						except subprocess.CalledProcessError as e:
							print(f"Failed to submit job: {e.output}")
					else:
						with open(job_submit_path + 'Resume' + job_id + '.submit', 'w') as f:
							f.write(f'#!/bin/bash\n')
							f.write(f'#PBS -A UOKL0017\n')
							f.write(f'#PBS -N DDP_BINN_Resume\n')
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
							f.write(f'python -u /glade/u/home/haodixu/BINN/Server_Script/binns_DDP.py {" ".join(sys.argv)} --whether_resume 1\n')

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
		print(f"Rank {rank}: Stopped training due to NaN encountered in any process.")
		# dist.destroy_process_group()
		# return

	# Ensure all processes reach the end
	dist.barrier()


	##################################################
	# Done training. Load best model for analysis
	##################################################
	# TODO: weights_only=False is not recommended. Should modify code to only save tensors.
	new_checkpoint = torch.load(data_dir_output + 'neural_network/' + job_id + '/opt_nn_' + job_id + '.pt', map_location=device, weights_only=False)
	if args.use_swa:
		# Update batchnorm stats of averaged model (required when using SWA)
		misc_utils.update_bn_custom(train_loader, swa_model, device)
		best_guess_model = swa_model
		best_guess_model.load_state_dict(new_checkpoint['swa_model_state_dict'])
	else:
		best_guess_model = model  # Do not need to create a new model
		best_guess_model.load_state_dict(new_checkpoint['model_state_dict'])
		best_guess_model = best_guess_model.module  # Remove DDP wrapper as this will only be run on one rank
	print("Loaded model for rank: {}".format(rank))

	dist.barrier()

	if rank == 0:
		# Plot all losses throughout training, including NSE
		plot_losses = [[train_metrics_history[~np.any(np.isnan(train_metrics_history), axis=1), 2].flatten().tolist(),
				 		val_metrics_history[~np.any(np.isnan(val_metrics_history), axis=1), 2].flatten().tolist()]]  # NSE first
		plot_labels = ["NSE"] + [f"{loss} loss" for loss in args.losses]
		y_ranges = [(0, 2)] + [None for loss in args.losses]
		plot_splits = ["train", "validation"]
		for loss_idx in range(len(args.losses)):
			plot_losses.append([train_loss_history[:, loss_idx].tolist(),
								val_loss_history[:, loss_idx].tolist()])
		visualization_utils.plot_multiple_losses(os.path.join(PLOT_DIR, "all_losses.png"), plot_losses, plot_labels, plot_splits, y_ranges)

		# Plot learning rate schedule
		if scheduler is not None:
			plt.plot(np.arange(lr_history.size), lr_history)
			plt.xlabel('Epoch #')
			plt.ylabel('Learning rate')
			plt.title('Learning rate schedule')
			plt.savefig(os.path.join(PLOT_DIR, "lr_schedule.png"))
			plt.close()


		#######################################################
		# Get best model's predictions on train/val/test sets
		#######################################################
		# TODO: This takes a long time and should be distributed.
		print("Rank 0 beginning prediction at time {}".format(datetime.now()))
		best_guess_model.eval()
		with torch.no_grad():
			# Get predictions for train examples, compute loss & plot
			best_guess_train_y_hat, best_guess_train_pred_para = best_guess_model(train_x.to(device), train_z.to(device), train_c.to(device),
																				  whether_predict=0, PRODA_para=train_proda_para.to(device))
			train_mae, train_smooth_l1_loss, train_mse, _, train_NSE, train_corr = fun_loss(best_guess_train_y_hat, train_y.to(device), best_guess_train_pred_para)
			print(f'Train - MSE: {train_mse.item():.2f}, MAE: {train_mae.item():.2f}, NSE: {train_NSE.item():.2f}')

			# Get predictions for val examples, compute loss & plot
			best_guess_val_y_hat, best_guess_val_pred_para = best_guess_model(val_x.to(device), val_z.to(device), val_c.to(device),
																			  whether_predict=0, PRODA_para=val_proda_para.to(device))
			val_mae, val_smooth_l1_loss, val_mse, _, val_NSE, val_corr = fun_loss(best_guess_val_y_hat, val_y.to(device), best_guess_val_pred_para)
			print(f'Val - MSE: {val_mse.item():.2f}, MAE: {val_mae.item():.2f}, NSE: {val_NSE.item():.2f}')

			if test_split_ratio != 0:
				# Get predictions for test examples, compute loss & plot
				best_guess_test_y_hat, best_guess_test_pred_para = best_guess_model(test_x.to(device), test_z.to(device), test_c.to(device),
																					whether_predict=0, PRODA_para=test_proda_para.to(device))
				test_mae, test_smooth_l1_loss, test_mse, _, test_NSE, test_corr = fun_loss(best_guess_test_y_hat, test_y.to(device), best_guess_test_pred_para)
				print(f'Test - MSE: {test_mse.item():.2f}, MAE: {test_mae.item():.2f}, NSE: {test_NSE.item():.2f}')

				pred_para = best_guess_test_pred_para.flatten()
				true_para = test_proda_para.flatten()
				test_para_NSE = torch.sum((pred_para - true_para)**2)/torch.sum((true_para - torch.mean(true_para))**2)
				test_para_corr = torch.corrcoef(torch.stack((pred_para, true_para)))[0, 1]

			# Also generate PREDICTED SOC for EACH SOIL LAYER (20)
			train_simu_all_layers, _ = best_guess_model(train_x.to(device), train_z.to(device), train_c.to(device), whether_predict=1, PRODA_para=train_proda_para.to(device))
			val_simu_all_layers, _ = best_guess_model(val_x.to(device), val_z.to(device), val_c.to(device), whether_predict=1, PRODA_para=val_proda_para.to(device))
			test_simu_all_layers, _ = best_guess_model(test_x.to(device), test_z.to(device), test_c.to(device), whether_predict=1, PRODA_para=test_proda_para.to(device))
			simu_soc_all_layers = torch.tensor(np.ones((wosis_profile_info.shape[0], 20))*np.nan, device=device, dtype = torch.float32)  # dtype = torch.float32,
			simu_soc_all_layers[train_profile_id, :] = train_simu_all_layers[:, 0:20]
			simu_soc_all_layers[val_profile_id, :] = val_simu_all_layers[:, 0:20]
			simu_soc_all_layers[test_profile_id, :] = test_simu_all_layers[:, 0:20]

		# create folder for the results
		os.makedirs(data_dir_output + 'neural_network/' + job_id + '/Validation', exist_ok=True)
		os.makedirs(data_dir_output + 'neural_network/' + job_id + '/Train', exist_ok=True)
		if test_split_ratio != 0:
			os.makedirs(data_dir_output + 'neural_network/' + job_id + '/Test', exist_ok=True)
		print("----------------- Finished train/val/test prediction " + str(datetime.now()) + "-----------------")

		# KAN-specific visualizations
		cached_nn_input = best_guess_model.new_input  # Store nn input in case following methods change it
		if args.model == "kan" and not args.residual:  # TODO pruning doesn't work for residual?
			# Produce edge/node importance scores
			best_guess_model.mlp.attribute()
			best_guess_model.mlp.node_attribute()

			# Plot the pruned model
			pruned_model = best_guess_model.mlp.prune()  # edge_quantile_th=0.2)  # node_th=0.03, edge_th=0.03)
			pruned_model.plot(folder=os.path.join(PLOT_DIR, "splines_pruned"), in_vars=var4nn, out_vars=predicted_para_names, scale=5, varscale=0.12)
			plt.savefig(os.path.join(PLOT_DIR, f"FINAL_kan_plot_pruned.png"))
			plt.close()

			# Plot the unpruned model
			best_guess_model.mlp.plot(folder=os.path.join(PLOT_DIR, "splines"), in_vars=var4nn, out_vars=predicted_para_names, scale=5, varscale=0.12)
			plt.savefig(os.path.join(PLOT_DIR, f"FINAL_kan_plot.png"))
			plt.close()

		# # NOTE Try testing on the pruned model!
		best_guess_model.new_input = cached_nn_input
		predicted_relationships, default_kl, default_l2 = misc_utils.plot_functional_relationships(args, best_guess_model, var4nn, predicted_para_names, PLOT_DIR, "FINAL", true_relationships)
		# best_guess_model.mlp.act_fun[0].mask = pruned_model.act_fun[0].mask
		print("PRedicted relationships shape", predicted_relationships.shape)
		np.save(os.path.join(PLOT_DIR, "predicted_relationships.npy"), predicted_relationships)  #.detach().cpu().numpy())


		# Summary csv file of all results. Create this if it doesn't exist
		results_summary_file = os.path.join(data_dir_output, f"neural_network/results_summary_{args.note}.csv")
		if not os.path.isfile(results_summary_file):
			with open(results_summary_file, mode='w') as f:
				csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
				csv_writer.writerow(['job_id', 'command', 'data_string', 'lr', 'weight_decay', 'seed', 'model_path', 'val_MSE', 'val_MAE', 'val_NSE', 'val_corr', 'test_MSE', 'test_MAE', 'test_NSE', 'test_corr', 'test_para_NSE', 'test_para_corr', 'relationship_kl', 'relationship_l2'])
		command_string = " ".join(sys.argv)
		data_string = f"Fold {args.cross_val_idx} {args.split} (data_seed = {args.data_seed}, n_datapoints = {args.n_datapoints})"

		# Add a row to the summary csv file
		with open(results_summary_file, mode='a+') as f:
			csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			best_model_path = data_dir_output + 'neural_network/' + job_id + '/opt_nn_' + job_id  + '.pt'
			csv_writer.writerow([job_id, command_string, data_string, args.lr, args.weight_decay, args.seed, best_model_path, val_mse.item(), val_mae.item(), 1-val_NSE.item(), val_corr.item(), test_mse.item(), test_mae.item(), 1-test_NSE.item(), test_corr.item(), 1-test_para_NSE.item(), test_para_corr.item(), default_kl, default_l2])


	if rank == 0 and args.plot:
		# Predict the SOC values based on Grid environmental information using the best model
		grid_simu_soc, grid_pred_para = best_guess_model(predict_data_x.to(device), predict_data_z.to(device), predict_data_c.to(device),
														 whether_predict = 1, PRODA_para=grid_PRODA_para.to(device))

		# Save the predicted SOC values, parameters and location data into csv files
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_simu_soc_' + job_id + '.csv', grid_simu_soc.detach().cpu().numpy(), delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_pred_para_' + job_id + '.csv', grid_pred_para.detach().cpu().numpy(), delimiter = ',')
		# save grid_env_info_US['Original_Lat'] and grid_env_info_US['Original_Lon'] to csv files
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_lons_' + job_id + '.csv', grid_env_info_US['original_lon'], delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_lats_' + job_id + '.csv', grid_env_info_US['original_lat'], delimiter = ',')
		print("----------------- Predictions for grid data " + str(datetime.now()) + "-----------------")

		# FINAL SUMMARY MAPS
		# Scatters of true-vs-predicted SOC (grid).
		# Each row represents a layer (or all layers), each column represents a split (train/val/test)
		titles = ["Train: All Depths", "Val: All Depths", "Test: All Depths"]
		y_hats = [best_guess_train_y_hat.flatten(), best_guess_val_y_hat.flatten(), best_guess_test_y_hat.flatten()]  # predictions
		ys = [train_y.flatten().to(device), val_y.flatten().to(device), test_y.flatten().to(device)]  # labels
		LAYER_BOUNDARIES = [0, 0.1, 0.3, 1.0, 50.0]
		for i in range(len(LAYER_BOUNDARIES) - 1):  # Loop through layers
			layer_loc_train = (train_z >= LAYER_BOUNDARIES[i]) & (train_z < LAYER_BOUNDARIES[i+1])  # True for observations within this layer that are non-nan
			layer_loc_val = (val_z >= LAYER_BOUNDARIES[i]) & (val_z < LAYER_BOUNDARIES[i+1]) 
			layer_loc_test = (test_z >= LAYER_BOUNDARIES[i]) & (test_z < LAYER_BOUNDARIES[i+1]) 
			y_hats.extend([best_guess_train_y_hat[layer_loc_train], best_guess_val_y_hat[layer_loc_val], best_guess_test_y_hat[layer_loc_test]])
			ys.extend([train_y[layer_loc_train].to(device), val_y[layer_loc_val].to(device), test_y[layer_loc_test].to(device)])
			layer_str = f'{LAYER_BOUNDARIES[i]}-{LAYER_BOUNDARIES[i+1]}m'
			titles.extend([f'Train: {layer_str}', f'Val: {layer_str}', f'Test: {layer_str}'])
		visualization_utils.plot_true_vs_predicted_multiple(os.path.join(PLOT_DIR, f"FINAL_scatters.png"), y_hats, ys, titles, cols=3)
		print("----------------- True vs predicted scatters " + str(datetime.now()) + "-----------------")

		# Maps of true-vs-predicted SOC (grid)
		# Each row represents a layer, each column represents a split (train/val/test) and {true or predicted}
		lons_list = []
		lats_list = []
		values_list = []
		vars_list = []

		for i in range(len(LAYER_BOUNDARIES) - 1):
			# For each site: compute average SOC over observations in this layer
			layer_loc_train = (train_z >= LAYER_BOUNDARIES[i]) & (train_z < LAYER_BOUNDARIES[i+1])  # True for observations within this layer that are non-nan
			train_true_soc = torch.where(layer_loc_train.to(device), train_y.to(device), torch.nan)  # Create tensor: only observations in this layer, nan elsewhere
			train_true_soc = torch.nanmean(train_true_soc, dim=1)  # For each site, average over observations in this layer. If none, return nan.
			train_pred_soc = torch.where(layer_loc_train.to(device), best_guess_train_y_hat, torch.nan)  # Same for predictions
			train_pred_soc = torch.nanmean(train_pred_soc, dim=1)

			# Repeat above for val data
			layer_loc_val = (val_z >= LAYER_BOUNDARIES[i]) & (val_z < LAYER_BOUNDARIES[i+1])  # True for observations within this layer that are non-nan
			val_true_soc = torch.where(layer_loc_val.to(device), val_y.to(device), torch.nan)  # Create tensor: only observations in this layer, nan elsewhere
			val_true_soc = torch.nanmean(val_true_soc, dim=1)  # For each site, average over observations in this layer. If none, return nan.
			val_pred_soc = torch.where(layer_loc_val.to(device), best_guess_val_y_hat, torch.nan)  # Same for predictions
			val_pred_soc = torch.nanmean(val_pred_soc, dim=1)

			# Repeat above for test data
			layer_loc_test = (test_z >= LAYER_BOUNDARIES[i]) & (test_z < LAYER_BOUNDARIES[i+1])  # True for observations within this layer that are non-nan
			test_true_soc = torch.where(layer_loc_test.to(device), test_y.to(device), torch.nan)  # Create tensor: only observations in this layer, nan elsewhere
			test_true_soc = torch.nanmean(test_true_soc, dim=1)  # For each site, average over observations in this layer. If none, return nan.
			test_pred_soc = torch.where(layer_loc_test.to(device), best_guess_test_y_hat, torch.nan)  # Same for predictions
			test_pred_soc = torch.nanmean(test_pred_soc, dim=1)

			# Collect results
			lons_list.extend([train_c[:, 0], train_c[:, 0], val_c[:, 0], val_c[:, 0], test_c[:, 0], test_c[:, 0]])
			lats_list.extend([train_c[:, 1], train_c[:, 1], val_c[:, 1], val_c[:, 1], test_c[:, 1], test_c[:, 1]])
			values_list.extend([train_true_soc, train_pred_soc, val_true_soc, val_pred_soc, test_true_soc, test_pred_soc])
			layer_str = f'{LAYER_BOUNDARIES[i]}-{LAYER_BOUNDARIES[i+1]}m'
			vars_list.extend([f'True SOC - Train: {layer_str}', f'Predicted SOC - Train: {layer_str}',
								f'True SOC - Val: {layer_str}', f'Predicted SOC - Val: {layer_str}',
								f'True SOC - Test: {layer_str}', f'Predicted SOC - Test: {layer_str}',])
		visualization_utils.plot_map_grid(os.path.join(PLOT_DIR, f"FINAL_soc_maps.png"),
				lons_list, lats_list, values_list, vars_list, us_only=True, cols=6)
		print("----------------- True vs predicted maps " + str(datetime.now()) + "-----------------")

		if args.model != "nn_only":
			# Parameter maps. Each row is a parameter, each column represents a split (train/val/test/grid).
			# Use PRODA parameters as "labels" to compare with our predicted parameters
			lons_list = []
			lats_list = []
			values_list = []
			vars_list = []
			for para_idx in para_index:  # Only plot parameters that were predicted by NN
				# NOTE: Only plot the grid maps for now as this takes a long time.
				lons_list.extend([predict_data_c[:, 0], predict_data_c[:, 0]])
				lats_list.extend([predict_data_c[:, 1], predict_data_c[:, 1]])
				values_list.extend([grid_PRODA_para[:, para_idx].to(device), grid_pred_para[:, para_idx]])
				para_name = para_names[para_idx]
				vars_list.extend([f'PRODA para {para_name} - Grid', f'Predicted para {para_name} - Grid'])

				# lons_list.extend([train_c[:, 0], train_c[:, 0], val_c[:, 0], val_c[:, 0], test_c[:, 0], test_c[:, 0], predict_data_c[:, 0], predict_data_c[:, 0]])
				# lats_list.extend([train_c[:, 1], train_c[:, 1], val_c[:, 1], val_c[:, 1], test_c[:, 1], test_c[:, 1], predict_data_c[:, 1], predict_data_c[:, 1]])
				# values_list.extend([train_proda_para[:, para_idx].to(device), best_guess_train_pred_para[:, para_idx],
				# 					val_proda_para[:, para_idx].to(device), best_guess_val_pred_para[:, para_idx],
				# 					test_proda_para[:, para_idx].to(device), best_guess_test_pred_para[:, para_idx],
				# 					grid_PRODA_para[:, para_idx].to(device), grid_pred_para[:, para_idx]])
				# para_name = para_names[para_idx]
				# vars_list.extend([f'PRODA para {para_name} - Train', f'Predicted para {para_name} - Train',
				# 					f'PRODA para {para_name} - Val', f'Predicted para {para_name} - Val',
				# 					f'PRODA para {para_name} - Test', f'Predicted para {para_name} - Test',
				# 					f'PRODA para {para_name} - Grid', f'Predicted para {para_name} - Grid'])
			visualization_utils.plot_map_grid(os.path.join(PLOT_DIR, f"FINAL_para_maps.png"),
					lons_list, lats_list, values_list, vars_list, us_only=True, cols=2)

			# Also plot scatters
			y_hats = []
			ys = []
			titles = []
			for para_idx in para_index:  # Only plot parameters that were predicted by NN
				y_hats.extend([best_guess_train_pred_para[:, para_idx], best_guess_val_pred_para[:, para_idx], best_guess_test_pred_para[:, para_idx], grid_pred_para[:, para_idx]])
				ys.extend([train_proda_para[:, para_idx], val_proda_para[:, para_idx], test_proda_para[:, para_idx], grid_PRODA_para[:, para_idx].to(device)])
				para_name = para_names[para_idx]
				titles.extend([f'Train: {para_name}', f'Val: {para_name}', f'Test: {para_name}', f'Grid: {para_name}'])
			visualization_utils.plot_true_vs_predicted_multiple(os.path.join(PLOT_DIR, f"FINAL_para_scatters.png"), y_hats, ys, titles, cols=4)

		print("-----------------Model Prediction Finished at " + str(datetime.now()) + "-----------------")

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


def find_free_port():
	"""Finds an unused port."""
	import socket
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		s.bind(('', 0))
		s.listen(1)
		return str(s.getsockname()[1])


if __name__ == '__main__':

	# Number of CPUs requester
	world_size = args.num_CPU
	processes = []

	# Create job ID
	job_id = create_output_folders(args)
	print("MAIN, JOB ID", job_id)
	print("Command:", " ".join(sys.argv))

	# Find free port
	port = find_free_port()

	# If not using DDP, just call the worker directly
	if args.use_ddp == 0:
		worker(rank=0, world_size=1, job_id=job_id, port=port)
		exit(0)

	# Spawn method is required if using GPU
	if torch.cuda.is_available():
		assert world_size == len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")), "If using GPU: world_size (num_CPU) must equal number of GPUs in CUDA_VISIBLE_DEVICES"
	import torch.multiprocessing as mp
	mp.set_start_method('spawn', force=True)

	# Create the processes
	for rank in range(world_size):
		p = mp.Process(target=worker, args=(rank, world_size, job_id, port))
		p.start()
		processes.append(p)

	for p in processes:
		p.join()

	
