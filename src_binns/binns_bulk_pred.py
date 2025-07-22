"""
Main prediction script for BINN, using DistributedDataParallel.

Run global prediction and bulk prediction with the best KAN model.

Percentage change of input data causes the change in the output data
"""
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
# from spatial_utils import *
from losses import binns_loss, compute_param_matching_loss, compute_param_violation_loss, compute_unconstrained_param_loss
import visualization_utils

# sys.path.append('C:/Users/hx293/Research_Data/BINN/')
# sys.path.append('/glade/u/home/haodixu/BINN')
# sys.path.append(r'/User/homes/ftao/Projects/BINNS/src_binns')

# Set HDF5_DISABLE_VERSION_CHECK to suppress version mismatch error
import os
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2'

from datetime import datetime, timedelta
import pandas as pd
from pandas import DataFrame as df
import numpy as np
from scipy.interpolate import pchip_interpolate

print("Start binns_DDP")



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

# @joshuafan: previously we set default dtype to float64 to avoid underflow in process-based model.
# Now checking float32 with fixed process-based model.
torch.set_default_dtype(torch.float32)

###################################
# Import CLM5 process-based model #
###################################
# fun_model_simu predicts at user-specified depths. fun_model_prediction predicts at 20 default layers.
from fun_matrix_COMPAS_Hardy import fun_model_simu, fun_model_prediction
from fun_matrix_COMPAS_Hardy_bulk_converge import fun_bulk_simu

# KAN Model
import pykan_josh

device = 'cpu'

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
parser.add_argument("--kan_absolute_deviation", action='store_true')
parser.add_argument("--kan_flat_entropy", type=int, default=1)

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
parser.add_argument("--soc_only_epochs", type=int, default=0, help="If set, train only on SOC data for this many epochs")
parser.add_argument("--best_model_val_metric", type=str, choices=["average_l1", "seperate_l1"], default="average_l1", help="Metric to use for selecting the best model. average_l1 is average of all L1 losses, seperate_l1 is L1 loss for soc and pom_maom.")

# Regularization
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--use_swa", action='store_true', help="Whether to use Stochastic Weight Averaging")
parser.add_argument("--clip_value", type=float, default=-1, help="Clip value for gradient clipping. -1 for no clipping.")

# Losses and loss weights
parser.add_argument("--losses", nargs="+", choices=["l1_loss", "smooth_l1", "l2_loss", "Smooth_l1_loss_SOC", "Smooth_l1_loss_POM", "Smooth_l1_loss_MAOM", "param_reg", "param_violation", "unconstrained_param", "param_matching", "jacobian",
                                                    "jacobian_sparsity", "spectral", "lipmlp", "cure", "senn_robustness", "senn_l1", "senn_sparsity", 
                                                    "nam_l2", "nam_entropy", "kan_l1", "kan_entropy", "kan_coef", "kan_coefdiff", "kan_coef_l1", "kan_coefdiff_l2", "kan_coefdiff2_l2", 
                                                    "spatial_error", "spatial_emb_smoothness", "param_smoothness", "residual"], default=["smooth_l1", "param_reg"],
                    help="Losses to use (can list any number). Note jacobian_sparsity cannot be optimized (non-differentiable): it is just something we track.")
parser.add_argument("--loss_weighting", default="manual", choices=["manual", "relobralo", "IMTL", "two_stage"])
parser.add_argument("--lambdas", nargs="+", type=float, default=[1.0, 10.0], help="If loss_weighting is manual, provide weights in the same order as `args.losses`")
parser.add_argument("--second_start", type=int, default=30, help="If loss_weighting is two_stage, epoch the second phase starts")
parser.add_argument("--second_lambdas", nargs="+", type=float, default=[1.0, 10.0], help="If loss_weighting is two_stage, weights for the second stage - in the same order as `args.losses`")
parser.add_argument("--param_reg_scale", type=float, default=10, help="Scale for param_reg loss.")

# Process-based model parameters
for para_idx in range(22):
    parser.add_argument(
        f"--para_{para_idx}",
        nargs=2, type=float, metavar=("LOW", "HIGH"),
        help=f"Lower/upper bound for para[{para_idx}]"
    )
    
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

################################################
# Data Directories (CHANGE THIS!!!)
################################################
# job_id = '20250630-003325_KAN_ONELAYER_GRIDMARGIN_COMPAS2_9917068_lr=1e-02_fold=0_seed=111'
# job_id = '20250630-220008_KAN_ONELAYER_GRIDMARGIN_COMPAS2_9926268_lr=1e-02_fold=0_seed=111'
# job_id = '20250704-212930_KAN_ONELAYER_GRIDMARGIN_COMPAS2_998751_lr=1e-02_fold=0_seed=111'
# job_id = '20250704-213024_KAN_ONELAYER_GRIDMARGIN_COMPAS2_998996_lr=1e-02_fold=0_seed=111'
job_id = '20250704-213005_KAN_ONELAYER_GRIDMARGIN_COMPAS2_998897_lr=1e-02_fold=0_seed=111'
# server path
job_submit_path = '/glade/u/home/haodixu/BINN/PBS_Submit/KAN_COMPAS2/'
data_dir_input = '/glade/u/home/haodixu/BINN/ENSEMBLE/INPUT_DATA/'
model_dir_input = '/glade/work/haodixu/BINN/BINNS/OUTPUT_DATA/neural_network/'
data_dir_output = model_dir_input + job_id + '/KAN_Propotional_Change_2_std/'

if not os.path.exists(data_dir_output):
    os.makedirs(data_dir_output)

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
    para_names = ['diffus', 'cryo', 'q10', 'efolding', 'taucwd', 'taul1', 'taul2', 'tau4doc', 'tau4mic', 'tau4poc', 'tau4maom', 'fl1_MIC', 'fl2_POC', 'fMIC_MAOM', 'fMIC_POC', 'fDOC_MAOM', 'CUEl1', 'CUEl2', 'CUEDOC', 'w-scaling', 'beta']
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
    var4nn = ["BIO1", "BIO12", "BIO3", "BIO15", \
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

# Add average values for Ald, Alo, Fed, and Feo
env_info["0.5_Feo_avg_Alo_avg"] = (env_info["Feo_avg"] / 2 + env_info["Alo_avg"])
env_info["0.5_Fed_avg_Ald_avg"] = (env_info["Fed_avg"] / 2 + env_info["Ald_avg"])
# Update the max/min for the new columns
# Feo_avg and Alo_avg
indices_Feo = env_info.columns.get_loc("Feo_avg")
indices_Alo = env_info.columns.get_loc("Alo_avg")
indices_Fed = env_info.columns.get_loc("Fed_avg")
indices_Ald = env_info.columns.get_loc("Ald_avg")
# Get new min by adding up the min of 0.5*Feo_avg and Alo_avg
min_temp = 0.5*col_max_min[indices_Feo, 0] + col_max_min[indices_Alo, 0]
max_temp = 0.5*col_max_min[indices_Feo, 1] + col_max_min[indices_Alo, 1]
new_max_min = np.array([[min_temp, max_temp]]) 
col_max_min = np.vstack((col_max_min, new_max_min))
# env_info["0.5_Feo_avg_Alo_avg"] = (env_info["0.5_Feo_avg_Alo_avg"] - col_max_min[-1, 0])/(col_max_min[-1, 1] - col_max_min[-1, 0])
# Fed_avg and Ald_avg
min_temp = 0.5*col_max_min[indices_Fed, 0] + col_max_min[indices_Ald, 0]
max_temp = 0.5*col_max_min[indices_Fed, 1] + col_max_min[indices_Ald, 1]
new_max_min = np.array([[min_temp, max_temp]])
col_max_min = np.vstack((col_max_min, new_max_min))
# env_info["0.5_Fed_avg_Ald_avg"] = (env_info["0.5_Fed_avg_Ald_avg"] - col_max_min[-1, 0])/(col_max_min[-1, 1] - col_max_min[-1, 0])
env_info["Clay_Silt_avg"] = (env_info["Clay_Content_avg"] + env_info["Silt_Content_avg"])
indices_Clay = env_info.columns.get_loc("Clay_Content_avg")
indices_Silt = env_info.columns.get_loc("Silt_Content_avg")
# Get new min by adding up the min of Clay_Content_avg and Silt_Content_avg
min_temp = col_max_min[indices_Clay, 0] + col_max_min[indices_Silt, 0]
max_temp = col_max_min[indices_Clay, 1] + col_max_min[indices_Silt, 1]
new_max_min = np.array([[min_temp, max_temp]])
col_max_min = np.vstack((col_max_min, new_max_min))
print("max/min for Clay_Silt_avg: ", col_max_min[-1, :])


# Update var4nn to include the new columns
if args.features == "ten":
    # Ten handcrafted features
    var4nn = ["BIO1", "BIO12", "BIO3", "BIO15", \
        # "Clay_Content_avg", "Sand_Content_avg","Silt_Content_avg", \
        "Clay_Silt_avg",\
        "Bulk_Density_avg",\
        "SWC_v_Wilting_Point_avg", "pH_Water_avg", "CEC_avg", \
        "Coarse_Fragments_avg", \
        # "cesm2_npp", "cesm2_vegc", \
        # '0.5_Feo_avg_Alo_avg', \
        # '0.5_Fed_avg_Ald_avg', \
        ]
	

print("Size of col_max_min: ", col_max_min.shape)

# Save the min/max values for each feature within var4nn
env_info_max_min_save = np.zeros((len(var4nn), 2))
# Scale numeric features to [0, 1] based on precomputed min/max 
warnings.filterwarnings("ignore")  # Ignore warnings about subtracting nan
for ivar in np.arange(3, len(col_max_min[:, 0])):
    if np.isnan(col_max_min[ivar, :]).any():
        pass
    else:
        env_info.iloc[:, ivar] = (env_info.iloc[:, ivar] - col_max_min[ivar, 0])/(col_max_min[ivar, 1] - col_max_min[ivar, 0])
        env_info.iloc[(env_info.iloc[:, ivar] > 1), ivar] = 1
        env_info.iloc[(env_info.iloc[:, ivar] < 0), ivar] = 0
        if env_info.columns[ivar] in var4nn:
            env_info_max_min_save[var4nn.index(env_info.columns[ivar]), :] = col_max_min[ivar, :]
            
warnings.resetwarnings()

# Retain orginal lat/lon
env_info["original_lon"] = original_lons
env_info["original_lat"] = original_lats

# Save the min/max values for each feature within var4nn into a txt file
np.savetxt(data_dir_output + '/env_info_max_min.txt', env_info_max_min_save, delimiter=',', header=','.join(var4nn), comments='')


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

# Convert data to torch tensors
current_data_x = torch.tensor(current_data_x, dtype=torch.float32)
current_data_y = torch.tensor(current_data_y, dtype=torch.float32)
current_data_z = torch.tensor(current_data_z, dtype=torch.float32)
current_data_c = torch.tensor(current_data_c, dtype=torch.float32)
current_data_profile_id = torch.tensor(current_data_profile_id, dtype=torch.int64)
current_PRODA_para = torch.tensor(np.ones((len(valid_profile_loc), 1)), dtype=torch.float32)  # Placeholder for PRODA parameters, if needed

# Store the mean values, std, and the range of the env_info
env_info_mean = torch.mean(current_data_x[:, 0:len(var4nn), 0, 0], dim=0)
env_info_std = torch.std(current_data_x[:, 0:len(var4nn), 0, 0], dim=0)
env_info_max = torch.max(current_data_x[:, 0:len(var4nn), 0, 0], dim=0).values
env_info_min = torch.min(current_data_x[:, 0:len(var4nn), 0, 0], dim=0).values

print("Training with features: ", var4nn)
print("Mean of env_info features: ", env_info_mean)
print("Std of env_info features: ", env_info_std)
print("Max of env_info features: ", env_info_max)
print("Min of env_info features: ", env_info_min)

# Save the mean and std of the env_info features
np.savetxt(data_dir_output + '/env_info_mean.txt', env_info_mean.numpy(), delimiter=',', header=','.join(var4nn), comments='')
np.savetxt(data_dir_output + '/env_info_std.txt', env_info_std.numpy(), delimiter=',', header=','.join(var4nn), comments='')
np.savetxt(data_dir_output + '/env_info_max_norm.txt', env_info_max.numpy(), delimiter=',', header=','.join(var4nn), comments='')
np.savetxt(data_dir_output + '/env_info_min_norm.txt', env_info_min.numpy(), delimiter=',', header=','.join(var4nn), comments='')


#---------------------------------------------------
# Grid env info for prediction
#---------------------------------------------------
# load grid env info
grid_env_info = np.genfromtxt(data_dir_input + 'wosis_2019_snap_shot/env_info_grid_with_Al_Fe.csv', delimiter = ',', skip_header = 1)
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
    'nbedrock', \
    'Ald_0_20',	'Ald_20_40', 'Ald_40_60', 'Ald_60_80', 'Ald_80_100', 'Alo_0_20', 'Alo_20_40', 'Alo_40_60', 'Alo_60_80', 'Alo_80_100',\
    'Fed_0_20', 'Fed_20_40', 'Fed_40_60', 'Fed_60_80', 'Fed_80_100', 'Feo_0_20', 'Feo_20_40', 'Feo_40_60', 'Feo_60_80', 'Feo_80_100'
]

grid_env_info = df(grid_env_info)
grid_env_info.columns = grid_env_info_names

# Remove the first 3 columns and the R_squared column from the col_max_min matrix
col_max_min_grid = np.delete(col_max_min, env_info_names.index("R_Squared"), axis=0)[3:, :]

# Logic to add columns for "average" variables (e.g. average over layers)
var4nn = ["BIO1", "BIO12", "BIO3", "BIO15", \
    "Clay_Content_avg", "Sand_Content_avg","Silt_Content_avg", \
    "Bulk_Density_avg",\
    "SWC_v_Wilting_Point_avg", "pH_Water_avg", "CEC_avg", \
    "Coarse_Fragments_avg", \
    #"cesm2_npp", "cesm2_vegc", \
    'Ald_avg', 'Alo_avg',\
    'Fed_avg', 'Feo_avg', \
    ]

for v in var4nn:
    if v not in env_info_names:
        var_prefix = v.split("_avg")[0]
        columns = grid_env_info.filter(regex=(f"{var_prefix}*"))
        grid_env_info[v] = columns.mean(axis=1)

grid_env_info["0.5_Feo_avg_Alo_avg"] = (grid_env_info["Feo_avg"] / 2 + grid_env_info["Alo_avg"])
grid_env_info["0.5_Fed_avg_Ald_avg"] = (grid_env_info["Fed_avg"] / 2 + grid_env_info["Ald_avg"])
grid_env_info["Clay_Silt_avg"] = (grid_env_info["Clay_Content_avg"] + grid_env_info["Silt_Content_avg"])

var4nn = ["BIO1", "BIO12", "BIO3", "BIO15", \
    # "Clay_Content_avg", "Sand_Content_avg","Silt_Content_avg", \
    "Clay_Silt_avg",\
    "Bulk_Density_avg",\
    "SWC_v_Wilting_Point_avg", "pH_Water_avg", "CEC_avg", \
    "Coarse_Fragments_avg", \
    # "cesm2_npp", "cesm2_vegc", \
    # '0.5_Feo_avg_Alo_avg', \
    # '0.5_Fed_avg_Ald_avg', \
    ]
# Save the col_max_min for the var4nn variables
col_max_min_grid_save = np.zeros((len(var4nn), 2))
# Normalize grid env info
for ivar in np.arange(0, len(col_max_min_grid[:, 0])):
    if np.isnan(col_max_min_grid[ivar, :]).any():
        pass
    else:
        grid_env_info.iloc[:, ivar] = (grid_env_info.iloc[:, ivar] - col_max_min_grid[ivar, 0])/(col_max_min_grid[ivar, 1] - col_max_min_grid[ivar, 0])
        grid_env_info.iloc[(grid_env_info.iloc[:, ivar] > 1), ivar] = 1
        grid_env_info.iloc[(grid_env_info.iloc[:, ivar] < 0), ivar] = 0
        if grid_env_info.columns[ivar] in var4nn:
            col_max_min_grid_save[var4nn.index(grid_env_info.columns[ivar]), :] = col_max_min_grid[ivar, :]

# Save the min/max values for each feature within var4nn into a txt file
np.savetxt(data_dir_output + '/grid_env_info_max_min.txt', col_max_min_grid_save, delimiter=',', header=','.join(var4nn), comments='')  

# Only keep the variables used in training the NN
grid_env_info = grid_env_info[var4nn]
grid_env_info["original_lon"] = original_lons_grid
grid_env_info["original_lat"] = original_lats_grid


grid_env_info_US = grid_env_info.copy()
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

# Define grid profile IDs based on the grid_env_info_num
grid_profile_ids = np.arange(grid_env_info_num)
print("min profile ID in grid env info: ", np.min(grid_profile_ids))
print("max profile ID in grid env info: ", np.max(grid_profile_ids))

# create dummy z since it is not used in the prediction
predict_data_z = np.ones((grid_env_info_num))*np.nan
predict_data_c = np.stack([grid_env_info_US["original_lon"], grid_env_info_US["original_lat"]], axis=1)

print("Shape of predict data x", predict_data_x.shape)
print("Shape of predict data z", predict_data_z.shape)
print("Shape of grid env info US", grid_env_info_US.shape)
print(datetime.now(), '------------grid env info prepared------------')


# Remove grids with missing features or forcing variables
nan_loc = np.sum(predict_data_x[:, 0:len(var4nn), 0, 0], axis=1) + \
            np.sum(model_force_pred_input_vector_cwd, axis=1) + \
            np.sum(model_force_pred_input_vector_litter1, axis=1) + \
            np.sum(model_force_pred_input_vector_litter2, axis=1) + \
            np.sum(model_force_pred_input_vector_litter3, axis=1) + \
            np.sum(model_force_pred_altmax_lastyear_profile, axis=1) + \
            np.sum(model_force_pred_altmax_current_profile, axis=1) + \
            np.sum(model_force_pred_nbedrock, axis=1) + \
            np.sum(model_force_pred_xio, axis=(1, 2)) + \
            np.sum(model_force_pred_xin, axis=(1, 2)) + \
            np.sum(model_force_pred_sand_vector, axis=(1, 2)) + \
            np.sum(model_force_pred_soil_temp_profile, axis=(1, 2)) + \
            np.sum(model_force_pred_soil_water_profile, axis=(1, 2))
nan_mask = ~np.isnan(nan_loc)
valid_grid_loc = np.where(nan_mask)[0]
# Filter the grid env info and forcing data based on valid grid locations
predict_data_x = predict_data_x[valid_grid_loc, :, :, :]
predict_data_z = predict_data_z[valid_grid_loc]
predict_data_c = predict_data_c[valid_grid_loc, :]
grid_profile_ids = grid_profile_ids[valid_grid_loc]

# Convert predict data to tensors
predict_data_x = torch.tensor(predict_data_x, dtype=torch.float32)
predict_data_y = torch.tensor(np.ones((len(grid_profile_ids), 1)), dtype=torch.float32)  # Dummy y for predict dataset
predict_data_z = torch.tensor(predict_data_z, dtype=torch.float32)
predict_data_c = torch.tensor(predict_data_c, dtype=torch.float32)
grid_PRODA_para = torch.tensor(np.ones((len(grid_profile_ids), 1)), dtype=torch.float32)
predict_profile_id = torch.tensor(grid_profile_ids, dtype=torch.int64)

# Store the mean values, std, and the range of the grid_env_info
grid_env_info_mean = torch.mean(predict_data_x[:, 0:len(var4nn), 0, 0], dim=0)
grid_env_info_std = torch.std(predict_data_x[:, 0:len(var4nn), 0, 0], dim=0)
grid_env_info_max = torch.max(predict_data_x[:, 0:len(var4nn), 0, 0], dim=0).values
grid_env_info_min = torch.min(predict_data_x[:, 0:len(var4nn), 0, 0], dim=0).values

print("Mean of grid_env_info features: ", grid_env_info_mean)
print("Std of grid_env_info features: ", grid_env_info_std)
print("Max of grid_env_info features: ", grid_env_info_max)
print("Min of grid_env_info features: ", grid_env_info_min)

# Save the mean and std of the grid_env_info features
np.savetxt(data_dir_output + '/grid_env_info_mean.txt', grid_env_info_mean.numpy(), delimiter=',', header=','.join(var4nn), comments='')
np.savetxt(data_dir_output + '/grid_env_info_std.txt', grid_env_info_std.numpy(), delimiter=',', header=','.join(var4nn), comments='')
np.savetxt(data_dir_output + '/grid_env_info_max_norm.txt', grid_env_info_max.numpy(), delimiter=',', header=','.join(var4nn), comments='')
np.savetxt(data_dir_output + '/grid_env_info_min_norm.txt', grid_env_info_min.numpy(), delimiter=',', header=','.join(var4nn), comments='')

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


# Start training
def worker(rank, world_size, job_id, port):
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
            
    # global model
    if args.bias_only_epochs >= 1 and args.whether_resume == 0:
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
                                                            para_index, current_data_x, current_data_y)
        
    if args.loss_weighting not in ["manual", "two_stage", "relobralo"]:
        raise ValueError("You selected an advanced loss_weighting method that depends on the LibMTL library. This is not implemented yet.")
    else:
        # Create model
        model = model_class(args, **model_kwargs).to(device)   

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

    # Loss function
    fun_loss = binns_loss

    train_dataset = MergeDataset(current_data_x.to(device), current_data_y.to(device), current_data_z.to(device), current_data_c.to(device), current_data_profile_id.to(device), current_PRODA_para.to(device))
    predict_dataset = MergeDataset(predict_data_x.to(device), predict_data_y.to(device), predict_data_z.to(device), predict_data_c.to(device), predict_profile_id.to(device), grid_PRODA_para.to(device))

    # Use DistributedSampler for distributed training
    if args.use_ddp == 1:
        train_sampler = DistributedSampler(train_dataset)
        predict_sampler = DistributedSampler(predict_dataset)
    else:
        train_sampler, predict_sampler = None, None

    # Data loaders with DistributedSampler
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)  #, num_workers=4, persistent_workers=True)
    predict_sampler = DataLoader(predict_dataset, batch_size=args.batch_size, sampler=predict_sampler)  # , num_workers=4, persistent_workers=True)

    # Load the checkpoint for assigning the model parameters
    new_checkpoint = torch.load(model_dir_input + job_id + '/opt_nn_' + job_id + '.pt', map_location=device, weights_only=False)
    model.load_state_dict(new_checkpoint['model_state_dict'])

    dist.barrier()

    # record start time
    start_time = time.time()

    # Get the max number of batches for all processes
    local_len = torch.tensor([len(predict_sampler)], device='cpu')
    dist.all_reduce(local_len, op=dist.ReduceOp.MAX)
    max_batches = local_len.item()

    # Initiate tensor to store the predictions
    predict_baseline_soc_local = torch.full((max_batches, args.batch_size, 20), torch.nan, device=device)
    predict_baseline_POM_local = torch.full((max_batches, args.batch_size, 20), torch.nan, device=device)
    predict_baseline_MAOM_local = torch.full((max_batches, args.batch_size, 20), torch.nan, device=device)
    predict_baseline_DOC_local = torch.full((max_batches, args.batch_size, 20), torch.nan, device=device)
    predict_baseline_MIC_local = torch.full((max_batches, args.batch_size, 20), torch.nan, device=device)

    predict_baseline_profile_id_local = torch.full((max_batches, args.batch_size, 1), torch.nan, device=device)
    predict_baseline_para_local = torch.full((max_batches, args.batch_size, len(para_names)), torch.nan, device=device)

    predict_baseline_carbon_input_local = torch.full((max_batches, args.batch_size, 1), torch.nan, device=device)
    predict_baseline_cpool_steady_state_local = torch.full((max_batches, args.batch_size, 140), torch.nan, device=device)
    predict_baseline_cpool_layer_local = torch.full((max_batches, args.batch_size, 20), torch.nan, device=device)
    predict_baseline_soc_layer_local = torch.full((max_batches, args.batch_size, 20), torch.nan, device=device)
    predict_baseline_total_res_time_local = torch.full((max_batches, args.batch_size, 20), torch.nan, device=device)
    predict_baseline_total_res_time_base_local = torch.full((max_batches, args.batch_size, 20), torch.nan, device=device)
    predict_baseline_res_time_base_pools_local = torch.full((max_batches, args.batch_size, 140), torch.nan, device=device)
    predict_baseline_t_scalar_local = torch.full((max_batches, args.batch_size, 20), torch.nan, device=device)
    predict_baseline_bulk_A_doc_local = torch.full((max_batches, args.batch_size, 1), torch.nan, device=device)
    predict_baseline_bulk_E_mic_local = torch.full((max_batches, args.batch_size, 1), torch.nan, device=device)
    predict_baseline_bulk_A_mic_local = torch.full((max_batches, args.batch_size, 1), torch.nan, device=device)
    predict_baseline_bulk_A_POM_local = torch.full((max_batches, args.batch_size, 1), torch.nan, device=device)
    predict_baseline_bulk_A_MAOM_local = torch.full((max_batches, args.batch_size, 1), torch.nan, device=device)
    predict_baseline_w_scalar_local = torch.full((max_batches, args.batch_size, 20), torch.nan, device=device)
    predict_baseline_bulk_K_local = torch.full((max_batches, args.batch_size, 1), torch.nan, device=device)
    predict_baseline_bulk_V_local = torch.full((max_batches, args.batch_size, 1), torch.nan, device=device)
    predict_baseline_bulk_xi_local = torch.full((max_batches, args.batch_size, 1), torch.nan, device=device)
    predict_baseline_bulk_I_local = torch.full((max_batches, args.batch_size, 1), torch.nan, device=device)
    predict_baseline_litter_fraction_local = torch.full((max_batches, args.batch_size, 1), torch.nan, device=device)

    grid_coord_local = torch.full((max_batches, args.batch_size, 2), torch.nan, device=device)

    ibatch = 0
    # First calculate the baseline parameters and the bulk processes
    for batch_info in predict_sampler:
        batch_x, batch_y, batch_z, batch_c, batch_profile_id, batch_proda_para = batch_info
        # Move data to the correct device
        batch_x = batch_x.to(device)
        batch_z = batch_z.to(device)
        batch_c = batch_c.to(device)
        batch_profile_id = batch_profile_id.to(device)
        batch_proda_para = batch_proda_para.to(device)
        # Forward pass to get the predictions
        model.eval()
        with torch.no_grad():
            baseline_soc_batch, baseline_params = model(batch_x, batch_z, batch_c, whether_predict = 1, PRODA_para = None)
        
        # Store the predictions
        predict_baseline_soc_local[ibatch, :batch_x.shape[0], :] = baseline_soc_batch[:, 0, :]
        predict_baseline_POM_local[ibatch, :batch_x.shape[0], :] = baseline_soc_batch[:, 1, :]
        predict_baseline_MAOM_local[ibatch, :batch_x.shape[0], :] = baseline_soc_batch[:, 2, :]
        predict_baseline_DOC_local[ibatch, :batch_x.shape[0], :] = baseline_soc_batch[:, 3, :]
        predict_baseline_MIC_local[ibatch, :batch_x.shape[0], :] = baseline_soc_batch[:, 4, :]

        predict_baseline_profile_id_local[ibatch, :batch_x.shape[0], 0] = batch_profile_id
        predict_baseline_para_local[ibatch, :batch_x.shape[0], :] = baseline_params
        grid_coord_local[ibatch, :batch_x.shape[0], 0] = batch_c[:, 0]  # Longitude
        grid_coord_local[ibatch, :batch_x.shape[0], 1] = batch_c[:, 1]  # Latitude

        # Bulk simulation
        baseline_carbon_input_temp, baseline_cpool_steady_state_temp, baseline_cpools_layer_temp, \
            baseline_soc_layer_temp, baseline_total_res_time_temp, baseline_total_res_time_base_temp, \
            baseline_res_time_base_pools_temp, baseline_t_scaler_temp, baseline_bulk_A_doc_temp, \
            baseline_bulk_E_mic_temp, baseline_bulk_A_mic_temp, baseline_bulk_A_POM_temp, \
            baseline_bulk_A_MAOM_temp, baseline_w_scaler_temp, baseline_bulk_K_temp, \
            baseline_bulk_V_temp, baseline_bulk_xi_temp, baseline_bulk_I_temp, \
            baseline_litter_fraction_temp = fun_bulk_simu(baseline_params, batch_x)
        
        # Store the bulk simulation results
        predict_baseline_carbon_input_local[ibatch, :batch_x.shape[0], :] = baseline_carbon_input_temp
        predict_baseline_cpool_steady_state_local[ibatch, :batch_x.shape[0], :] = baseline_cpool_steady_state_temp
        predict_baseline_cpool_layer_local[ibatch, :batch_x.shape[0], :] = baseline_cpools_layer_temp
        predict_baseline_soc_layer_local[ibatch, :batch_x.shape[0], :] = baseline_soc_layer_temp
        predict_baseline_total_res_time_local[ibatch, :batch_x.shape[0], :] = baseline_total_res_time_temp
        predict_baseline_total_res_time_base_local[ibatch, :batch_x.shape[0], :] = baseline_total_res_time_base_temp
        predict_baseline_res_time_base_pools_local[ibatch, :batch_x.shape[0], :] = baseline_res_time_base_pools_temp
        predict_baseline_t_scalar_local[ibatch, :batch_x.shape[0], :] = baseline_t_scaler_temp
        predict_baseline_bulk_A_doc_local[ibatch, :batch_x.shape[0], :] = baseline_bulk_A_doc_temp
        predict_baseline_bulk_E_mic_local[ibatch, :batch_x.shape[0], :] = baseline_bulk_E_mic_temp
        predict_baseline_bulk_A_mic_local[ibatch, :batch_x.shape[0], :] = baseline_bulk_A_mic_temp
        predict_baseline_bulk_A_POM_local[ibatch, :batch_x.shape[0], :] = baseline_bulk_A_POM_temp
        predict_baseline_bulk_A_MAOM_local[ibatch, :batch_x.shape[0], :] = baseline_bulk_A_MAOM_temp
        predict_baseline_w_scalar_local[ibatch, :batch_x.shape[0], :] = baseline_w_scaler_temp
        predict_baseline_bulk_K_local[ibatch, :batch_x.shape[0], :] = baseline_bulk_K_temp
        predict_baseline_bulk_V_local[ibatch, :batch_x.shape[0], :] = baseline_bulk_V_temp
        predict_baseline_bulk_xi_local[ibatch, :batch_x.shape[0], :] = baseline_bulk_xi_temp
        predict_baseline_bulk_I_local[ibatch, :batch_x.shape[0], :] = baseline_bulk_I_temp
        predict_baseline_litter_fraction_local[ibatch, :batch_x.shape[0], :] = baseline_litter_fraction_temp
        
        ibatch += 1

    # End of the loop over batches
    
    # Synchronize all processes
    # Reshape the local tensors to gather them correctly (max_batches, batch_size, ...) to (max_batches * batch_size, ...)
    predict_baseline_soc_local = predict_baseline_soc_local.view(-1, 20)  # (max_batches * batch_size, 20)
    predict_baseline_POM_local = predict_baseline_POM_local.view(-1, 20)  # (max_batches * batch_size, 20)
    predict_baseline_MAOM_local = predict_baseline_MAOM_local.view(-1, 20)  # (max_batches * batch_size, 20)
    predict_baseline_DOC_local = predict_baseline_DOC_local.view(-1, 20)  # (max_batches * batch_size, 20)
    predict_baseline_MIC_local = predict_baseline_MIC_local.view(-1, 20)  # (max_batches * batch_size, 20)

    predict_baseline_profile_id_local = predict_baseline_profile_id_local.view(-1, 1)  # (max_batches * batch_size, 1)
    predict_baseline_para_local = predict_baseline_para_local.view(-1, len(para_names))  # (max_batches * batch_size, num_params)
    grid_coord_local = grid_coord_local.view(-1, 2)  # (max_batches * batch_size, 2)

    predict_baseline_carbon_input_local = predict_baseline_carbon_input_local.view(-1, 1)  # (max_batches * batch_size, 1)
    predict_baseline_cpool_steady_state_local = predict_baseline_cpool_steady_state_local.view(-1, 140)  # (max_batches * batch_size, 140)
    predict_baseline_cpool_layer_local = predict_baseline_cpool_layer_local.view(-1, 20)  # (max_batches * batch_size, 20)
    predict_baseline_soc_layer_local = predict_baseline_soc_layer_local.view(-1, 20)  # (max_batches * batch_size, 20)
    predict_baseline_total_res_time_local = predict_baseline_total_res_time_local.view(-1, 20)  # (max_batches * batch_size, 20)
    predict_baseline_total_res_time_base_local = predict_baseline_total_res_time_base_local.view(-1, 20)  # (max_batches * batch_size, 20)
    predict_baseline_res_time_base_pools_local = predict_baseline_res_time_base_pools_local.view(-1, 140)  # (max_batches * batch_size, 140)
    predict_baseline_t_scalar_local = predict_baseline_t_scalar_local.view(-1, 20)  # (max_batches * batch_size, 20)
    predict_baseline_bulk_A_doc_local = predict_baseline_bulk_A_doc_local.view(-1, 1)  # (max_batches * batch_size, 1)
    predict_baseline_bulk_E_mic_local = predict_baseline_bulk_E_mic_local.view(-1, 1)  # (max_batches * batch_size, 1)
    predict_baseline_bulk_A_mic_local = predict_baseline_bulk_A_mic_local.view(-1, 1)  # (max_batches * batch_size, 1)
    predict_baseline_bulk_A_POM_local = predict_baseline_bulk_A_POM_local.view(-1, 1)  # (max_batches * batch_size, 1)
    predict_baseline_bulk_A_MAOM_local = predict_baseline_bulk_A_MAOM_local.view(-1, 1)  # (max_batches * batch_size, 1)
    predict_baseline_w_scalar_local = predict_baseline_w_scalar_local.view(-1, 20)  # (max_batches * batch_size, 20)
    predict_baseline_bulk_K_local = predict_baseline_bulk_K_local.view(-1, 1)  # (max_batches * batch_size, 1)
    predict_baseline_bulk_V_local = predict_baseline_bulk_V_local.view(-1, 1)  # (max_batches * batch_size, 1)
    predict_baseline_bulk_xi_local = predict_baseline_bulk_xi_local.view(-1, 1)  # (max_batches * batch_size, 1)
    predict_baseline_bulk_I_local = predict_baseline_bulk_I_local.view(-1, 1)  # (max_batches * batch_size, 1)
    predict_baseline_litter_fraction_local = predict_baseline_litter_fraction_local.view(-1, 1)  # (max_batches * batch_size, 1)

    # Initialize lists to gather tensors from all processes
    predict_baseline_soc_all = [torch.full_like(predict_baseline_soc_local, torch.nan) for _ in range(dist.get_world_size())]
    predict_baseline_POM_all = [torch.full_like(predict_baseline_POM_local, torch.nan) for _ in range(dist.get_world_size())]
    predict_baseline_MAOM_all = [torch.full_like(predict_baseline_MAOM_local, torch.nan) for _ in range(dist.get_world_size())]
    predict_baseline_DOC_all = [torch.full_like(predict_baseline_DOC_local, torch.nan) for _ in range(dist.get_world_size())]
    predict_baseline_MIC_all = [torch.full_like(predict_baseline_MIC_local, torch.nan) for _ in range(dist.get_world_size())]

    predict_baseline_profile_id_all = [torch.full_like(predict_baseline_profile_id_local, torch.nan) for _ in range(dist.get_world_size())]
    predict_baseline_para_all = [torch.full_like(predict_baseline_para_local, torch.nan) for _ in range(dist.get_world_size())]
    grid_coord_all = [torch.full_like(grid_coord_local, torch.nan) for _ in range(dist.get_world_size())]

    predict_baseline_carbon_input_all = [torch.full_like(predict_baseline_carbon_input_local, torch.nan) for _ in range(dist.get_world_size())]
    predict_baseline_cpool_steady_state_all = [torch.full_like(predict_baseline_cpool_steady_state_local, torch.nan) for _ in range(dist.get_world_size())]
    predict_baseline_cpool_layer_all = [torch.full_like(predict_baseline_cpool_layer_local, torch.nan) for _ in range(dist.get_world_size())]
    predict_baseline_soc_layer_all = [torch.full_like(predict_baseline_soc_layer_local, torch.nan) for _ in range(dist.get_world_size())]
    predict_baseline_total_res_time_all = [torch.full_like(predict_baseline_total_res_time_local, torch.nan) for _ in range(dist.get_world_size())]
    predict_baseline_total_res_time_base_all = [torch.full_like(predict_baseline_total_res_time_base_local, torch.nan) for _ in range(dist.get_world_size())]
    predict_baseline_res_time_base_pools_all = [torch.full_like(predict_baseline_res_time_base_pools_local, torch.nan) for _ in range(dist.get_world_size())]
    predict_baseline_t_scalar_all = [torch.full_like(predict_baseline_t_scalar_local, torch.nan) for _ in range(dist.get_world_size())]
    predict_baseline_bulk_A_doc_all = [torch.full_like(predict_baseline_bulk_A_doc_local, torch.nan) for _ in range(dist.get_world_size())]
    predict_baseline_bulk_E_mic_all = [torch.full_like(predict_baseline_bulk_E_mic_local, torch.nan) for _ in range(dist.get_world_size())]
    predict_baseline_bulk_A_mic_all = [torch.full_like(predict_baseline_bulk_A_mic_local, torch.nan) for _ in range(dist.get_world_size())]
    predict_baseline_bulk_A_POM_all = [torch.full_like(predict_baseline_bulk_A_POM_local, torch.nan) for _ in range(dist.get_world_size())]
    predict_baseline_bulk_A_MAOM_all = [torch.full_like(predict_baseline_bulk_A_MAOM_local, torch.nan) for _ in range(dist.get_world_size())]
    predict_baseline_w_scalar_all = [torch.full_like(predict_baseline_w_scalar_local, torch.nan) for _ in range(dist.get_world_size())]
    predict_baseline_bulk_K_all = [torch.full_like(predict_baseline_bulk_K_local, torch.nan) for _ in range(dist.get_world_size())]
    predict_baseline_bulk_V_all = [torch.full_like(predict_baseline_bulk_V_local, torch.nan) for _ in range(dist.get_world_size())]
    predict_baseline_bulk_xi_all = [torch.full_like(predict_baseline_bulk_xi_local, torch.nan) for _ in range(dist.get_world_size())]
    predict_baseline_bulk_I_all = [torch.full_like(predict_baseline_bulk_I_local, torch.nan) for _ in range(dist.get_world_size())]
    predict_baseline_litter_fraction_all = [torch.full_like(predict_baseline_litter_fraction_local, torch.nan) for _ in range(dist.get_world_size())]

    # Gather the tensors from all processes
    dist.all_gather(predict_baseline_soc_all, predict_baseline_soc_local)
    dist.all_gather(predict_baseline_POM_all, predict_baseline_POM_local)
    dist.all_gather(predict_baseline_MAOM_all, predict_baseline_MAOM_local)
    dist.all_gather(predict_baseline_DOC_all, predict_baseline_DOC_local)
    dist.all_gather(predict_baseline_MIC_all, predict_baseline_MIC_local)

    dist.all_gather(predict_baseline_profile_id_all, predict_baseline_profile_id_local)
    dist.all_gather(predict_baseline_para_all, predict_baseline_para_local)
    dist.all_gather(grid_coord_all, grid_coord_local)

    dist.all_gather(predict_baseline_carbon_input_all, predict_baseline_carbon_input_local)
    dist.all_gather(predict_baseline_cpool_steady_state_all, predict_baseline_cpool_steady_state_local)
    dist.all_gather(predict_baseline_cpool_layer_all, predict_baseline_cpool_layer_local)
    dist.all_gather(predict_baseline_soc_layer_all, predict_baseline_soc_layer_local)
    dist.all_gather(predict_baseline_total_res_time_all, predict_baseline_total_res_time_local)
    dist.all_gather(predict_baseline_total_res_time_base_all, predict_baseline_total_res_time_base_local)
    dist.all_gather(predict_baseline_res_time_base_pools_all, predict_baseline_res_time_base_pools_local)
    dist.all_gather(predict_baseline_t_scalar_all, predict_baseline_t_scalar_local)
    dist.all_gather(predict_baseline_bulk_A_doc_all, predict_baseline_bulk_A_doc_local)
    dist.all_gather(predict_baseline_bulk_E_mic_all, predict_baseline_bulk_E_mic_local)
    dist.all_gather(predict_baseline_bulk_A_mic_all, predict_baseline_bulk_A_mic_local)
    dist.all_gather(predict_baseline_bulk_A_POM_all, predict_baseline_bulk_A_POM_local)
    dist.all_gather(predict_baseline_bulk_A_MAOM_all, predict_baseline_bulk_A_MAOM_local)
    dist.all_gather(predict_baseline_w_scalar_all, predict_baseline_w_scalar_local)
    dist.all_gather(predict_baseline_bulk_K_all, predict_baseline_bulk_K_local)
    dist.all_gather(predict_baseline_bulk_V_all, predict_baseline_bulk_V_local)
    dist.all_gather(predict_baseline_bulk_xi_all, predict_baseline_bulk_xi_local)
    dist.all_gather(predict_baseline_bulk_I_all, predict_baseline_bulk_I_local)
    dist.all_gather(predict_baseline_litter_fraction_all, predict_baseline_litter_fraction_local)
    # End of gathering

    # Concatenate the gathered tensors along the first dimension
    predict_baseline_soc_all = torch.cat(predict_baseline_soc_all, dim=0)
    predict_baseline_POM_all = torch.cat(predict_baseline_POM_all, dim=0)
    predict_baseline_MAOM_all = torch.cat(predict_baseline_MAOM_all, dim=0)
    predict_baseline_DOC_all = torch.cat(predict_baseline_DOC_all, dim=0)
    predict_baseline_MIC_all = torch.cat(predict_baseline_MIC_all, dim=0)

    predict_baseline_profile_id_all = torch.cat(predict_baseline_profile_id_all, dim=0)
    predict_baseline_para_all = torch.cat(predict_baseline_para_all, dim=0)
    grid_coord_all = torch.cat(grid_coord_all, dim=0)

    predict_baseline_carbon_input_all = torch.cat(predict_baseline_carbon_input_all, dim=0)
    predict_baseline_cpool_steady_state_all = torch.cat(predict_baseline_cpool_steady_state_all, dim=0)
    predict_baseline_cpool_layer_all = torch.cat(predict_baseline_cpool_layer_all, dim=0)
    predict_baseline_soc_layer_all = torch.cat(predict_baseline_soc_layer_all, dim=0)
    predict_baseline_total_res_time_all = torch.cat(predict_baseline_total_res_time_all, dim=0)
    predict_baseline_total_res_time_base_all = torch.cat(predict_baseline_total_res_time_base_all, dim=0)
    predict_baseline_res_time_base_pools_all = torch.cat(predict_baseline_res_time_base_pools_all, dim=0)
    predict_baseline_t_scalar_all = torch.cat(predict_baseline_t_scalar_all, dim=0)
    predict_baseline_bulk_A_doc_all = torch.cat(predict_baseline_bulk_A_doc_all, dim=0)
    predict_baseline_bulk_E_mic_all = torch.cat(predict_baseline_bulk_E_mic_all, dim=0)
    predict_baseline_bulk_A_mic_all = torch.cat(predict_baseline_bulk_A_mic_all, dim=0)
    predict_baseline_bulk_A_POM_all = torch.cat(predict_baseline_bulk_A_POM_all, dim=0)
    predict_baseline_bulk_A_MAOM_all = torch.cat(predict_baseline_bulk_A_MAOM_all, dim=0)
    predict_baseline_w_scalar_all = torch.cat(predict_baseline_w_scalar_all, dim=0)
    predict_baseline_bulk_K_all = torch.cat(predict_baseline_bulk_K_all, dim=0)
    predict_baseline_bulk_V_all = torch.cat(predict_baseline_bulk_V_all, dim=0)
    predict_baseline_bulk_xi_all = torch.cat(predict_baseline_bulk_xi_all, dim=0)
    predict_baseline_bulk_I_all = torch.cat(predict_baseline_bulk_I_all, dim=0)
    predict_baseline_litter_fraction_all = torch.cat(predict_baseline_litter_fraction_all, dim=0)

    # Assign the gathered tensors to the local variables
    predict_baseline_soc = torch.full((grid_env_info_num, 20), torch.nan, device=device)
    predict_baseline_POM = torch.full((grid_env_info_num, 20), torch.nan, device=device)
    predict_baseline_MAOM = torch.full((grid_env_info_num, 20), torch.nan, device=device)
    predict_baseline_DOC = torch.full((grid_env_info_num, 20), torch.nan, device=device)
    predict_baseline_MIC = torch.full((grid_env_info_num, 20), torch.nan, device=device)

    predict_baseline_profile_id = torch.full((grid_env_info_num, 1), torch.nan, device=device)
    predict_baseline_para = torch.full((grid_env_info_num, len(para_names)), torch.nan, device=device)
    grid_coord = torch.full((grid_env_info_num, 2), torch.nan, device=device)

    predict_baseline_carbon_input = torch.full((grid_env_info_num, 1), torch.nan, device=device)
    predict_baseline_cpool_steady_state = torch.full((grid_env_info_num, 140), torch.nan, device=device)
    predict_baseline_cpool_layer = torch.full((grid_env_info_num, 20), torch.nan, device=device)
    predict_baseline_soc_layer = torch.full((grid_env_info_num, 20), torch.nan, device=device)
    predict_baseline_total_res_time = torch.full((grid_env_info_num, 20), torch.nan, device=device)
    predict_baseline_total_res_time_base = torch.full((grid_env_info_num, 20), torch.nan, device=device)
    predict_baseline_res_time_base_pools = torch.full((grid_env_info_num, 140), torch.nan, device=device)
    predict_baseline_t_scalar = torch.full((grid_env_info_num, 20), torch.nan, device=device)
    predict_baseline_bulk_A_doc = torch.full((grid_env_info_num, 1), torch.nan, device=device)
    predict_baseline_bulk_E_mic = torch.full((grid_env_info_num, 1), torch.nan, device=device)
    predict_baseline_bulk_A_mic = torch.full((grid_env_info_num, 1), torch.nan, device=device)
    predict_baseline_bulk_A_POM = torch.full((grid_env_info_num, 1), torch.nan, device=device)
    predict_baseline_bulk_A_MAOM = torch.full((grid_env_info_num, 1), torch.nan, device=device)
    predict_baseline_w_scalar = torch.full((grid_env_info_num, 20), torch.nan, device=device)
    predict_baseline_bulk_K = torch.full((grid_env_info_num, 1), torch.nan, device=device)
    predict_baseline_bulk_V = torch.full((grid_env_info_num, 1), torch.nan, device=device)
    predict_baseline_bulk_xi = torch.full((grid_env_info_num, 1), torch.nan, device=device)
    predict_baseline_bulk_I = torch.full((grid_env_info_num, 1), torch.nan, device=device)
    predict_baseline_litter_fraction = torch.full((grid_env_info_num, 1), torch.nan, device=device)

    # Assign the gathered tensors to the local variables based on the predict_baseline_profile_id_all
    # for id in predict_baseline_profile_id_all
    predict_baseline_profile_id_all_squueezed = predict_baseline_profile_id_all.squeeze()
    
    # Save the predictions to a file
    if rank == 0:
        for profile_id in predict_baseline_profile_id_all_squueezed:
            if torch.isnan(profile_id).any():
                continue
            profile_id_int = int(profile_id.item())
            idx = int(np.where(predict_baseline_profile_id_all_squueezed == profile_id_int)[0][0])
            predict_baseline_soc[profile_id_int, :] = predict_baseline_soc_all[idx, :]
            predict_baseline_POM[profile_id_int, :] = predict_baseline_POM_all[idx, :]
            predict_baseline_MAOM[profile_id_int, :] = predict_baseline_MAOM_all[idx, :]
            predict_baseline_DOC[profile_id_int, :] = predict_baseline_DOC_all[idx, :]
            predict_baseline_MIC[profile_id_int, :] = predict_baseline_MIC_all[idx, :]

            predict_baseline_profile_id[profile_id_int, 0] = profile_id_int
            predict_baseline_para[profile_id_int, :] = predict_baseline_para_all[idx, :]
            grid_coord[profile_id_int, 0] = grid_coord_all[idx, 0]  # Longitude
            grid_coord[profile_id_int, 1] = grid_coord_all[idx, 1]  # Latitude

            predict_baseline_carbon_input[profile_id_int, 0] = predict_baseline_carbon_input_all[idx, 0]
            predict_baseline_cpool_steady_state[profile_id_int, :] = predict_baseline_cpool_steady_state_all[idx, :]
            predict_baseline_cpool_layer[profile_id_int, :] = predict_baseline_cpool_layer_all[idx, :]
            predict_baseline_soc_layer[profile_id_int, :] = predict_baseline_soc_layer_all[idx, :]
            predict_baseline_total_res_time[profile_id_int, :] = predict_baseline_total_res_time_all[idx, :]
            predict_baseline_total_res_time_base[profile_id_int, :] = predict_baseline_total_res_time_base_all[idx, :]
            predict_baseline_res_time_base_pools[profile_id_int, :] = predict_baseline_res_time_base_pools_all[idx, :]
            predict_baseline_t_scalar[profile_id_int, :] = predict_baseline_t_scalar_all[idx, :]
            predict_baseline_bulk_A_doc[profile_id_int, 0] = predict_baseline_bulk_A_doc_all[idx, 0]
            predict_baseline_bulk_E_mic[profile_id_int, 0] = predict_baseline_bulk_E_mic_all[idx, 0]
            predict_baseline_bulk_A_mic[profile_id_int, 0] = predict_baseline_bulk_A_mic_all[idx, 0]
            predict_baseline_bulk_A_POM[profile_id_int, 0] = predict_baseline_bulk_A_POM_all[idx, 0]
            predict_baseline_bulk_A_MAOM[profile_id_int, 0] = predict_baseline_bulk_A_MAOM_all[idx, 0]
            predict_baseline_w_scalar[profile_id_int, :] = predict_baseline_w_scalar_all[idx, :]
            predict_baseline_bulk_K[profile_id_int, 0] = predict_baseline_bulk_K_all[idx, 0]
            predict_baseline_bulk_V[profile_id_int, 0] = predict_baseline_bulk_V_all[idx, 0]
            predict_baseline_bulk_xi[profile_id_int, 0] = predict_baseline_bulk_xi_all[idx, 0]
            predict_baseline_bulk_I[profile_id_int, 0] = predict_baseline_bulk_I_all[idx, 0]
            predict_baseline_litter_fraction[profile_id_int, 0] = predict_baseline_litter_fraction_all[idx, 0]

        # Create the output directory if it doesn't exist
        if not os.path.exists(data_dir_output + '/baseline/'):
            os.makedirs(data_dir_output + '/baseline/')
        # Save the predictions to files
        np.savetxt(data_dir_output + '/baseline/' + 'soc.txt', predict_baseline_soc.cpu().numpy(), delimiter=',')
        np.savetxt(data_dir_output + '/baseline/' + 'POM.txt', predict_baseline_POM.cpu().numpy(), delimiter=',')
        np.savetxt(data_dir_output + '/baseline/' + 'MAOM.txt', predict_baseline_MAOM.cpu().numpy(), delimiter=',')
        np.savetxt(data_dir_output + '/baseline/' + 'DOC.txt', predict_baseline_DOC.cpu().numpy(), delimiter=',')
        np.savetxt(data_dir_output + '/baseline/' + 'MIC.txt', predict_baseline_MIC.cpu().numpy(), delimiter=',')

        np.savetxt(data_dir_output + '/baseline/' + 'profile_id.txt', predict_baseline_profile_id.cpu().numpy(), delimiter=',')
        np.savetxt(data_dir_output + '/baseline/' + 'para.txt', predict_baseline_para.cpu().numpy(), delimiter=',')
        np.savetxt(data_dir_output + '/baseline/' + 'grid_coord.txt', grid_coord.cpu().numpy(), delimiter=',')

        np.savetxt(data_dir_output + '/baseline/' + 'carbon_input.txt', predict_baseline_carbon_input.cpu().numpy(), delimiter=',')
        np.savetxt(data_dir_output + '/baseline/' + 'cpool_steady_state.txt', predict_baseline_cpool_steady_state.cpu().numpy(), delimiter=',')
        np.savetxt(data_dir_output + '/baseline/' + 'cpool_layer.txt', predict_baseline_cpool_layer.cpu().numpy(), delimiter=',')
        np.savetxt(data_dir_output + '/baseline/' + 'soc_layer.txt', predict_baseline_soc_layer.cpu().numpy(), delimiter=',')
        np.savetxt(data_dir_output + '/baseline/' + 'total_res_time.txt', predict_baseline_total_res_time.cpu().numpy(), delimiter=',')
        np.savetxt(data_dir_output + '/baseline/' + 'total_res_time_base.txt', predict_baseline_total_res_time_base.cpu().numpy(), delimiter=',')
        np.savetxt(data_dir_output + '/baseline/' + 'res_time_base_pools.txt', predict_baseline_res_time_base_pools.cpu().numpy(), delimiter=',')
        np.savetxt(data_dir_output + '/baseline/' + 't_scalar.txt', predict_baseline_t_scalar.cpu().numpy(), delimiter=',')
        np.savetxt(data_dir_output + '/baseline/' + 'bulk_A_doc.txt', predict_baseline_bulk_A_doc.cpu().numpy(), delimiter=',')
        np.savetxt(data_dir_output + '/baseline/' + 'bulk_E_mic.txt', predict_baseline_bulk_E_mic.cpu().numpy(), delimiter=',')
        np.savetxt(data_dir_output + '/baseline/' + 'bulk_A_mic.txt', predict_baseline_bulk_A_mic.cpu().numpy(), delimiter=',')
        np.savetxt(data_dir_output + '/baseline/' + 'bulk_A_POM.txt', predict_baseline_bulk_A_POM.cpu().numpy(), delimiter=',')
        np.savetxt(data_dir_output + '/baseline/' + 'bulk_A_MAOM.txt', predict_baseline_bulk_A_MAOM.cpu().numpy(), delimiter=',')
        np.savetxt(data_dir_output + '/baseline/' + 'w_scalar.txt', predict_baseline_w_scalar.cpu().numpy(), delimiter=',')
        np.savetxt(data_dir_output + '/baseline/' + 'bulk_K.txt', predict_baseline_bulk_K.cpu().numpy(), delimiter=',')
        np.savetxt(data_dir_output + '/baseline/' + 'bulk_V.txt', predict_baseline_bulk_V.cpu().numpy(), delimiter=',')
        np.savetxt(data_dir_output + '/baseline/' + 'bulk_xi.txt', predict_baseline_bulk_xi.cpu().numpy(), delimiter=',')
        np.savetxt(data_dir_output + '/baseline/' + 'bulk_I.txt', predict_baseline_bulk_I.cpu().numpy(), delimiter=',')
        np.savetxt(data_dir_output + '/baseline/' + 'litter_fraction.txt', predict_baseline_litter_fraction.cpu().numpy(), delimiter=',')

        print("Baseline predictions saved successfully with time taken: {:.2f} seconds".format(time.time() - start_time))

    dist.barrier() 

    # For each var4nn, assign 0.5 to that variable, and then cahnge the variable by propotional_change
    # Define the propotional change from -50 to 50 percent
    # propotional_change = np.linspace(-0.5, 0.5, num=11)  # 11 values from -50% to 50%
    # If change between -2*std and 2*std
    propotional_change = np.linspace(-2, 2, num=9)  # 9 values from -2 to 2
    # If change between -1*std and 1*std
    # propotional_change = np.linspace(-1, 1, num=11)  # 11 values from -1 to 1


    # Save the propotional change to a file
    if rank == 0:
        np.savetxt(data_dir_output + '/propotional_change.txt', propotional_change, delimiter=',')
    dist.barrier()

    # Initialize a for loop to iterate over each variable in var4nn
    for var_idx, var_name in enumerate(var4nn):
        var_start_time = time.time()  # Record the start time for each variable

        # Create the output directory for the variable if it doesn't exist
        if rank == 0:
            if not os.path.exists(data_dir_output + f'/{var_name}/'):
                os.makedirs(data_dir_output + f'/{var_name}/')
        # Initialize a tensor to store the updated environment information
        grid_env_info_update = torch.full((len(propotional_change), 1), torch.nan, device=device)
        # Iterate over each propotional change
        for change in propotional_change:
            change_start_time = time.time()  # Record the start time for each propotional change
            change_idx = propotional_change.tolist().index(change)  # Get the index of the current change
            ibatch = 0  
            # Initialize a tensor to store the predictions for each propotional change
            predict_prop_change_soc_local = torch.full((max_batches, args.batch_size, 20), torch.nan, device=device)
            predict_prop_change_POM_local = torch.full((max_batches, args.batch_size, 20), torch.nan, device=device)
            predict_prop_change_MAOM_local = torch.full((max_batches, args.batch_size, 20), torch.nan, device=device)
            predict_prop_change_DOC_local = torch.full((max_batches, args.batch_size, 20), torch.nan, device=device)
            predict_prop_change_MIC_local = torch.full((max_batches, args.batch_size, 20), torch.nan, device=device)

            predict_prop_change_profile_id_local = torch.full((max_batches, args.batch_size, 1), torch.nan, device=device)
            predict_prop_change_para_local = torch.full((max_batches, args.batch_size, len(para_names)), torch.nan, device=device)

            predict_prop_change_carbon_input_local = torch.full((max_batches, args.batch_size, 1), torch.nan, device=device)
            predict_prop_change_cpool_steady_state_local = torch.full((max_batches, args.batch_size, 140), torch.nan, device=device)
            predict_prop_change_cpool_layer_local = torch.full((max_batches, args.batch_size, 20), torch.nan, device=device)
            predict_prop_change_soc_layer_local = torch.full((max_batches, args.batch_size, 20), torch.nan, device=device)
            predict_prop_change_total_res_time_local = torch.full((max_batches, args.batch_size, 20), torch.nan, device=device)
            predict_prop_change_total_res_time_base_local = torch.full((max_batches, args.batch_size, 20), torch.nan, device=device)
            predict_prop_change_res_time_base_pools_local = torch.full((max_batches, args.batch_size, 140), torch.nan, device=device)
            predict_prop_change_t_scalar_local = torch.full((max_batches, args.batch_size, 20), torch.nan, device=device)
            predict_prop_change_bulk_A_doc_local = torch.full((max_batches, args.batch_size, 1), torch.nan, device=device)
            predict_prop_change_bulk_E_mic_local = torch.full((max_batches, args.batch_size, 1), torch.nan, device=device)
            predict_prop_change_bulk_A_mic_local = torch.full((max_batches, args.batch_size, 1), torch.nan, device=device)
            predict_prop_change_bulk_A_POM_local = torch.full((max_batches, args.batch_size, 1), torch.nan, device=device)
            predict_prop_change_bulk_A_MAOM_local = torch.full((max_batches, args.batch_size, 1), torch.nan, device=device)
            predict_prop_change_w_scalar_local = torch.full((max_batches, args.batch_size, 20), torch.nan, device=device)
            predict_prop_change_bulk_K_local = torch.full((max_batches, args.batch_size, 1), torch.nan, device=device)
            predict_prop_change_bulk_V_local = torch.full((max_batches, args.batch_size, 1), torch.nan, device=device)
            predict_prop_change_bulk_xi_local = torch.full((max_batches, args.batch_size, 1), torch.nan, device=device)
            predict_prop_change_bulk_I_local = torch.full((max_batches, args.batch_size, 1), torch.nan, device=device)
            predict_prop_change_litter_fraction_local = torch.full((max_batches, args.batch_size, 1), torch.nan, device=device)     

            grid_coord_prop_change_local = torch.full((max_batches, args.batch_size, 2), torch.nan, device=device)       

            for batch_info in predict_sampler:
                batch_x, batch_y, batch_z, batch_c, batch_profile_id, batch_proda_para = batch_info
                # Move data to the correct device
                batch_x = batch_x.to(device)
                batch_z = batch_z.to(device)
                batch_c = batch_c.to(device)
                batch_profile_id = batch_profile_id.to(device)
                batch_proda_para = batch_proda_para.to(device)

                # # Assign 0.5 to the variable at var_idx
                # if grid_env_info_mean[var_idx]*1.5 < grid_env_info_max[var_idx] and grid_env_info_mean[var_idx]*0.5 > grid_env_info_min[var_idx]:
                #     env_info_update_value = grid_env_info_mean[var_idx]  # Set the variable to its mean value
                # else:
                #     env_info_update_value = (grid_env_info_max[var_idx] + grid_env_info_min[var_idx]) / 2  # Set the variable to its mid value
                
                env_info_update_value = grid_env_info_mean[var_idx]
                
                batch_x[:, var_idx, 0, 0] = env_info_update_value
                
                # Apply the propotional change to the current environment variable
                # batch_x[:, var_idx, 0, 0] *= (1 + change)              
                # grid_env_info_update[change_idx, 0] = env_info_update_value * (1 + change)  # Update the environment information with the propotional change
                
                # If use std as change
                batch_x[:, var_idx, 0, 0] += change * grid_env_info_std[var_idx]
                grid_env_info_update[change_idx, 0] = env_info_update_value + change * grid_env_info_std[var_idx]  # Update the environment information with the propotional change
                
                if grid_env_info_update[change_idx, 0] < 0:
                    grid_env_info_update[change_idx, 0] = 0
                    batch_x[:, var_idx, 0, 0] = 0 
                    if rank == 0:
                        print(f"Warning: {var_name} value below minimum, set to minimum: {grid_env_info_min[var_idx]} during propotional change {change}")
                elif grid_env_info_update[change_idx, 0] > 1:
                    grid_env_info_update[change_idx, 0] = 1
                    batch_x[:, var_idx, 0, 0] = 1 
                    if rank == 0:
                        print(f"Warning: {var_name} value above maximum, set to maximum: {grid_env_info_max[var_idx]} during propotional change {change}")

                # Forward pass to get the predictions with the propotional change applied
                model.eval()
                with torch.no_grad():
                    baseline_soc_batch, baseline_params = model(batch_x, batch_z, batch_c, whether_predict=1, PRODA_para=None)
                
                # Store the predictions
                predict_prop_change_soc_local[ibatch, :batch_x.shape[0], :] = baseline_soc_batch[:, 0, :]
                predict_prop_change_POM_local[ibatch, :batch_x.shape[0], :] = baseline_soc_batch[:, 1, :]
                predict_prop_change_MAOM_local[ibatch, :batch_x.shape[0], :] = baseline_soc_batch[:, 2, :]
                predict_prop_change_DOC_local[ibatch, :batch_x.shape[0], :] = baseline_soc_batch[:, 3, :]
                predict_prop_change_MIC_local[ibatch, :batch_x.shape[0], :] = baseline_soc_batch[:, 4, :]

                predict_prop_change_profile_id_local[ibatch, :batch_x.shape[0], 0] = batch_profile_id
                predict_prop_change_para_local[ibatch, :batch_x.shape[0], :] = baseline_params

                grid_coord_prop_change_local[ibatch, :batch_x.shape[0], 0] = batch_x[:, 0, 0, 0]  # Longitude
                grid_coord_prop_change_local[ibatch, :batch_x.shape[0], 1] = batch_x[:, 0, 0, 1]  # Latitude

                # Bulk simulation
                prop_change_carbon_input_temp, prop_change_cpool_steady_state_temp, prop_change_cpools_layer_temp, \
                    prop_change_soc_layer_temp, prop_change_total_res_time_temp, prop_change_total_res_time_base_temp, \
                    prop_change_res_time_base_pools_temp, prop_change_t_scaler_temp, prop_change_bulk_A_doc_temp, \
                    prop_change_bulk_E_mic_temp, prop_change_bulk_A_mic_temp, prop_change_bulk_A_POM_temp, \
                    prop_change_bulk_A_MAOM_temp, prop_change_w_scaler_temp, prop_change_bulk_K_temp, \
                    prop_change_bulk_V_temp, prop_change_bulk_xi_temp, prop_change_bulk_I_temp, \
                    prop_change_litter_fraction_temp = fun_bulk_simu(baseline_params, batch_x)
                
                # Store the bulk simulation results
                predict_prop_change_carbon_input_local[ibatch, :batch_x.shape[0], :] = prop_change_carbon_input_temp
                predict_prop_change_cpool_steady_state_local[ibatch, :batch_x.shape[0], :] = prop_change_cpool_steady_state_temp
                predict_prop_change_cpool_layer_local[ibatch, :batch_x.shape[0], :] = prop_change_cpools_layer_temp
                predict_prop_change_soc_layer_local[ibatch, :batch_x.shape[0], :] = prop_change_soc_layer_temp
                predict_prop_change_total_res_time_local[ibatch, :batch_x.shape[0], :] = prop_change_total_res_time_temp
                predict_prop_change_total_res_time_base_local[ibatch, :batch_x.shape[0], :] = prop_change_total_res_time_base_temp
                predict_prop_change_res_time_base_pools_local[ibatch, :batch_x.shape[0], :] = prop_change_res_time_base_pools_temp
                predict_prop_change_t_scalar_local[ibatch, :batch_x.shape[0], :] = prop_change_t_scaler_temp
                predict_prop_change_bulk_A_doc_local[ibatch, :batch_x.shape[0], :] = prop_change_bulk_A_doc_temp
                predict_prop_change_bulk_E_mic_local[ibatch, :batch_x.shape[0], :] = prop_change_bulk_E_mic_temp
                predict_prop_change_bulk_A_mic_local[ibatch, :batch_x.shape[0], :] = prop_change_bulk_A_mic_temp
                predict_prop_change_bulk_A_POM_local[ibatch, :batch_x.shape[0], :] = prop_change_bulk_A_POM_temp
                predict_prop_change_bulk_A_MAOM_local[ibatch, :batch_x.shape[0], :] = prop_change_bulk_A_MAOM_temp
                predict_prop_change_w_scalar_local[ibatch, :batch_x.shape[0], :] = prop_change_w_scaler_temp
                predict_prop_change_bulk_K_local[ibatch, :batch_x.shape[0], :] = prop_change_bulk_K_temp
                predict_prop_change_bulk_V_local[ibatch, :batch_x.shape[0], :] = prop_change_bulk_V_temp
                predict_prop_change_bulk_xi_local[ibatch, :batch_x.shape[0], :] = prop_change_bulk_xi_temp
                predict_prop_change_bulk_I_local[ibatch, :batch_x.shape[0], :] = prop_change_bulk_I_temp
                predict_prop_change_litter_fraction_local[ibatch, :batch_x.shape[0], :] = prop_change_litter_fraction_temp

                ibatch += 1 

                # End of batch loop
            
            # Synchronize all processes
            # Reshape the local tensors to gather them correctly (max_batches, batch_size, ...) to (max_batches * batch_size, ...)
            predict_prop_change_soc_local = predict_prop_change_soc_local.view(-1, 20)
            predict_prop_change_POM_local = predict_prop_change_POM_local.view(-1, 20)
            predict_prop_change_MAOM_local = predict_prop_change_MAOM_local.view(-1, 20)
            predict_prop_change_DOC_local = predict_prop_change_DOC_local.view(-1, 20)
            predict_prop_change_MIC_local = predict_prop_change_MIC_local.view(-1, 20)
            
            predict_prop_change_profile_id_local = predict_prop_change_profile_id_local.view(-1, 1)
            predict_prop_change_para_local = predict_prop_change_para_local.view(-1, len(para_names))
            grid_coord_prop_change_local = grid_coord_prop_change_local.view(-1, 2)

            predict_prop_change_carbon_input_local = predict_prop_change_carbon_input_local.view(-1, 1)
            predict_prop_change_cpool_steady_state_local = predict_prop_change_cpool_steady_state_local.view(-1, 140)
            predict_prop_change_cpool_layer_local = predict_prop_change_cpool_layer_local.view(-1, 20)
            predict_prop_change_soc_layer_local = predict_prop_change_soc_layer_local.view(-1, 20)
            predict_prop_change_total_res_time_local = predict_prop_change_total_res_time_local.view(-1, 20)
            predict_prop_change_total_res_time_base_local = predict_prop_change_total_res_time_base_local.view(-1, 20)
            predict_prop_change_res_time_base_pools_local = predict_prop_change_res_time_base_pools_local.view(-1, 140)
            predict_prop_change_t_scalar_local = predict_prop_change_t_scalar_local.view(-1, 20)
            predict_prop_change_bulk_A_doc_local = predict_prop_change_bulk_A_doc_local.view(-1, 1)
            predict_prop_change_bulk_E_mic_local = predict_prop_change_bulk_E_mic_local.view(-1, 1)
            predict_prop_change_bulk_A_mic_local = predict_prop_change_bulk_A_mic_local.view(-1, 1)
            predict_prop_change_bulk_A_POM_local = predict_prop_change_bulk_A_POM_local.view(-1, 1)
            predict_prop_change_bulk_A_MAOM_local = predict_prop_change_bulk_A_MAOM_local.view(-1, 1)
            predict_prop_change_w_scalar_local = predict_prop_change_w_scalar_local.view(-1, 20)
            predict_prop_change_bulk_K_local = predict_prop_change_bulk_K_local.view(-1, 1)
            predict_prop_change_bulk_V_local = predict_prop_change_bulk_V_local.view(-1, 1)
            predict_prop_change_bulk_xi_local = predict_prop_change_bulk_xi_local.view(-1, 1)
            predict_prop_change_bulk_I_local = predict_prop_change_bulk_I_local.view(-1, 1)
            predict_prop_change_litter_fraction_local = predict_prop_change_litter_fraction_local.view(-1, 1)

            # Initialize lists to gather tensors from all processes
            predict_prop_change_soc_all = [torch.full_like(predict_prop_change_soc_local, torch.nan) for _ in range(dist.get_world_size())]
            predict_prop_change_POM_all = [torch.full_like(predict_prop_change_POM_local, torch.nan) for _ in range(dist.get_world_size())]
            predict_prop_change_MAOM_all = [torch.full_like(predict_prop_change_MAOM_local, torch.nan) for _ in range(dist.get_world_size())]
            predict_prop_change_DOC_all = [torch.full_like(predict_prop_change_DOC_local, torch.nan) for _ in range(dist.get_world_size())]
            predict_prop_change_MIC_all = [torch.full_like(predict_prop_change_MIC_local, torch.nan) for _ in range(dist.get_world_size())]
            
            predict_prop_change_profile_id_all = [torch.full_like(predict_prop_change_profile_id_local, torch.nan) for _ in range(dist.get_world_size())]
            predict_prop_change_para_all = [torch.full_like(predict_prop_change_para_local, torch.nan) for _ in range(dist.get_world_size())]
            grid_coord_all = [torch.full_like(grid_coord_prop_change_local, torch.nan) for _ in range(dist.get_world_size())]

            predict_prop_change_carbon_input_all = [torch.full_like(predict_prop_change_carbon_input_local, torch.nan) for _ in range(dist.get_world_size())]
            predict_prop_change_cpool_steady_state_all = [torch.full_like(predict_prop_change_cpool_steady_state_local, torch.nan) for _ in range(dist.get_world_size())]
            predict_prop_change_cpool_layer_all = [torch.full_like(predict_prop_change_cpool_layer_local, torch.nan) for _ in range(dist.get_world_size())]
            predict_prop_change_soc_layer_all = [torch.full_like(predict_prop_change_soc_layer_local, torch.nan) for _ in range(dist.get_world_size())]
            predict_prop_change_total_res_time_all = [torch.full_like(predict_prop_change_total_res_time_local, torch.nan) for _ in range(dist.get_world_size())]
            predict_prop_change_total_res_time_base_all = [torch.full_like(predict_prop_change_total_res_time_base_local, torch.nan) for _ in range(dist.get_world_size())]
            predict_prop_change_res_time_base_pools_all = [torch.full_like(predict_prop_change_res_time_base_pools_local, torch.nan) for _ in range(dist.get_world_size())]
            predict_prop_change_t_scalar_all = [torch.full_like(predict_prop_change_t_scalar_local, torch.nan) for _ in range(dist.get_world_size())]
            predict_prop_change_bulk_A_doc_all = [torch.full_like(predict_prop_change_bulk_A_doc_local, torch.nan) for _ in range(dist.get_world_size())]
            predict_prop_change_bulk_E_mic_all = [torch.full_like(predict_prop_change_bulk_E_mic_local, torch.nan) for _ in range(dist.get_world_size())]
            predict_prop_change_bulk_A_mic_all = [torch.full_like(predict_prop_change_bulk_A_mic_local, torch.nan) for _ in range(dist.get_world_size())]
            predict_prop_change_bulk_A_POM_all = [torch.full_like(predict_prop_change_bulk_A_POM_local, torch.nan) for _ in range(dist.get_world_size())]
            predict_prop_change_bulk_A_MAOM_all = [torch.full_like(predict_prop_change_bulk_A_MAOM_local, torch.nan) for _ in range(dist.get_world_size())]
            predict_prop_change_w_scalar_all = [torch.full_like(predict_prop_change_w_scalar_local, torch.nan) for _ in range(dist.get_world_size())]
            predict_prop_change_bulk_K_all = [torch.full_like(predict_prop_change_bulk_K_local, torch.nan) for _ in range(dist.get_world_size())]
            predict_prop_change_bulk_V_all = [torch.full_like(predict_prop_change_bulk_V_local, torch.nan) for _ in range(dist.get_world_size())]
            predict_prop_change_bulk_xi_all = [torch.full_like(predict_prop_change_bulk_xi_local, torch.nan) for _ in range(dist.get_world_size())]
            predict_prop_change_bulk_I_all = [torch.full_like(predict_prop_change_bulk_I_local, torch.nan) for _ in range(dist.get_world_size())]
            predict_prop_change_litter_fraction_all = [torch.full_like(predict_prop_change_litter_fraction_local, torch.nan) for _ in range(dist.get_world_size())]

            # Gather the tensors from all processes
            dist.all_gather(predict_prop_change_soc_all, predict_prop_change_soc_local)
            dist.all_gather(predict_prop_change_POM_all, predict_prop_change_POM_local)
            dist.all_gather(predict_prop_change_MAOM_all, predict_prop_change_MAOM_local)
            dist.all_gather(predict_prop_change_DOC_all, predict_prop_change_DOC_local)
            dist.all_gather(predict_prop_change_MIC_all, predict_prop_change_MIC_local)

            dist.all_gather(predict_prop_change_profile_id_all, predict_prop_change_profile_id_local)
            dist.all_gather(predict_prop_change_para_all, predict_prop_change_para_local)
            dist.all_gather(grid_coord_all, grid_coord_prop_change_local)

            dist.all_gather(predict_prop_change_carbon_input_all, predict_prop_change_carbon_input_local)
            dist.all_gather(predict_prop_change_cpool_steady_state_all, predict_prop_change_cpool_steady_state_local)
            dist.all_gather(predict_prop_change_cpool_layer_all, predict_prop_change_cpool_layer_local)
            dist.all_gather(predict_prop_change_soc_layer_all, predict_prop_change_soc_layer_local)
            dist.all_gather(predict_prop_change_total_res_time_all, predict_prop_change_total_res_time_local)
            dist.all_gather(predict_prop_change_total_res_time_base_all, predict_prop_change_total_res_time_base_local)
            dist.all_gather(predict_prop_change_res_time_base_pools_all, predict_prop_change_res_time_base_pools_local)
            dist.all_gather(predict_prop_change_t_scalar_all, predict_prop_change_t_scalar_local)
            dist.all_gather(predict_prop_change_bulk_A_doc_all, predict_prop_change_bulk_A_doc_local)
            dist.all_gather(predict_prop_change_bulk_E_mic_all, predict_prop_change_bulk_E_mic_local)
            dist.all_gather(predict_prop_change_bulk_A_mic_all, predict_prop_change_bulk_A_mic_local)
            dist.all_gather(predict_prop_change_bulk_A_POM_all, predict_prop_change_bulk_A_POM_local)
            dist.all_gather(predict_prop_change_bulk_A_MAOM_all, predict_prop_change_bulk_A_MAOM_local)
            dist.all_gather(predict_prop_change_w_scalar_all, predict_prop_change_w_scalar_local)
            dist.all_gather(predict_prop_change_bulk_K_all, predict_prop_change_bulk_K_local)
            dist.all_gather(predict_prop_change_bulk_V_all, predict_prop_change_bulk_V_local)
            dist.all_gather(predict_prop_change_bulk_xi_all, predict_prop_change_bulk_xi_local)
            dist.all_gather(predict_prop_change_bulk_I_all, predict_prop_change_bulk_I_local)
            dist.all_gather(predict_prop_change_litter_fraction_all, predict_prop_change_litter_fraction_local)
            # End of gathering

            # Concatenate the gathered tensors along the first dimension
            predict_prop_change_soc_all = torch.cat(predict_prop_change_soc_all, dim=0)
            predict_prop_change_POM_all = torch.cat(predict_prop_change_POM_all, dim=0)
            predict_prop_change_MAOM_all = torch.cat(predict_prop_change_MAOM_all, dim=0)
            predict_prop_change_DOC_all = torch.cat(predict_prop_change_DOC_all, dim=0)
            predict_prop_change_MIC_all = torch.cat(predict_prop_change_MIC_all, dim=0)

            predict_prop_change_profile_id_all = torch.cat(predict_prop_change_profile_id_all, dim=0)
            predict_prop_change_para_all = torch.cat(predict_prop_change_para_all, dim=0)
            grid_coord_all = torch.cat(grid_coord_all, dim=0)

            predict_prop_change_carbon_input_all = torch.cat(predict_prop_change_carbon_input_all, dim=0)
            predict_prop_change_cpool_steady_state_all = torch.cat(predict_prop_change_cpool_steady_state_all, dim=0)
            predict_prop_change_cpool_layer_all = torch.cat(predict_prop_change_cpool_layer_all, dim=0)
            predict_prop_change_soc_layer_all = torch.cat(predict_prop_change_soc_layer_all, dim=0)
            predict_prop_change_total_res_time_all = torch.cat(predict_prop_change_total_res_time_all, dim=0)
            predict_prop_change_total_res_time_base_all = torch.cat(predict_prop_change_total_res_time_base_all, dim=0)
            predict_prop_change_res_time_base_pools_all = torch.cat(predict_prop_change_res_time_base_pools_all, dim=0)
            predict_prop_change_t_scalar_all = torch.cat(predict_prop_change_t_scalar_all, dim=0)
            predict_prop_change_bulk_A_doc_all = torch.cat(predict_prop_change_bulk_A_doc_all, dim=0)
            predict_prop_change_bulk_E_mic_all = torch.cat(predict_prop_change_bulk_E_mic_all, dim=0)
            predict_prop_change_bulk_A_mic_all = torch.cat(predict_prop_change_bulk_A_mic_all, dim=0)
            predict_prop_change_bulk_A_POM_all = torch.cat(predict_prop_change_bulk_A_POM_all, dim=0)
            predict_prop_change_bulk_A_MAOM_all = torch.cat(predict_prop_change_bulk_A_MAOM_all, dim=0)
            predict_prop_change_w_scalar_all = torch.cat(predict_prop_change_w_scalar_all, dim=0)
            predict_prop_change_bulk_K_all = torch.cat(predict_prop_change_bulk_K_all, dim=0)
            predict_prop_change_bulk_V_all = torch.cat(predict_prop_change_bulk_V_all, dim=0)
            predict_prop_change_bulk_xi_all = torch.cat(predict_prop_change_bulk_xi_all, dim=0)
            predict_prop_change_bulk_I_all = torch.cat(predict_prop_change_bulk_I_all, dim=0)
            predict_prop_change_litter_fraction_all = torch.cat(predict_prop_change_litter_fraction_all, dim=0)

            # Assign the gathered tensors to the local variables'
            predict_prop_change_soc = torch.full((grid_env_info_num, 20), torch.nan, device=device)
            predict_prop_change_POM = torch.full((grid_env_info_num, 20), torch.nan, device=device)
            predict_prop_change_MAOM = torch.full((grid_env_info_num, 20), torch.nan, device=device)
            predict_prop_change_DOC = torch.full((grid_env_info_num, 20), torch.nan, device=device)
            predict_prop_change_MIC = torch.full((grid_env_info_num, 20), torch.nan, device=device)

            predict_prop_change_profile_id = torch.full((grid_env_info_num, 1), torch.nan, device=device)
            predict_prop_change_para = torch.full((grid_env_info_num, len(para_names)), torch.nan, device=device)
            grid_coord = torch.full((grid_env_info_num, 2), torch.nan, device=device)

            predict_prop_change_carbon_input = torch.full((grid_env_info_num, 1), torch.nan, device=device)
            predict_prop_change_cpool_steady_state = torch.full((grid_env_info_num, 140), torch.nan, device=device)
            predict_prop_change_cpool_layer = torch.full((grid_env_info_num, 20), torch.nan, device=device)
            predict_prop_change_soc_layer = torch.full((grid_env_info_num, 20), torch.nan, device=device)
            predict_prop_change_total_res_time = torch.full((grid_env_info_num, 20), torch.nan, device=device)
            predict_prop_change_total_res_time_base = torch.full((grid_env_info_num, 20), torch.nan, device=device)
            predict_prop_change_res_time_base_pools = torch.full((grid_env_info_num, 140), torch.nan, device=device)
            predict_prop_change_t_scalar = torch.full((grid_env_info_num, 20), torch.nan, device=device)
            predict_prop_change_bulk_A_doc = torch.full((grid_env_info_num, 1), torch.nan, device=device)
            predict_prop_change_bulk_E_mic = torch.full((grid_env_info_num, 1), torch.nan, device=device)
            predict_prop_change_bulk_A_mic = torch.full((grid_env_info_num, 1), torch.nan, device=device)
            predict_prop_change_bulk_A_POM = torch.full((grid_env_info_num, 1), torch.nan, device=device)
            predict_prop_change_bulk_A_MAOM = torch.full((grid_env_info_num, 1), torch.nan, device=device)
            predict_prop_change_w_scalar = torch.full((grid_env_info_num, 20), torch.nan, device=device)
            predict_prop_change_bulk_K = torch.full((grid_env_info_num, 1), torch.nan, device=device)
            predict_prop_change_bulk_V = torch.full((grid_env_info_num, 1), torch.nan, device=device)
            predict_prop_change_bulk_xi = torch.full((grid_env_info_num, 1), torch.nan, device=device)
            predict_prop_change_bulk_I = torch.full((grid_env_info_num, 1), torch.nan, device=device)
            predict_prop_change_litter_fraction = torch.full((grid_env_info_num, 1), torch.nan, device=device)


            predict_prop_change_profile_id_all_squueezed = predict_prop_change_profile_id_all.squeeze()

            if rank == 0:
                for profile_id in predict_prop_change_profile_id_all_squueezed:
                    if torch.isnan(profile_id).any():
                        continue
                    profile_id_int = int(profile_id.item())
                    idx = int(np.where(predict_prop_change_profile_id_all_squueezed == profile_id_int)[0][0])
                    predict_prop_change_soc[profile_id_int, :] = predict_prop_change_soc_all[idx, :]
                    predict_prop_change_POM[profile_id_int, :] = predict_prop_change_POM_all[idx, :]
                    predict_prop_change_MAOM[profile_id_int, :] = predict_prop_change_MAOM_all[idx, :]
                    predict_prop_change_DOC[profile_id_int, :] = predict_prop_change_DOC_all[idx, :]
                    predict_prop_change_MIC[profile_id_int, :] = predict_prop_change_MIC_all[idx, :]

                    predict_prop_change_profile_id[profile_id_int, 0] = profile_id_int
                    predict_prop_change_para[profile_id_int, :] = predict_prop_change_para_all[idx, :]
                    grid_coord[profile_id_int, 0] = grid_coord_all[idx, 0]  # Longitude
                    grid_coord[profile_id_int, 1] = grid_coord_all[idx, 1]  # Latitude

                    predict_prop_change_carbon_input[profile_id_int, 0] = predict_prop_change_carbon_input_all[idx, 0]
                    predict_prop_change_cpool_steady_state[profile_id_int, :] = predict_prop_change_cpool_steady_state_all[idx, :]
                    predict_prop_change_cpool_layer[profile_id_int, :] = predict_prop_change_cpool_layer_all[idx, :]
                    predict_prop_change_soc_layer[profile_id_int, :] = predict_prop_change_soc_layer_all[idx, :]
                    predict_prop_change_total_res_time[profile_id_int, :] = predict_prop_change_total_res_time_all[idx, :]
                    predict_prop_change_total_res_time_base[profile_id_int, :] = predict_prop_change_total_res_time_base_all[idx, :]
                    predict_prop_change_res_time_base_pools[profile_id_int, :] = predict_prop_change_res_time_base_pools_all[idx, :]
                    predict_prop_change_t_scalar[profile_id_int, :] = predict_prop_change_t_scalar_all[idx, :]
                    predict_prop_change_bulk_A_doc[profile_id_int, 0] = predict_prop_change_bulk_A_doc_all[idx, 0]
                    predict_prop_change_bulk_E_mic[profile_id_int, 0] = predict_prop_change_bulk_E_mic_all[idx, 0]
                    predict_prop_change_bulk_A_mic[profile_id_int, 0] = predict_prop_change_bulk_A_mic_all[idx, 0]
                    predict_prop_change_bulk_A_POM[profile_id_int, 0] = predict_prop_change_bulk_A_POM_all[idx, 0]
                    predict_prop_change_bulk_A_MAOM[profile_id_int, 0] = predict_prop_change_bulk_A_MAOM_all[idx, 0]
                    predict_prop_change_w_scalar[profile_id_int, :] = predict_prop_change_w_scalar_all[idx, :]
                    predict_prop_change_bulk_K[profile_id_int, 0] = predict_prop_change_bulk_K_all[idx, 0]
                    predict_prop_change_bulk_V[profile_id_int, 0] = predict_prop_change_bulk_V_all[idx, 0]
                    predict_prop_change_bulk_xi[profile_id_int, 0] = predict_prop_change_bulk_xi_all[idx, 0]
                    predict_prop_change_bulk_I[profile_id_int, 0] = predict_prop_change_bulk_I_all[idx, 0]
                    predict_prop_change_litter_fraction[profile_id_int, 0] = predict_prop_change_litter_fraction_all[idx, 0]
                
                # Save the predictions to a file
                np.savetxt(data_dir_output + f'/{var_name}/' + f'prop_change_{change:.2f}_soc.txt', predict_prop_change_soc.cpu().numpy(), delimiter=',')
                np.savetxt(data_dir_output + f'/{var_name}/' + f'prop_change_{change:.2f}_POM.txt', predict_prop_change_POM.cpu().numpy(), delimiter=',')
                np.savetxt(data_dir_output + f'/{var_name}/' + f'prop_change_{change:.2f}_MAOM.txt', predict_prop_change_MAOM.cpu().numpy(), delimiter=',')
                np.savetxt(data_dir_output + f'/{var_name}/' + f'prop_change_{change:.2f}_DOC.txt', predict_prop_change_DOC.cpu().numpy(), delimiter=',')
                np.savetxt(data_dir_output + f'/{var_name}/' + f'prop_change_{change:.2f}_MIC.txt', predict_prop_change_MIC.cpu().numpy(), delimiter=',')

                np.savetxt(data_dir_output + f'/{var_name}/' + f'prop_change_{change:.2f}_profile_id.txt', predict_prop_change_profile_id.cpu().numpy(), delimiter=',')
                np.savetxt(data_dir_output + f'/{var_name}/' + f'prop_change_{change:.2f}_para.txt', predict_prop_change_para.cpu().numpy(), delimiter=',')
                np.savetxt(data_dir_output + f'/{var_name}/' + f'prop_change_{change:.2f}_grid_coord.txt', grid_coord.cpu().numpy(), delimiter=',')

                np.savetxt(data_dir_output + f'/{var_name}/' + f'prop_change_{change:.2f}_carbon_input.txt', predict_prop_change_carbon_input.cpu().numpy(), delimiter=',')
                np.savetxt(data_dir_output + f'/{var_name}/' + f'prop_change_{change:.2f}_cpool_steady_state.txt', predict_prop_change_cpool_steady_state.cpu().numpy(), delimiter=',')
                np.savetxt(data_dir_output + f'/{var_name}/' + f'prop_change_{change:.2f}_cpool_layer.txt', predict_prop_change_cpool_layer.cpu().numpy(), delimiter=',')
                np.savetxt(data_dir_output + f'/{var_name}/' + f'prop_change_{change:.2f}_soc_layer.txt', predict_prop_change_soc_layer.cpu().numpy(), delimiter=',')
                np.savetxt(data_dir_output + f'/{var_name}/' + f'prop_change_{change:.2f}_total_res_time.txt', predict_prop_change_total_res_time.cpu().numpy(), delimiter=',')
                np.savetxt(data_dir_output + f'/{var_name}/' + f'prop_change_{change:.2f}_total_res_time_base.txt', predict_prop_change_total_res_time_base.cpu().numpy(), delimiter=',')
                np.savetxt(data_dir_output + f'/{var_name}/' + f'prop_change_{change:.2f}_res_time_base_pools.txt', predict_prop_change_res_time_base_pools.cpu().numpy(), delimiter=',')
                np.savetxt(data_dir_output + f'/{var_name}/' + f'prop_change_{change:.2f}_t_scalar.txt', predict_prop_change_t_scalar.cpu().numpy(), delimiter=',')
                np.savetxt(data_dir_output + f'/{var_name}/' + f'prop_change_{change:.2f}_bulk_A_doc.txt', predict_prop_change_bulk_A_doc.cpu().numpy(), delimiter=',')
                np.savetxt(data_dir_output + f'/{var_name}/' + f'prop_change_{change:.2f}_bulk_E_mic.txt', predict_prop_change_bulk_E_mic.cpu().numpy(), delimiter=',')
                np.savetxt(data_dir_output + f'/{var_name}/' + f'prop_change_{change:.2f}_bulk_A_mic.txt', predict_prop_change_bulk_A_mic.cpu().numpy(), delimiter=',')
                np.savetxt(data_dir_output + f'/{var_name}/' + f'prop_change_{change:.2f}_bulk_A_POM.txt', predict_prop_change_bulk_A_POM.cpu().numpy(), delimiter=',')
                np.savetxt(data_dir_output + f'/{var_name}/' + f'prop_change_{change:.2f}_bulk_A_MAOM.txt', predict_prop_change_bulk_A_MAOM.cpu().numpy(), delimiter=',')
                np.savetxt(data_dir_output + f'/{var_name}/' + f'prop_change_{change:.2f}_w_scalar.txt', predict_prop_change_w_scalar.cpu().numpy(), delimiter=',')
                np.savetxt(data_dir_output + f'/{var_name}/' + f'prop_change_{change:.2f}_bulk_K.txt', predict_prop_change_bulk_K.cpu().numpy(), delimiter=',')
                np.savetxt(data_dir_output + f'/{var_name}/' + f'prop_change_{change:.2f}_bulk_V.txt', predict_prop_change_bulk_V.cpu().numpy(), delimiter=',')
                np.savetxt(data_dir_output + f'/{var_name}/' + f'prop_change_{change:.2f}_bulk_xi.txt', predict_prop_change_bulk_xi.cpu().numpy(), delimiter=',')
                np.savetxt(data_dir_output + f'/{var_name}/' + f'prop_change_{change:.2f}_bulk_I.txt', predict_prop_change_bulk_I.cpu().numpy(), delimiter=',')
                np.savetxt(data_dir_output + f'/{var_name}/' + f'prop_change_{change:.2f}_litter_fraction.txt', predict_prop_change_litter_fraction.cpu().numpy(), delimiter=',')
                
                # Save the grid environment info update
                np.savetxt(data_dir_output + f'/{var_name}/' + f'grid_env_info_update.txt', grid_env_info_update.cpu().numpy(), delimiter=',')
                
                print(f"Data used for environmental variable {var_name} is {batch_x[0, var_idx, 0, 0].item()} with change {change:.2f}")
                print(f"Proportional change {change:.2f} predictions saved successfully for variable {var_name} with time taken: {time.time() - change_start_time:.2f} seconds")
            # End of if rank == 0
            dist.barrier()  
        
        if rank == 0:
            print(f"All proportional changes for variable {var_name} processed with time taken: {time.time() - var_start_time:.2f} seconds")
        dist.barrier()  



if __name__ == '__main__':
	# Number of CPUs requester
	world_size = args.num_CPU
	processes = []
	port='12355'

	# Create job ID
	print("MAIN, JOB ID", job_id)
	print("Command:", " ".join(sys.argv))

	for rank in range(world_size):
		p = Process(target=worker, args=(rank, world_size, job_id, port))
		p.start()
		processes.append(p)

	for p in processes:
		p.join()            
                




