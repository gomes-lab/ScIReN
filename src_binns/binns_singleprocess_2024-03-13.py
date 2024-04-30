# Distributed Data Parallel (DDP) training script for BINN
# Initializer script for DDP, automatically resubmit jobs after 11.5 hours (call DDP_resume.py)
# Import the required libraries
import csv
import functools
import math
import sys
import time
import random
import warnings
import subprocess
import argparse
from mlp import GCN_BINN, mlp_wrapper
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
from pe_gcn_model import GridCellSpatialRelationEncoder

# sys.path.append('C:/Users/hx293/Research_Data/BINN/')
# sys.path.append('/glade/u/home/haodixu/BINN')
# sys.path.append(r'/User/homes/ftao/Projects/BINNS/src_binns')
sys.path.append('/glade/work/haodixu/BINN')

# Set HDF5_DISABLE_VERSION_CHECK to suppress version mismatch error
import os
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2'
import psutil
import gc

from datetime import datetime, timedelta
import pandas as pd
from pandas import DataFrame as df
import numpy as np
from scipy.interpolate import pchip_interpolate
# parallel computing
# import concurrent.futures
# import torch.multiprocessing as mp
# # initialize the multiprocessing for pytorch


import os
import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import multiprocessing
from multiprocessing import Process

from scipy.io import loadmat
import netCDF4 as ncread 
import mat73

from matplotlib import pyplot as plt
from collections import OrderedDict

#####################################
# Import Different Versions of CLM5 #
#####################################

# from fun_matrix_clm5 import fun_model_simu
from fun_matrix_clm5_vectorized import fun_model_simu, fun_model_prediction
# from fun_matrix_clm5_vectorized_old_V import fun_model_simu, fun_model_prediction
# from fun_matrix_clm5_GPU import fun_model_simu
# from fun_matrix_clm5_parallel import fun_model_simu
# from fun_matrix_clm5_parallel_update_V_matrix import fun_model_simu
import misc_utils
import visualization_utils
from fun_matrix_clm5_vectorized_bulk_converge import fun_bulk_simu
# from fun_matrix_clm5_vectorized_prediction import fun_model_prediction

import LibMTL.weighting as weighting_method
import LibMTL.architecture as architecture_method

################################################
# @joshuafan: Command-line arguments
################################################
parser = argparse.ArgumentParser()

# Model architecture and note
parser.add_argument("--note", type=str, default="", help="Optional name to give to the model")
parser.add_argument("--model", type=str, default="old_mlp", choices=['old_mlp', 'new_mlp', 'lipmlp', 'gcn'], help="Model type")
parser.add_argument("--categorical", type=str, default="embedding", choices=["embedding", "one_hot"], help="Which embedding to use for categorical variables")
parser.add_argument("--embed_dim", type=int, default=5, help="Embedding dim for each categorical variable (if using embeddings)")
parser.add_argument("--use_bn", action='store_true', help="Whether to use batchnorm")
parser.add_argument("--vertical_mixing", type=str, choices=['original', 'simple_one_intercept', 'simple_two_intercepts'], help="Vertical mixing matrix parameterization. Simple uses a log-log relationship with depth")

# Training
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--n_datapoints", type=int, default=-1, help="Set this to sample this many datapoints randomly (includes train+val+test). -1 to use the whole dataset")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_epochs", type=int, default=1500)
parser.add_argument("--patience", type=int, default=500)

# Regularization
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--use_swa", action='store_true', help="Whether to use Stochastic Weight Averaging")
parser.add_argument("--clip_value", type=float, default=1, help="Clip value for gradient clipping")

# Positional encoding
parser.add_argument("--lonlat_features", action='store_true', help="Whether longitude and latitude should be passed as features")
parser.add_argument("--pos_enc", type=str, default='none', choices=['none', 'early', 'late'], 
					help="Whether to use positional encoding. 'early' means that positional encoding is concatenated with other features. 'late' means that it is only used as an error term for the latent parameters.")
parser.add_argument("--k", type=int, default=20, help="Nearest neighbors for graph (GCN only)")
# Loss terms
parser.add_argument("--losses", nargs="+", choices=["l1", "l2", "param_reg", "spectral", "lipmlp", "cure"], default=["l1", "param_reg"])
parser.add_argument("--loss_weighting", choices=["manual", "relobralo", "IMTL", "two_stage"])
parser.add_argument("--lambdas", nargs="+", type=float, default=[1.0], help="If loss_weighting is manual, provide weights in the same order that you listed losses in `args.losses`")
parser.add_argument("--second_start", type=int, default=30, help="If loss_weighting is two_stage, epoch the second phase starts")
parser.add_argument("--second_lambdas", nargs="+", type=float, default=[1.0], help="If loss_weighting is two_stage, provide weights in the same order that you listed losses in `args.losses`")

# Relobralo specific hyperparams
parser.add_argument("--relobralo_alpha", type=float, default=0.9, help="Exponential decay rate for Relobralo")
parser.add_argument("--relobralo_temp", type=float, default=0.1, help="Softmax temperature for Relobralo")
parser.add_argument("--relobralo_saudade", type=float, default=0.999, help="Saudade (1 minus probability of looking back to epoch 0)")

# Computational environment
parser.add_argument("--use_ddp", type=int, default=0, help="Whether to use DDP")
parser.add_argument("--num_CPU", type=int, default=32)
parser.add_argument("--scheduler", type=str, default="pbs", choices=["pbs", "slurm"], help="Job scheduler system")
parser.add_argument("--whether_resume", type=int, default=0, help="Whether to resume training from a previous model")
parser.add_argument("--previous_job_id", type=str, default="", help="Previous job id to resume from (otherwise, resumes using envir variable PREVIOUS_JOB_ID)")

args = parser.parse_args()

# If loss_weighting is manual, make sure the correct number of lambdas were provided
if args.loss_weighting in ["manual", "relobralo", "two_stage"]:
	assert(len(args.losses) == len(args.lambdas))
	args.lambdas = torch.tensor(args.lambdas)

# @joshuafan: Set random seeds to try to ensure reproducibility
random.seed(args.seed)
np.random.seed(args.seed) # set the random seed of numpy
torch.manual_seed(args.seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# print the number of cores
cpu_count = multiprocessing.cpu_count()
thread_count = torch.get_num_threads()
# print("Number of CPUs: ", torch.cpu.device_count())  # No idea why it is not working on NCAR server
print("Number of Cores: ", cpu_count)
print("Number of threads: ", thread_count)

# Device - @joshuafan changed
if torch.cuda.is_available():
	cuda_visible_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
	assert len(cuda_visible_devices) == 1, "Multi-GPU training not supported yet"
	dev = f'cuda:{cuda_visible_devices[0]}'
else:
	dev = 'cpu'
device = torch.device(dev)
print(datetime.now(), '------------all packages loaded------------ device:', dev)

# @joshuafan: Create a "job id" using timestamp, note, and PBS jobid
job_begin_time = time.time()
if args.whether_resume == 0:
	# If not resuming, create a new job id
	job_id = time.strftime("%Y%m%d-%H%M%S")  # Convert datetime to string: https://stackoverflow.com/questions/10607688/how-to-create-a-file-name-with-the-current-date-time-in-python
	if args.note != "":
		job_id += ("_" + args.note)
	pbs_job_id = os.environ.get('PBS_JOBID')
	if pbs_job_id is not None:
		pbs_job_id = pbs_job_id.split('.')[0]
		job_id += ("_" + pbs_job_id)
	print(f"New job ID: {job_id}")
else:
	# If resuming, use the same job id as before
	if args.previous_job_id != "":
		job_id = args.previous_job_id
	else:
		job_id = os.environ.get('PREVIOUS_JOB_ID')
	print(f"Previous job ID: {job_id}")



################################################
# input data
################################################
cesm2_case_name = 'sasu_f05_g16_checked_step4'
start_year = 661
end_year = 680

time_domain = 'whole_time' # 'whole_time', 'before_1985', 'after_1985', 'random_half_1', 'random_half_2'
model_name = 'cesm2_clm5_cen_vr_v2'

start_id = 1
end_id = 5000
is_resubmit = 0

# # @joshuafan changed
# # pathway
# # data_dir_input = '/Users/phoenix/Google_Drive/Tsinghua_Luo/Projects/DATAHUB/ENSEMBLE/INPUT_DATA/'
# # data_dir_output = '/Users/phoenix/Google_Drive/Tsinghua_Luo/Projects/DATAHUB/BINNS/OUTPUT_DATA/'
# # data_dir_input = 'C:/Users/hx293/Research_Data/BINN/ENSEMBLE/INPUT_DATA/'
# # data_dir_output = 'C:/Users/hx293/Unsync_Data/BINN_output/'
# # server path
# # job_submit_path = '/glade/u/home/haodixu/BINN/PBS_Submit/Bulk_Converge/'
# # data_dir_input = '/glade/u/home/haodixu/BINN/ENSEMBLE/INPUT_DATA/'
# # data_dir_output = '/glade/work/haodixu/BINN/BINNS/OUTPUT_DATA/'
# # job_submit_path = '/glade/u/home/haodixu/BINN/PBS_Submit/Bulk_Converge/'
# # data_dir_input = '/glade/u/home/haodixu/BINN/ENSEMBLE/INPUT_DATA/'
# # data_dir_output = '/glade/work/haodixu/BINN/BINNS/OUTPUT_DATA/'
job_submit_path = '/mnt/beegfs/bulk/mirror/jyf6/datasets/BINNS/aida_submit/Bulk_Converge'
data_dir_input = '/mnt/beegfs/bulk/mirror/jyf6/datasets/BINNS/INPUT_DATA/'
data_dir_output = '/mnt/beegfs/bulk/mirror/jyf6/datasets/BINNS/OUTPUT_DATA/'
os.makedirs(job_submit_path, exist_ok=True)
os.makedirs(os.path.join(data_dir_output, "neural_network"), exist_ok=True)
os.makedirs(os.path.join(data_dir_output, "neural_network", job_id), exist_ok=True)
# create folder for the model parameters
os.makedirs(os.path.join(data_dir_output, "neural_network", job_id, "model_parameters"), exist_ok=True)
# create folder for the model training history
os.makedirs(os.path.join(data_dir_output, "neural_network", job_id, "model_training_history"), exist_ok=True)
# create folder for the visualization
PLOT_DIR = os.path.join(data_dir_output, 'neural_network', job_id, 'visualizations')
os.makedirs(PLOT_DIR, exist_ok=True)
# create folder for the model prediction
os.makedirs(os.path.join(data_dir_output, "neural_network", job_id, "Prediction"), exist_ok=True)
# create folder for the bulk simulation
os.makedirs(os.path.join(data_dir_output, "neural_network", job_id, "Bulk_Simulation"), exist_ok=True)

# Load checkpoint if resuming
if args.whether_resume == 1:
	# checkpoint path
	checkpoint_path = data_dir_output + 'neural_network/' + job_id + '/checkpoint_' + job_id + '.pt'
	if not os.path.exists(checkpoint_path):
		checkpoint_path = data_dir_output + 'neural_network/' + job_id + '/opt_nn_' + job_id + '.pt'

	# Load Checkpoint
	checkpoint_main = torch.load(checkpoint_path, map_location=device)
	
	# Delete the job submit file
	if os.path.exists(job_submit_path + 'Resume' + job_id + '.submit'):
		os.remove(job_submit_path + 'Resume' + job_id + '.submit')
	


# constants
month_num = 12 
soil_cpool_num = 7
soil_decom_num = 20

#-------------------------------
# wosis data
#-------------------------------
# load wosis data

# The site information for each SOC profile. 
# Names for each column are "profile_id" "country_id" "country_name" "lon" "lat" "layer_num" “date”. 
nc_data_middle = ncread.Dataset(data_dir_input + 'wosis_2019_snap_shot/soc_profile_wosis_2019_snapshot_hugelius_mishra.nc') # wosis profile info
wosis_profile_info = nc_data_middle['soc_profile_info'][:].data.transpose()
nc_data_middle.close()

# The full dataset which contains SOC content information at each layer
# layer_info: "profile_id, date, upper_depth, lower_depth, node_depth, soc_layer_weight, soc_stock, bulk_denstiy, is_pedo"
nc_data_middle = ncread.Dataset(data_dir_input + 'wosis_2019_snap_shot/soc_data_integrate_wosis_2019_snapshot_hugelius_mishra.nc') # wosis SOC info
wosis_soc_info = nc_data_middle['data_soc_integrate'][:].data.transpose()
nc_data_middle.close()

#-------------------------------
# PRODA Predicted Parameters
#-------------------------------
# load PRODA predicted parameters

# The site information for each parameter prediction.
# Get the profile id for predicted parameters
# data from nn_site_loc_full_cesm2_clm5_cen_vr_v2_whole_time_exp_pc_cesm2_23_cross_valid_0_1.csv to 9
for i in range(1, 10):
	# contains one column of profile id
	nn_site_loc_temp = pd.read_csv(data_dir_input + 'PRODA_Results/nn_site_loc_full_cesm2_clm5_cen_vr_v2_whole_time_exp_pc_cesm2_23_cross_valid_0_' + str(i) + '.csv')
	# contains the predicted parameters (21) for each profile
	nn_site_para_temp = pd.read_csv(data_dir_input + 'PRODA_Results/nn_para_result_full_cesm2_clm5_cen_vr_v2_whole_time_exp_pc_cesm2_23_cross_valid_0_' + str(i) + '.csv')
	# create a dataframe to store the profile id and the parameters
	if i == 1:
		# initialize the dataframe
		PRODA_para = pd.DataFrame(nn_site_loc_temp)
		# rename the column
		PRODA_para.columns = ['profile_id']
		# add the parameters
		PRODA_para = pd.concat([PRODA_para, nn_site_para_temp], axis = 1)
	else:
		# add the parameters
		PRODA_para = pd.concat([PRODA_para, nn_site_para_temp], axis = 1)
# end
# Get the mean value for each parameter for each profile
for i in range(1,22):
	PRODA_para['mean_' + str(i)] = PRODA_para.iloc[:, i:21*10:21].mean(axis = 1)
# end
# Drop the original columns
PRODA_para = PRODA_para.drop(PRODA_para.columns[1:21*9], axis = 1)
# print the head of the dataframe
print(PRODA_para.head())

#-------------------------------
# CLM5 constants
#-------------------------------
para_names = ['slope', 'intercept', 'q10', 'efolding', 'taucwd', 'taul1', 'taul2', 'tau4s1', 'tau4s2', 'tau4s3', 'fl1s1', 'fl2s1', 'fl3s2', 'fs1s2', 'fs1s3', 'fs2s1', 'fs2s3', 'fs3s1', 'fcwdl2', 'w-scaling', 'beta']
if args.vertical_mixing == 'simple_two_intercepts':
	para_names.append('intercept_leach')
# para_names = ['diffus', 'cryo', 'q10', 'efolding', 'taucwd', 'taul1', 'taul2', 'tau4s1', 'tau4s2', 'tau4s3', 'fl1s1', 'fl2s1', 'fl3s2', 'fs1s2', 'fs1s3', 'fs2s1', 'fs2s3', 'fs3s1', 'fcwdl2', 'w-scaling', 'beta']

# parameters index for retrieval test
# If choosing all parameters
para_index = np.arange(0, len(para_names))
# para_index = [0, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20]

# soil depths info
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

# representative points 
sample_profile_id = loadmat(data_dir_input + 'wosis_2019_snap_shot/wosis_2019_snapshot_hugelius_mishra_representative_profiles.mat')
sample_profile_id = sample_profile_id['sample_profile_id']
# convert the number to be starting from 0 in python world
sample_profile_id = sample_profile_id - 1

### Use 2000 profiles for testing ###
# profile_collection = np.reshape(sample_profile_id[:, 0:20], [2000, 1])

# if use the whole dataset
# profile_collection = np.arange(0, 20000)
# if select 
# profile_collection = np.arange(0, wosis_profile_info.shape[0])
# profile_collection = np.reshape(profile_collection, [profile_collection.shape[0], 1])	

# choose the profile id with lat and lon within the range of the United States
profile_collection = np.where(
	(wosis_profile_info[:, 2] == 156) & 
	(wosis_profile_info[:, 3] >= -124.763068) & 
	(wosis_profile_info[:, 3] <= -66.949895) & 
	(wosis_profile_info[:, 4] >= 24.5) & 
	(wosis_profile_info[:, 4] <= 49.384358)
)[0]

################################
# If use same dataset as PRODA #
################################
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

# Choose random 2000 profiles for testing.
if args.n_datapoints != -1:
	profile_collection = np.random.choice(profile_collection, args.n_datapoints, replace=False)

###############################################################################################################

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
obs_lon_lat_loc = np.ones([len(profile_collection), 2])*np.nan

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
	
	# interpolation
	# if num_layers > 1:
	# 	wosis_layer_depth = wosis_layer_depth[valid_soc_loc]/100 # convert unit from cm to m
	# 	wosis_layer_obs = wosis_layer_obs[valid_soc_loc]
	# 	
	# 	wosis_layer_depth = wosis_layer_depth + (np.random.rand(num_layers)-0.5)*10**(-7)
	# 	sort_index = np.argsort(wosis_layer_depth)
	# 	wosis_layer_depth = wosis_layer_depth[sort_index]
	# 	wosis_layer_obs = wosis_layer_obs[sort_index]
	# 	
	# 	interp_soc = pchip_interpolate(wosis_layer_depth, wosis_layer_obs, zsoi)
	# 	interp_soc[interp_soc <= 0] = np.nan
	# 	interp_start_loc = np.where(abs(wosis_layer_depth[0] - zsoi) == min(abs(wosis_layer_depth[0] - zsoi)))[0]
	# 	interp_end_loc = np.where(abs(wosis_layer_depth[-1] - zsoi) == min(abs(wosis_layer_depth[-1] - zsoi)))[0]
	# 	if (interp_start_loc < 19) & (interp_end_loc < 19):
	# 		# print('multilayer profile: ', iprofile_hat)
	# 		obs_soc_matrix[iprofile_hat, interp_start_loc[0]:interp_end_loc[0]] = interp_soc[interp_start_loc[0]:interp_end_loc[0]]
	# elif num_layers == 1:
	# 	wosis_layer_depth = wosis_layer_depth[valid_soc_loc]/100 # convert unit from cm to m
	# 	wosis_layer_obs = wosis_layer_obs[valid_soc_loc]
	# 	
	# 	closest_loc = np.where(abs(wosis_layer_depth[0] - zsoi) == min(abs(wosis_layer_depth[0] - zsoi)))[0]
	# 	obs_soc_matrix[iprofile_hat, closest_loc] = wosis_layer_obs[0]
	# elif num_layers == 0:
	# 	print('invalid profile: ', iprofile_hat)
	# # end if num_layers > 3:

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
#---------------------------------------------------
# env info
#---------------------------------------------------
# environmental info of soil profiles

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

categorical_vars = [['ESA_Land_Cover'], ['Texture_USDA_0cm', 'Texture_USDA_30cm', 'Texture_USDA_100cm'], 
					['USDA_Suborder'], ['WRB_Subgroup'], ['Koppen_Climate_2018']]  # Variables inside a sub-list share the same categories
categorical_vars_flattened = [item for sublist in categorical_vars for item in sublist]

env_info = loadmat(data_dir_input + 'wosis_2019_snap_shot/wosis_2019_snapshot_hugelius_mishra_env_info.mat')
env_info = env_info['EnvInfo']
original_lons = env_info[:, 3].copy()
original_lats = env_info[:, 4].copy()

col_max_min = loadmat(data_dir_input + 'wosis_2019_snap_shot/world_grid_envinfo_present_cesm2_clm5_cen_vr_v2_whole_time_col_max_min.mat')
col_max_min = col_max_min['col_max_min']

# Don't want to transform categorical variables, so set max/min to nan
for group in categorical_vars:
	for var in group:
		idx = env_info_names.index(var)
		col_max_min[idx, :] = np.nan

# warnings.filterwarnings("error")
for ivar in np.arange(3, len(col_max_min[:, 0])):
	if np.isnan(col_max_min[ivar, :]).any():
		pass
	else:
		env_info[:, ivar] = (env_info[:, ivar] - col_max_min[ivar, 0])/(col_max_min[ivar, 1] - col_max_min[ivar, 0])
		env_info[(env_info[:, ivar] > 1), ivar] = 1
		env_info[(env_info[:, ivar] < 0), ivar] = 0
	# except:
	# 	print('error in variable: ', ivar)
# warnings.resetwarnings()

env_info = df(env_info)

# env_info_scaled = loadmat(data_dir_input + 'wosis_2019_snap_shot/wosis_2019_snapshot_hugelius_mishra_env_info_' + model_name  + '_' + time_domain + '_maxmin_scaled.mat')
# env_info_scaled = df(env_info_scaled['profile_env_info'])
# env_info = env_info_scaled



env_info.columns = env_info_names
env_info["original_lon"] = original_lons  # Save original lon/lat (before rescaling)
env_info["original_lat"] = original_lats

# # @joshuafan added temporarily
# env_info.index = env_info.ProfileNum
# print("Env info old shape", env_info.shape)
# print("Env info", env_info.head())
# print(profile_collection[0:5, 0])

# variables used in training the NN
var4nn = ['Lon', 'Lat', \
'ESA_Land_Cover', \
# 'IGBP', \
# 'Climate', \
# 'Soil_Type', \
# 'NPPmean', 'NPPmax', 'NPPmin', \
# 'Veg_Cover', \
'BIO1', 'BIO2', 'BIO3', 'BIO4', 'BIO5', 'BIO6', 'BIO7', 'BIO8', 'BIO9', 'BIO10', 'BIO11', 'BIO12', 'BIO13', 'BIO14', 'BIO15', 'BIO16', 'BIO17', 'BIO18', 'BIO19', \
'Abs_Depth_to_Bedrock', \
'Bulk_Density_0cm', 'Bulk_Density_30cm', 'Bulk_Density_100cm',\
'CEC_0cm', 'CEC_30cm', 'CEC_100cm', \
'Clay_Content_0cm', 'Clay_Content_30cm', 'Clay_Content_100cm', \
'Coarse_Fragments_v_0cm', 'Coarse_Fragments_v_30cm', 'Coarse_Fragments_v_100cm', \
# 'Depth_Bedrock_R', \
'Garde_Acid', \
'Occurrence_R_Horizon', \
'pH_Water_0cm', 'pH_Water_30cm', 'pH_Water_100cm', \
'Sand_Content_0cm', 'Sand_Content_30cm', 'Sand_Content_100cm', \
'Silt_Content_0cm', 'Silt_Content_30cm', 'Silt_Content_100cm', \
'SWC_v_Wilting_Point_0cm', 'SWC_v_Wilting_Point_30cm', 'SWC_v_Wilting_Point_100cm', \
'Texture_USDA_0cm', 'Texture_USDA_30cm', 'Texture_USDA_100cm', \
'USDA_Suborder', \
'WRB_Subgroup', \
# 'Drought', \
'Elevation', \
# 'Max_Depth', \
'Koppen_Climate_2018', \
'cesm2_npp', 'cesm2_npp_std', \
# 'cesm2_gpp', 'cesm2_gpp_std', \
'cesm2_vegc', \
'nbedrock']

# If desired, remove lon/lat as features
if not args.lonlat_features:
	var4nn.remove('Lon')
	var4nn.remove('Lat')
	print(f"Not using Lon/Lat as features. Remaining features: {var4nn}")


#---------------------------------------------------
# training data
#---------------------------------------------------
assert len(var4nn) >= 20
current_data_x = np.ones((len(profile_collection), len(var4nn), 12, 13))*np.nan
current_data_x[:, 0:len(var4nn), 0, 0] = np.array(env_info.loc[profile_collection[:, 0], var4nn])
current_data_x[:, 0:12, 0, 1] = model_force_input_vector_cwd
current_data_x[:, 0:12, 0, 2] = model_force_input_vector_litter1
current_data_x[:, 0:12, 0, 3] = model_force_input_vector_litter2
current_data_x[:, 0:12, 0, 4] = model_force_input_vector_litter3
current_data_x[:, 0:12, 0, 5] = model_force_altmax_lastyear_profile
current_data_x[:, 0:12, 0, 6] = model_force_altmax_current_profile
current_data_x[:, 0:12, 0, 7] = model_force_nbedrock

current_data_x[:, 0:20, 0:12, 8] = model_force_xio
current_data_x[:, 0:20, 0:12, 9] = model_force_xin
current_data_x[:, 0:20, 0:12, 10] = model_force_sand_vector
current_data_x[:, 0:20, 0:12, 11] = model_force_soil_temp_profile
current_data_x[:, 0:20, 0:12, 12] = model_force_soil_water_profile


current_data_y = obs_soc_matrix
current_data_z = obs_depth_matrix

lons = np.array(env_info.loc[profile_collection[:, 0], "original_lon"])
lats = np.array(env_info.loc[profile_collection[:, 0], "original_lat"])
current_data_c = np.stack([lons, lats], axis=1)  # [profile, 2]: lon/lat of each site


# for col_idx, col_name in enumerate(var4nn):
# 	envir_var_values = current_data_x[:, col_idx, 0, 0]
# 	categorical = (col_name in categorical_vars_flattened)
# 	visualization_utils.plot_observations_world_map(lons, lats, envir_var_values, PLOT_DIR, col_name, categorical=categorical)

# # Plot SOC observation labels within each layer. If a profile has multiple observations 
# # in a layer, pick the first one
# layer_top = 0
# for layer_idx in range(len(zisoi)):
# 	layer_bottom = zisoi[layer_idx]
# 	this_layer_y = np.ones((current_data_y.shape[0])) * np.nan

# 	# Loop through all profiles
# 	for j in range(current_data_y.shape[0]):
# 		# Get depth of each SOC observation
# 		depths = current_data_z[j]

# 		# Select SOC observations whose depth falls within the current layer
# 		this_layer_this_profile_y = current_data_y[j, (~np.isnan(depths)) & (depths >= layer_top) & (depths < layer_bottom)]
# 		if len(this_layer_this_profile_y) > 1:
# 			continue
# 			print("Oddly enough this profile had more than 2 observations in the same soil layer")
# 			print("Layer", layer_top, "to", layer_bottom)
# 			print("Observation depths", depths)
# 		elif len(this_layer_this_profile_y) == 0:
# 			continue
# 		else:
# 			this_layer_y[j] = this_layer_this_profile_y[0]
# 	layer_name = "Layer {} ({:.2f}-{:.2f} m)".format(layer_idx, layer_top, layer_bottom)
# 	col_name = "soc_layer{}_{:.2f}-{:.2f}m".format(layer_idx, layer_top, layer_bottom)
# 	visualization_utils.plot_observations_world_map(lons, lats, this_layer_y, PLOT_DIR, col_name)
# 	layer_top = layer_bottom


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

valid_profile_loc = np.where(np.isnan(nan_loc) == False)[0] ### Why change the shape from 26915 to 26934??? ###

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
print("Shape of obs upper depth matrix", obs_upper_depth_matrix.shape)
print("Shape of obs lower depth matrix", obs_lower_depth_matrix.shape)
# env_info = env_info.loc[valid_profile_loc, :]

# Select PRODA parameters so that the Profile_IDs match the current data
PRODA_para = PRODA_para.loc[PRODA_para['profile_id'].isin(current_data_profile_id)]
PRODA_para = PRODA_para.sort_values(by='profile_id')                 
print("Shape of PRODA para", PRODA_para.shape)



# Train, validation, test split
if args.whether_resume == 0:
	if test_split_ratio == 0:
		train_loc = np.random.choice(np.arange(0, len(current_data_x[:, 0])), size = round((1-nn_split_ratio)*len(current_data_x[:, 0])), replace = False)
		val_loc = np.setdiff1d(np.arange(0, len(current_data_x[:, 0])), train_loc)

		train_y = torch.tensor(current_data_y[train_loc, :], dtype = torch.float32)
		val_y = torch.tensor(current_data_y[val_loc, :], dtype = torch.float32)

		train_z = torch.tensor(current_data_z[train_loc, :], dtype = torch.float32)
		val_z = torch.tensor(current_data_z[val_loc, :], dtype = torch.float32)

		train_c = torch.tensor(current_data_c[train_loc, :], dtype = torch.float32)
		val_c = torch.tensor(current_data_c[val_loc, :], dtype = torch.float32)

		train_x = torch.tensor(current_data_x[train_loc, :, :, :], dtype = torch.float32)
		# train_x = train_x.requires_grad_(True)
		val_x = torch.tensor(current_data_x[val_loc, :, :, :], dtype = torch.float32)
		# val_x = val_x.requires_grad_(True)

		train_profile_id = torch.tensor(current_data_profile_id[train_loc], dtype = torch.long)
		val_profile_id = torch.tensor(current_data_profile_id[val_loc], dtype = torch.long)
	else:
		# Determine the number of training samples based on the ratios
		train_loc = np.random.choice(np.arange(0, len(current_data_x[:, 0])), size=round((1 - nn_split_ratio - test_split_ratio) * len(current_data_x[:, 0])), replace=False)
		# The remaining data after removing the training samples
		remaining_loc = np.setdiff1d(np.arange(0, len(current_data_x[:, 0])), train_loc)
		# Split the remaining data into validation and test sets
		num_val_samples = round(nn_split_ratio / (nn_split_ratio + test_split_ratio) * len(remaining_loc))
		val_loc = np.random.choice(remaining_loc, size=num_val_samples, replace=False)
		test_loc = np.setdiff1d(remaining_loc, val_loc)

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
		# train_x = train_x.requires_grad_(True)
		val_x = torch.tensor(current_data_x[val_loc, :, :, :], dtype=torch.float32)
		# val_x = val_x.requires_grad_(True)
		test_x = torch.tensor(current_data_x[test_loc, :, :, :], dtype=torch.float32)
		# test_x = test_x.requires_grad_(True)

		train_profile_id = torch.tensor(current_data_profile_id[train_loc], dtype=torch.long)
		val_profile_id = torch.tensor(current_data_profile_id[val_loc], dtype=torch.long)
		test_profile_id = torch.tensor(current_data_profile_id[test_loc], dtype=torch.long)
else:
	# load train, val, and test indices
	train_loc = checkpoint_main['train_indices']
	val_loc = checkpoint_main['val_indices']
	test_loc = checkpoint_main['test_indices']
	# split the data
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

	train_profile_id = torch.tensor(current_data_profile_id[train_loc], dtype=torch.long)
	val_profile_id = torch.tensor(current_data_profile_id[val_loc], dtype=torch.long)
	test_profile_id = torch.tensor(current_data_profile_id[test_loc], dtype=torch.long)


#---------------------------------------------------
# Grid env info for prediction
#---------------------------------------------------
# load grid env info
grid_env_info = loadmat(data_dir_input + 'wosis_2019_snap_shot/world_grid_envinfo_present.mat')
grid_env_info = grid_env_info['EnvInfo']
original_lons_grid = grid_env_info[:, 0].copy()  # Save original lon/lat (before rescaling)
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

# Remove the first 3 columns and the last column from the col_max_min matrix
col_max_min_grid = col_max_min[3:-1, :]

# Normalize grid env info
for ivar in np.arange(0, len(col_max_min_grid[:, 0])):
	if np.isnan(col_max_min_grid[ivar, :]).any():
		pass
	else:
		grid_env_info[:, ivar] = (grid_env_info[:, ivar] - col_max_min_grid[ivar, 0])/(col_max_min_grid[ivar, 1] - col_max_min_grid[ivar, 0])
		grid_env_info[(grid_env_info[:, ivar] > 1), ivar] = 1
		grid_env_info[(grid_env_info[:, ivar] < 0), ivar] = 0

grid_env_info = df(grid_env_info)
grid_env_info.columns = grid_env_info_names
# Only keep the variables used in training the NN
grid_env_info = grid_env_info[var4nn]
grid_env_info["original_lon"] = original_lons_grid
grid_env_info["original_lat"] = original_lats_grid
# Exclude all rows with nan values
grid_env_info = grid_env_info.dropna(axis=0, how='any')

# Select the rows with lon and lat values within continental US
grid_env_info_US = grid_env_info[(grid_env_info["original_lon"] >= -124.763068) 
								& (grid_env_info["original_lon"] <= -66.949895)
								& (grid_env_info["original_lat"] >= 24.521694)
								& (grid_env_info["original_lat"] <= 49.384358)]
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
predict_data_x = np.ones((grid_env_info_num, len(var4nn), 12, 13))*np.nan
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
predict_data_c = np.stack([original_lons_grid, original_lats_grid], axis=1)

# print(datetime.now(), '------------grid env info prepared------------')
print(datetime.now(), '------------ FINISHED ALL PREPROCESSING ------------')




#---------------------------------------------------
# constants for NN
#---------------------------------------------------
nn_training_name = job_id + '_' + model_name

# writer = SummaryWriter(data_dir_output + 'tensorboard/' + nn_training_name)

#---------------------------------------------------
# define the loss function                          
#---------------------------------------------------
def binns_loss(y_pred, y_true, pred_para, plot_path=""):
	# process modeling
	soc_simu = y_pred
	# observations
	soc_true = y_true
	# predicted parameters
	pred_para = pred_para

	# flatten simulation
	soc_simu_vector = torch.reshape(soc_simu, [1, -1])
	soc_true_vector = torch.reshape(soc_true, [1, -1])
	# exclude nan
	valid_loc = torch.where(torch.isnan(soc_simu_vector+soc_true_vector) == False)
	soc_simu_vector = soc_simu_vector[valid_loc]
	soc_true_vector = soc_true_vector[valid_loc]

	# If desired, plot true vs predicted here
	if plot_path != "":
		visualization_utils.plot_true_vs_predicted(plot_path, soc_simu_vector, soc_true_vector)

	# modeling inefficiency
	modeling_inefficiency = torch.sum((soc_simu_vector - soc_true_vector)**2)/torch.sum((soc_true_vector - torch.mean(soc_true_vector))**2)

	# Regularization for predicted parameters using cosh
	# Encourage parameters to be around 0.5
	target_value = 0.5
	scale_factor = 10
	param_reg_loss = torch.mean(torch.cosh(scale_factor*(pred_para - target_value)) - 1)

	# Calculate the supervised losses
	l1_loss = torch.nn.functional.smooth_l1_loss(soc_simu_vector, soc_true_vector, reduction='mean')
	l2_loss = torch.nn.functional.mse_loss(soc_simu_vector, soc_true_vector, reduction='mean')

	return l1_loss, l2_loss, param_reg_loss, modeling_inefficiency
# end binns loss


#---------------------------------------------------
# simplified loss function that only takes in pred/true 
# and returns a single value (smooth l1)     
#---------------------------------------------------
def binns_loss_simple(y_pred, y_true):
	# process modeling
	soc_simu = y_pred
	# observations
	soc_true = y_true

	# flatten simulation
	soc_simu_vector = torch.reshape(soc_simu, [1, -1])
	soc_true_vector = torch.reshape(soc_true, [1, -1])

	# exclude nan
	valid_loc = torch.where(torch.isnan(soc_simu_vector+soc_true_vector) == False)
	soc_simu_vector = soc_simu_vector[valid_loc]
	soc_true_vector = soc_true_vector[valid_loc]
	loss = torch.nn.functional.smooth_l1_loss(soc_simu_vector, soc_true_vector, reduction='mean')
	return loss


#---------------------------------------------------
# NN by PyTorch
#---------------------------------------------------
# define model
class nn_model(nn.Module):
	def __init__(self, var_idx_to_emb, vertical_mixing, pos_enc):
		super().__init__()

		# Dict from categorical variable index -> Embedding layer we use
		self.var_idx_to_emb = var_idx_to_emb
		self.vertical_mixing = vertical_mixing
		self.pos_enc = pos_enc

		# List of non-categorical variable indices
		self.non_categorical_indices = list(set(list(range(len(var4nn)))).difference(var_idx_to_emb.keys()))
		self.new_input_size = len(self.non_categorical_indices)
		for idx, emb in self.var_idx_to_emb.items():
			## for embedding layer ##
			self.new_input_size += emb.embedding_dim
			# ## for one-hot encoding ##
			# self.new_input_size += emb

		# Number of parameters (output dim of MLP)
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
			self.new_input_size += self.num_params  # Add the spatial embeddings

		# Neural network layers
		# first layer
		self.l1 = nn.Linear(self.new_input_size, 128)
		# torch.nn.init.xavier_uniform_(self.l1.weight)
		# nn.init.zeros_(self.l1.bias)
		
		
		# second layer
		self.l2 = nn.Linear(128, 128)
		# torch.nn.init.xavier_uniform_(self.l2.weight)
		# nn.init.zeros_(self.l2.bias)


		# third layer
		# self.l3 = nn.Linear(128, 128)
		# torch.nn.init.xavier_uniform_(self.l3.weight)
		# nn.init.zeros_(self.l3.bias)

		# fourth layer
		self.l4 = nn.Linear(128, 128)
		# torch.nn.init.xavier_uniform_(self.l4.weight)
		# nn.init.zeros_(self.l4.bias)


		# fifth layer
		self.l5 = nn.Linear(128, self.num_params)
		# torch.nn.init.xavier_uniform_(self.l5.weight)
		# nn.init.zeros_(self.l5.bias)

		# Dropout layers
		self.dropout = nn.Dropout(0)

		# leaky relu
		self.leaky_relu = nn.LeakyReLU(negative_slope=0.3)


		# sigmoid parameters
		self.temp_sigmoid = nn.Parameter(torch.tensor(0.0), requires_grad=True)

		# sigmoid
		self.sigmoid = nn.Sigmoid()

		# hardtanh
		# self.hardtanh = nn.Hardtanh(min_val=0, max_val=1, inplace=True)

		# softsign
		# self.softsign = nn.Softsign()

		# batch normalization
		self.bn1 = nn.BatchNorm1d(128)
		self.bn2 = nn.BatchNorm1d(128)
		# self.bn3 = nn.BatchNorm1d(128)
		self.bn4 = nn.BatchNorm1d(128)

		# Initialize weights
		gain_leaky_relu = nn.init.calculate_gain('leaky_relu', 0.3)
		gain_sigmoid = nn.init.calculate_gain('sigmoid')
		nn.init.xavier_uniform_(self.l1.weight, gain=gain_leaky_relu)
		nn.init.xavier_uniform_(self.l2.weight, gain=gain_leaky_relu)
		# nn.init.xavier_uniform_(self.l3.weight, gain=gain_leaky_relu)
		nn.init.xavier_uniform_(self.l4.weight, gain=gain_leaky_relu)
		nn.init.xavier_uniform_(self.l5.weight, gain=gain_sigmoid)

		# Initialize biases
		nn.init.zeros_(self.l1.bias)
		nn.init.zeros_(self.l2.bias)
		# nn.init.zeros_(self.l3.bias)
		nn.init.zeros_(self.l4.bias)
		nn.init.zeros_(self.l5.bias)



		# # Transform from 256 to 512
		# self.transform_h1_to_h2 = nn.Linear(256, 512)
		# torch.nn.init.xavier_uniform_(self.transform_h1_to_h2.weight)
		# nn.init.zeros_(self.transform_h1_to_h2.bias)

		# # Transform from 512 to 256
		# self.transform_h3_to_h4 = nn.Linear(512, 256)
		# torch.nn.init.xavier_uniform_(self.transform_h3_to_h4.weight)
		# nn.init.zeros_(self.transform_h3_to_h4.bias)

	def forward(self, input_var, wosis_depth, coords, whether_predict, return_input=False):
		predictor = input_var[:, :, 0, 0]
		forcing = input_var[:, :, :, :]
		obs_depth = wosis_depth
		coords = coords.unsqueeze(1).detach().cpu().numpy()  # coords should be a numpy array

		# Compute embeddings for all categorical variables
		embs = []
		for idx, embedding_layer in self.var_idx_to_emb.items():
			## embedding layer ##
			emb = embedding_layer(predictor[:, idx].int())
			emb = F.normalize(emb, p=2, dim=1) # Normalize embeddings
			
			## one-hot encoding ##
			# emb = 0.1*F.one_hot(predictor[:, idx].long(), num_classes=embedding_layer)
			########################
			embs.append(emb)
		all_embs = torch.concatenate(embs, dim=1)
		new_input = torch.concatenate([predictor[:, self.non_categorical_indices], all_embs], dim=1)

		# Spatial Encoding
		if self.pos_enc == "early":
			spatial_embeddings = self.spatial_encoder(coords).squeeze(1) # Remove the channel dimension. [batch, params]
			new_input = torch.concatenate([spatial_embeddings, new_input], dim=1)

		# hidden layers
		h1 = self.l1(new_input)
		h1 = self.bn1(h1)
		h1 = self.leaky_relu(h1)
		h1 = self.dropout(h1)
		# transformed_h1 = self.transform_h1_to_h2(h1)
		h2 = self.l2(h1)
		h2 = self.bn2(h2)
		h2 = self.leaky_relu(h2) # + h1 # residual connection
		h2 = self.dropout(h2)
		# # transformed_h2 = self.transform_h2_to_h3(h2)
		# h3 = self.l3(h2)
		# h3 = self.bn3(h3)
		# h3 = self.leaky_relu(h3) # residual connection
		# h3 = self.dropout(h3)
		# transformed_h3 = self.transform_h3_to_h4(h3)
		h4 = self.l4(h2) ### remember to change back to h2 if only use 4 layers ###
		h4 = self.bn4(h4)
		h4 = self.leaky_relu(h4) # + h3 # residual connection
		h4 = self.dropout(h4)
		mlp_output = self.l5(h4)

		# Clamp temp_sigmoid to be between 10 and 200
		clamped_temp_sigmoid = 10 + 99 * self.sigmoid(self.temp_sigmoid) # try with a smaller range

		# Positional encoder correction (if using)
		if self.pos_enc == "late":
			spatial_embeddings = self.spatial_encoder(coords).squeeze(1) # Remove the channel dimension. [batch, params]
			mlp_output += spatial_embeddings

		# Pass parameters through sigmoid to constrain their range
		h5 = self.sigmoid(mlp_output / clamped_temp_sigmoid)

		# # check if h5 is nan
		# if torch.isnan(h5).any():
		# 	print("Rank {} h5 is nan".format(os.environ['RANK']))
		# elif torch.isinf(h5).any():
		# 	print("Rank {} h5 is inf".format(os.environ['RANK']))
		# parallel computing
		# biogeochemical model
		# profile_num = h5.shape[0]

		##########################
		# classic python version #
		##########################

		# simu_soc = (torch.ones((profile_num, 200))*np.nan).share_memory_()
		# args_list = [(i, h5[i], forcing[i], obs_depth[i]) for i in range(profile_num)]
		# with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count-4) as executor:
		# 	# Dispatch tasks to worker threads
		# 	for iprofile, profile_output in executor.map(fun_model_simu, args_list):
		# 		simu_soc[iprofile] = profile_output

		# # cleanup after each batch
		# gc.collect()

		#########################
		# torch multiprocessing #
		#########################

		# simu_soc = torch.Tensor(profile_num, 200).share_memory_()
		# simu_soc.fill_(np.nan)

		# # Create a list of arguments to pass to the function
		# args_list = [(i, h5[i], forcing[i], obs_depth[i]) for i in range(profile_num)]

		# with mp.Pool(processes=cpu_count-4) as pool:
		# 	# Dispatch tasks to worker threads
		# 	for iprofile, profile_output in pool.starmap(fun_model_simu, args_list):
		# 			simu_soc[iprofile] = profile_output

		##############################
		# Without Parallel Computing #
		##############################
		if whether_predict == 1:
			simu_soc = fun_model_prediction(h5, forcing, self.vertical_mixing)
		else:
			simu_soc = fun_model_simu(h5, forcing, obs_depth, self.vertical_mixing)

		if return_input:
			return simu_soc, h5, new_input
		else:
			return simu_soc, h5
# end nn_model

# Helper function to combine the training data into a single tensor
class MergeDataset(Dataset):
	def __init__(self, data_x, data_y, data_z, data_c, profile_id):
		self.data_x = data_x
		self.data_y = data_y
		self.data_z = data_z
		self.data_c = data_c
		self.profile_id = profile_id

	def __len__(self):
		return len(self.data_x)

	def __getitem__(self, idx):
		return self.data_x[idx], self.data_y[idx], self.data_z[idx], self.data_c[idx], self.profile_id[idx]


# Start training
# Filename to store average losses
avg_loss_filename = 'avg_loss_' + nn_training_name + '.txt'
avg_NSE_filename = 'avg_NSE_' + nn_training_name + '.txt'

# # Initialize the process group
# # os.environ['RANK'] = str(rank)
# # os.environ['WORLD_SIZE'] = str(world_size)
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12355'

# # Initialize distributed environment
# if dev == 'cpu':
#     dist.init_process_group('gloo', rank=rank, world_size=world_size, timeout=timedelta(hours=12))  # gloo for CPU
# else:
#     dist.init_process_group('nccl', rank=rank, world_size=world_size, timeout=timedelta(hours=12))  # gloo for CPU


# Create embeddings for categorical variables (each int maps to a different category)
var_idx_to_emb = dict()  # Column index to Embedding layer to use
for group in categorical_vars:
	n_categories = int(np.nanmax(env_info[group]) + 1)
	if args.categorical == "embedding":
		emb = nn.Embedding(num_embeddings=n_categories, embedding_dim=args.embed_dim).to(device)
	elif args.categorical == "one_hot":
		emb = n_categories  # Just store the number of categories for one-hot encoding
	for var in group:
		idx = var4nn.index(var)
		var_idx_to_emb[idx] = emb

# Initialize model
global model
if args.model == 'old_mlp':
	model_class = nn_model
	model_kwargs = {"var_idx_to_emb": var_idx_to_emb, 
					"vertical_mixing": args.vertical_mixing,
					"pos_enc": args.pos_enc}
elif args.model == 'new_mlp' or args.model == "lipmlp":
	model_class = mlp_wrapper
	model_kwargs = {"input_vars": len(var4nn),
				 	"var_idx_to_emb": var_idx_to_emb,
					"vertical_mixing": args.vertical_mixing,
					"pos_enc": args.pos_enc,
					"lipschitz": False,
					"one_hot": (args.categorical == "one_hot"),
					"use_bn": args.use_bn,
					"losses": args.losses,
					"device": device}
	if args.model == "lipmlp":
		model_kwargs["lipschitz"] = True
elif args.model == 'gcn':
	model_class = GCN_BINN
	model_kwargs = {"input_vars": len(var4nn),
				 	"var_idx_to_emb": var_idx_to_emb,
					"vertical_mixing": args.vertical_mixing,
					"pos_enc": args.pos_enc,
					"k": args.k,
					"one_hot": (args.categorical == "one_hot"),
					"losses": args.losses,
					"device": device}
else:
	raise ValueError("Invalid args.model")


if args.loss_weighting not in ["manual", "two_stage", "relobralo"]:
	weighting = weighting_method.__dict__[args.loss_weighting]

	# Attempt to match the MTLmodel API. Create a class that derives from
 	# both our architecture class (customized for BINN) and our weighting method
  	# class (provided by LibMTL)
	class MTLmodel(model_class, weighting):
		def __init__(self, **kwargs):
			super(MTLmodel, self).__init__(**kwargs)
			self.init_param()

	model = MTLmodel(**model_kwargs).to(device)
else:
	model = model_class(**model_kwargs).to(device)

# Create distributed version of the model
if args.use_ddp == 1:
	model = DDP(model)
	model_without_ddp = model.module
else:
	model_without_ddp = model

# Loss and optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
if args.use_swa:
	swa_model = AveragedModel(model)
	# scheduler = CosineAnnealingLR(optimizer, T_max=20, verbose=True)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr, gamma=0.1)
	swa_start = 25
	swa_scheduler = SWALR(optimizer, swa_lr=args.lr)
else:
	# Add a learning rate scheduler that decreases the learning rate by a factor of 0.1 every 50 epochs
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr, gamma=0.1)

if args.whether_resume == 1:
	# Load the model from the checkpoint
	checkpoint_worker = torch.load(checkpoint_path, map_location=device)
	state_dict = checkpoint_worker['model_state_dict']
	model.load_state_dict(state_dict)
	optimizer.load_state_dict(checkpoint_worker['optimizer_state_dict'])
	if args.use_swa:
		swa_model.load_state_dict(checkpoint_worker['swa_model_state_dict'])


# Loss function
fun_loss = binns_loss

# Initialize datasets
train_dataset = MergeDataset(train_x, train_y, train_z, train_c, train_profile_id)
val_dataset = MergeDataset(val_x, val_y, val_z, val_c, val_profile_id)

# # Use DistributedSampler for distributed training
# train_sampler = DistributedSampler(train_dataset)
# val_sampler = DistributedSampler(val_dataset)

# Data loaders with DistributedSampler
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=thread_count)  #, sampler=train_sampler)  # Set drop_last=True to avoid 1-example batches during training, which causes error with BatchNorm https://stackoverflow.com/questions/65882526/expected-more-than-1-value-per-channel-when-training-got-input-size-torch-size
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=thread_count)  #, sampler=val_sampler)

# training and validation loop
num_epoch = args.n_epochs

if args.whether_resume == 0:
	# record the loss history
	train_loss_history = {loss: np.ones((num_epoch, 1))*np.nan for loss in args.losses}
	val_loss_history = {loss: np.ones((num_epoch, 1))*np.nan for loss in args.losses}
	val_NSE_history = np.ones((num_epoch, 1))*np.nan
	best_model_epoch = torch.tensor(0) # epoch with the best model so far

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

	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/nn_obs_soc_' + job_id + '.csv', binn_obs_soc, delimiter = ',')

	# try to save the predicted parameters before training
	val_pred_soc = torch.tensor(np.ones((wosis_profile_info.shape[0], 200))*np.nan, dtype = torch.float32, device=device)
	val_pred_para = torch.tensor(np.ones((wosis_profile_info.shape[0], len(para_names)))*np.nan, dtype = torch.float32, device=device)
	model.eval()
	with torch.no_grad():
		temp_SOC, temp_pred_para = model(val_x.to(device), val_z.to(device), val_c.to(device), whether_predict=0)
	val_pred_soc[val_profile_id, :] = temp_SOC.detach()
	val_pred_para[val_profile_id, :] = temp_pred_para.detach()
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/model_training_history/nn_val_pred_soc_' + job_id + "_initial" + '.csv', val_pred_soc.detach().cpu().numpy(), delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/model_parameters/nn_val_pred_soc_' + job_id + "_initial" + '.csv', val_pred_para.detach().cpu().numpy(), delimiter = ',')

else: 
	# record the loss history
	train_loss_history = checkpoint_worker['train_loss_history']
	val_loss_history = checkpoint_worker['val_loss_history']
	val_NSE_history = checkpoint_worker['val_NSE_history']
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

# print the model structure
print(model)

# record start time
start_time = time.time()
whether_break = torch.tensor(0).to(device)  # Set this flag in case we have 0 training epochs

for iepoch in range(start_epoch, num_epoch):
	epoch_start = time.time()

	# Initialize the break flag for this epoch
	whether_break = torch.tensor(0).to(device)

	# Initialize the storage for the predicted parameters
	val_pred_soc = torch.tensor(np.ones((wosis_profile_info.shape[0], 200))*np.nan, dtype = torch.float32, device=device)
	val_pred_para = torch.tensor(np.ones((wosis_profile_info.shape[0], len(para_names)))*np.nan, dtype = torch.float32, device=device)
	train_pred_para = torch.tensor(np.ones((wosis_profile_info.shape[0], len(para_names)))*np.nan, dtype = torch.float32, device=device)

	# clear gradients
	optimizer.zero_grad()
	model.zero_grad()

	# -------------------------------------training
	loss_record_train = {loss: list() for loss in args.losses}
	NSE_record_train = list()
	ibatch = 0
	model.train()
	# train_loader.sampler.set_epoch(iepoch)  # Set sampler's epoch number, so we use a different order per epoch

	batch_start = time.time()
	#torch.autograd.set_detect_anomaly(True) <- helps debug gradient anomalies but is VERY SLOW
	for batch_info in train_loader:
		batch_x, batch_y, batch_z, batch_c, batch_profile_id = batch_info

		if batch_x.shape[0] == 1 and args.use_bn:  # Batch size of 1 during training does not work with BatchNorm
			continue

		ibatch = ibatch + 1
		batch_x = batch_x.to(device)
		batch_y = batch_y.to(device)
		batch_c = batch_c.to(device)

		#------------ 1 forward
		# train_nn_start = time.time()
		batch_y_hat, batch_pred_para = model(batch_x, batch_z, batch_c, whether_predict=0)
		# train_nn_end = time.time()
		# print("Forward time", train_nn_end-train_nn_start)

		# Check if batch_pred_para is nan or inf
		if torch.isnan(batch_pred_para).any() or torch.isinf(batch_pred_para).any():
			whether_break = torch.tensor(1).to(device)
			for ipara in range(batch_pred_para.shape[0]):
				if torch.isnan(batch_pred_para[ipara]).any() or torch.isinf(batch_pred_para[ipara]).any():
					print(f"Epoch {iepoch} batch {ibatch} parameter {ipara} is {batch_pred_para[ipara]}")

		# Check for extreme para values
		if torch.any(batch_pred_para < 0.00001) or torch.any(batch_pred_para > 0.99999):
			print("Extreme param values")
			print(batch_pred_para)
		# process = psutil.Process()
		# print("Train memory usage", process.memory_info().rss / 1e9, "GB")  # in bytes

		#------------ 2 compute the objective function
		# train_loss_start = time.time()
		smooth_l1_loss, l2_loss, param_reg_loss, train_NSE = fun_loss(batch_y_hat, batch_y, batch_pred_para)
		# train_loss_end = time.time()

		# Compute additional losses if using. If we are not using them, set them to nan
		lipmlp_loss = np.nan
		spectral_loss = np.nan
		# c_reg_loss = np.nan
		cure_loss = np.nan

		# Lipschitz loss if using
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

		if "cure" in args.losses:
			cure_loss, grad_norm = misc_utils.regularizer(batch_x, batch_y, batch_z, model, binns_loss_simple)

		#------------ 3 cleaning gradients
		optimizer.zero_grad()

		# Compute gradients wrt process parameters
		# print("Para shape", batch_pred_para.shape)  # [batch, num_para]
		# gradients = torch.autograd.grad(outputs=obj, inputs=batch_pred_para,
		# 									grad_outputs=torch.ones(obj.size()).to(device), 
		# 								create_graph=True, retain_graph=True)[0]
		# print("Gradients shape", gradients.shape)  # [batch, num_para]
		# print(gradients)

		#------------ 4 accumulate partical derivatives of objective respect to parameters
		# backward_start = time.time()
		loss_dict = {"l1": smooth_l1_loss,
			   		 "l2": l2_loss,
					 "param_reg": param_reg_loss,
					 "lipmlp": lipmlp_loss,
					 # "c_reg": c_reg_loss,
					 "spectral": spectral_loss,
					 "cure": cure_loss}
		
		if args.loss_weighting in ["manual", "two_stage", "relobralo"]:
			total_loss = 0.
			for idx, loss in enumerate(args.losses):
				total_loss = total_loss + args.lambdas[idx] * loss_dict[loss]
			total_loss.backward()
		else:
			train_losses = torch.zeros(len(args.losses)).to(device)  # Store all losses in a tensor
			for loss_idx, loss in enumerate(args.losses):
				train_losses[loss_idx] = loss_dict[loss]
			alphas = model.backward(train_losses)
			if ibatch == 1:
				print("Alphas", alphas)

		# # Compute gradient w.r.t. last MLP layer's weights (which output the parameters)
		# print("L5 grad", model_without_ddp.mlp.layer_output.weight.grad.shape, model_without_ddp.mlp.layer_output.weight.grad[:, 0])

		# Check grads
		# print("Layer output grad", model_without_ddp.mlp.layer_output.weight.grad)
		# print("First layer", model_without_ddp.mlp.layers[0].weight.grad)

		# # Fetch weight & gradient of model param
		# if torch.any(torch.isnan(model_without_ddp.mlp.layer_output.weight)):
		# 	print("Weight was nan")
		# 	print("Weight", model_without_ddp.mlp.layer_output.weight)
		# 	exit(1)
		# if torch.any(torch.isnan(model_without_ddp.mlp.layer_output.weight.grad)):
		# 	print("Grad was nan")
		# 	print("grad", model_without_ddp.mlp.layer_output.weight.grad)
		# 	exit(1)

		# clip gradients
		# torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)
		# torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=args.clip_value)

		#------------ 5 step in the opposite direction of the gradient
		# optimizer_start = time.time()
		optimizer.step()
		# optimizer_end = time.time()

		for loss in args.losses:
			loss_record_train[loss].append(loss_dict[loss].item())
		NSE_record_train.append(train_NSE.item())

		# flush all printed output
		sys.stdout.flush()
		batch_start = time.time()

	# end for batch_info in train_loader:

	# print(f"Process {rank} finished epoch {iepoch}")

	# Ensure all processes reach this point to synchronize
	# Use all_reduce to check if any process has encountered NaN
	# dist.all_reduce(whether_break, op=dist.ReduceOp.MAX)

	# Check if batch_pred_para is nan or inf
	if whether_break.item() == 1:
		print(f"Breaking after epoch {iepoch}")
		break  # Break out of the epoch loop if NaN detected in any process

	# training time
	train_time = time.time() - epoch_start
	
	# print(f'Epoch {iepoch + 1}, Rank {rank}, train loss: {torch.tensor(loss_record_train).mean():.1f}, time: {(time.time()-epoch_start):.2f}')
	# print(f"-----------------Epoch {iepoch + 1} - Rank {rank} - Model Weights: {model_without_ddp.l1.weight.data} - {model_without_ddp.l2.weight.data} - {model_without_ddp.l3.weight.data} - {model_without_ddp.l4.weight.data} - {model_without_ddp.l5.weight.data}-----------------")


	# writer.add_scalar('training loss', torch.tensor(loss_record_train).mean(), iepoch+1)
	# Ensure all processes reach this point before proceeding
	# dist.barrier()
	# print(f"Process {rank} passed barrier. Epoch {iepoch}")

	# -------------------------------------validation
	loss_record_val = {loss: list() for loss in args.losses}
	NSE_record_val = list()
	ibatch = 0
	model.eval()
	batch_start = time.time()
	for batch_info in val_loader:
		batch_x, batch_y, batch_z, batch_c, batch_profile_id = batch_info
		ibatch = ibatch + 1
		batch_x = batch_x.to(device)
		batch_y = batch_y.to(device)
		# 1 forward
		with torch.no_grad():
			batch_y_hat, batch_pred_para = model(batch_x, batch_z, batch_c, whether_predict=0)

		# 2 compute the objective function
		l1_loss, l2_loss, param_reg_loss, val_NSE = fun_loss(batch_y_hat, batch_y, batch_pred_para)

		# Compute additional losses if using. Not strictly necessary but this helps us see if there
  		# is a difference between the losses for train/validation sets
		# If we are not using them, set them to nan
		lipmlp_loss = np.nan
		spectral_loss = np.nan
		# c_reg_loss = np.nan
		cure_loss = np.nan
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
		if "cure" in args.losses:
			cure_loss, grad_norm = misc_utils.regularizer(batch_x, batch_y, batch_z, model, binns_loss_simple)
		loss_dict = {"l1": smooth_l1_loss,
			   		 "l2": l2_loss,
					 "param_reg": param_reg_loss,
					 "lipmlp": lipmlp_loss,
					 # "c_reg": c_reg_loss,
					 "spectral": spectral_loss,
					 "cure": cure_loss}
		for loss in args.losses:
			if np.isnan(loss_dict[loss].item()):
				print(batch_y_hat)
			loss_record_val[loss].append(loss_dict[loss])
		NSE_record_val.append(val_NSE.item())
	# end for batch_info in val_loader: 

	# record the time
	hist_time = time.time() - start_time

	# # print time
	# print('Training time for rank {}: {:.5f}'.format(rank, train_time))
	# print('Backward time for rank {}: {:.5f}'.format(rank, backward_end - backward_start))
	# print('Optimizer time for rank {}: {:.5f}'.format(rank, optimizer_end - optimizer_start))

	# # Gather losses from all processes
	# all_train_losses = {loss: [torch.tensor(0.0, device=device) for _ in range(world_size)] for loss in args.losses}
	# all_val_losses = {loss: [torch.tensor(0.0, device=device) for _ in range(world_size)] for loss in args.losses}
	# all_train_times = [torch.tensor(0.0, device=device) for _ in range(world_size)]
	# all_train_NSE = [torch.tensor(0.0, device=device) for _ in range(world_size)]
	# all_val_NSE = [torch.tensor(0.0, device=device) for _ in range(world_size)]
	# all_hist_times = [torch.tensor(0.0, device=device) for _ in range(world_size)]

	# # Gather validation parameters predictions from all processes
	# for loss in args.losses:
 	# 	 dist.all_gather(all_train_losses[loss], torch.tensor(loss_record_train[loss], device=device).mean())
	# 	 dist.all_gather(all_val_losses[loss], torch.tensor(loss_record_val[loss], device=device).mean())
	# dist.all_gather(all_train_times, torch.tensor(train_time, device=device))
	# dist.all_gather(all_train_NSE, torch.tensor(NSE_record_train, device=device).mean())
	# dist.all_gather(all_val_NSE, torch.tensor(NSE_record_val, device=device).mean())
	# dist.all_gather(all_hist_times, torch.tensor(hist_time, device=device))

	# record the loss history
	for loss in args.losses:
		train_loss_history[loss][iepoch, :] = torch.tensor(loss_record_train[loss]).mean().detach().cpu().numpy()
		val_loss_history[loss][iepoch, :] = torch.tensor(loss_record_val[loss]).mean().detach().cpu().numpy()
	val_NSE_history[iepoch, :] = torch.tensor(NSE_record_val).mean().detach().cpu().numpy()

	# writer.add_scalars('loss', {'training': torch.stack(all_train_losses).mean(), 'validation': torch.stack(all_val_losses).mean()}, iepoch+1)
	train_losses_epoch = {loss: round(train_loss_history[loss][iepoch, 0], 2) for loss in args.losses}
	val_losses_epoch = {loss: round(val_loss_history[loss][iepoch, 0], 2) for loss in args.losses}

	print(f'Epoch {iepoch}, train losses: {train_losses_epoch}, validation loss: {val_losses_epoch}, time: {train_time:.2f}')
	print(f'Epoch {iepoch}, train NSE: {torch.tensor(NSE_record_train).mean():.2f}, validation NSE: {torch.tensor(NSE_record_val).mean():.2f}, time: {train_time:.2f}', flush=True)

	# Relobralo update
	if args.loss_weighting == "relobralo" and iepoch >= 1:
		with torch.no_grad():
			loss_curr = torch.tensor([train_loss_history[loss][iepoch, 0] for loss in args.losses])
			loss_prev = torch.tensor([train_loss_history[loss][iepoch-1, 0] for loss in args.losses])
			loss_init = torch.tensor([train_loss_history[loss][0, 0] for loss in args.losses])
			lambda_bal_prev = F.softmax(loss_curr / (args.relobralo_temp * loss_prev), dim=0)  # Based on ratio of current loss & prev epoch loss
			print("Lambda bal prev", lambda_bal_prev)
			lambda_bal_init = F.softmax(loss_curr / (args.relobralo_temp * loss_init), dim=0)  # Based on ratio of current loss & epoch 0 loss
			lambda_hist = args.relobralo_saudade * args.lambdas + (1-args.relobralo_saudade) * lambda_bal_init
			args.lambdas = args.relobralo_alpha * lambda_hist + (1-args.relobralo_alpha) * lambda_bal_prev
			args.lambdas = args.lambdas / args.lambdas.sum()
			print("lambdas", args.lambdas)

	# if args.loss_weighting == "relobralo_modified" and iepoch >= 1:
	# 	with torch.no_grad():
	# 		loss_curr = torch.tensor([train_loss_history[loss][iepoch, 0] for loss in args.losses])
	# 		loss_prev = torch.tensor([train_loss_history[loss][iepoch-1, 0] for loss in args.losses])
	# 		loss_init = torch.tensor([train_loss_history[loss][0, 0] for loss in args.losses])
	# 		lambda_bal_prev = F.softmax(loss_curr / (args.relobralo_temp * loss_prev), dim=0)  # Based on ratio of current loss & prev epoch loss
	# 		lambda_hist = args.relobralo_saudade * args.lambdas + (1-args.relobralo_saudade) * lambda_bal_init
	# 		args.lambdas = args.relobralo_alpha * lambda_hist + (1-args.relobralo_alpha) * lambda_bal_prev

	# elif rank == 1:
	# 	with open(os.path.join(data_dir_output, "neural_network", job_id, avg_loss_filename), "a") as f:
	# 		f.write(f'{iepoch + 1}, {torch.stack(all_train_losses).mean():.6f}, {torch.stack(all_val_losses).mean():.6f}, {torch.stack(all_train_times).mean():.2f}, {torch.stack(all_hist_times).mean():.2f}\n')
	# elif rank == 4: 
	# 	with open(os.path.join(data_dir_output, "neural_network", job_id, avg_NSE_filename), "a") as f:
	# 		f.write(f'{iepoch + 1}, {torch.stack(all_train_NSE).mean():.6f}, {torch.stack(all_val_NSE).mean():.6f}, {torch.stack(all_train_times).mean():.2f}, {torch.stack(all_hist_times).mean():.2f}\n')
	
	######################
	## Para per 2 epoch ##
	######################
	# elif rank == 5:
	# # 	# 	print(f"Epoch {iepoch+1}, Clamped Sigmoid Parameters: {sigmoid_para_val.item():.2f}")
	# # 	# try to track the parameters change during training process
	# # 	# if iepoch in [0, 5, 10, 15, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000]:
	# 	if iepoch % 5 == 0:
	# 		eval_start_time = time.time()
	# 		model.eval()
	# 		# print("Starting time to predict parameters: {}".format(datetime.now()))
	# 		with torch.no_grad():
	# 			temp_soc_simu, temp_pred_para = model(val_x.to(device), val_z.to(device), whether_predict=0)
	# 		# save validation parameters 
	# 		val_pred_soc[val_profile_id, :] = temp_soc_simu.detach().cpu()
	# 		val_pred_para[val_profile_id, :] = temp_pred_para.detach().cpu()
	# 		# print("Ending time to predict parameters: {}".format(datetime.now()))
	# 		# save data
	# 		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/model_training_history/nn_val_pred_soc_' + job_id + "_" + str(iepoch) + '.csv', val_pred_soc.detach().cpu().numpy(), delimiter = ',')
	# 		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/model_parameters/nn_val_pred_para_' + job_id + "_" + str(iepoch) + '.csv', val_pred_para.detach().cpu().numpy(), delimiter = ',')
	# 		# print time
	# 		print('Epoch {} - Rank {}: {:.5f}'.format(iepoch, rank, time.time() - eval_start_time))
		

	# 	# elif rank == 6:
	# 	# try to track the parameters change during training process
	# 	if iepoch in [0, 5, 10, 15, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900]:
	# 		model.eval()
	# 		with torch.no_grad():
	# 			temp_train_soc_simu, temp_train_pred_para = model(train_x.to(device), train_z.to(device))
	# 		# save validation parameters
	# 		train_pred_para[train_profile_id, :] = temp_train_pred_para.detach().cpu()
	# 		# save data
	# 		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/nn_train_pred_para_' + job_id + "_" + str(iepoch) + '.csv', train_pred_para.detach().cpu().numpy(), delimiter = ',')


	if val_NSE_history[iepoch, :] <= best_val_NSE:  # @joshuafan: removed the iepoch==0 condition
		print(f'Best model updated at epoch {iepoch}')
		best_model_epoch = torch.tensor(iepoch, device=device)

		# save best model
		checkpoint_best_model = {
			'epoch': iepoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'best_val_loss': best_val_loss,
			'best_val_NSE': best_val_NSE,
			'best_model_epoch': best_model_epoch,
			'train_loss_history': train_loss_history,
			'val_loss_history': val_loss_history,
			'val_NSE_history': val_NSE_history,
			'train_indices': train_loc,
			'val_indices': val_loc,
			'test_indices': test_loc,
			'epochs_without_improvement': epochs_without_improvement,
		}
		if args.use_swa:
			checkpoint_best_model['swa_model_state_dict'] = swa_model.state_dict()
		
		best_model_path = data_dir_output + 'neural_network/' + job_id + '/opt_nn_' + job_id  + '.pt'
		torch.save(checkpoint_best_model, best_model_path)

		# # run the model to predict the parameters with val data and save the results
		# eval_start_time = time.time()
		# with torch.no_grad():
		# 	temp_soc_simu, temp_pred_para = model(val_x.to(device), val_z.to(device), whether_predict=0)
		# 	grid_simu_soc, grid_pred_para = model(torch.tensor(predict_data_x, dtype=torch.float32, device=device), torch.tensor(predict_data_z, dtype=torch.float32, device=device), whether_predict = 1)
		# # save validation parameters 
		# val_pred_soc[val_profile_id, :] = temp_soc_simu.detach()
		# val_pred_para[val_profile_id, :] = temp_pred_para.detach()
		# # print("Ending time to predict parameters: {}".format(datetime.now()))
		# # save data
		# np.savetxt(data_dir_output + 'neural_network/' + job_id + '/model_training_history/nn_val_pred_soc_' + job_id + "_" + str(iepoch) + '.csv', val_pred_soc.detach().cpu().numpy(), delimiter = ',')
		# np.savetxt(data_dir_output + 'neural_network/' + job_id + '/model_parameters/nn_val_pred_para_' + job_id + "_" + str(iepoch) + '.csv', val_pred_para.detach().cpu().numpy(), delimiter = ',')

		# @joshuafan temporarily removed
		# # Bulk simulation for the grid data
		# carbon_input_best, cpool_steady_state_best, cpools_layer_best, soc_layer_best, total_res_time_best, \
		# total_res_time_base_best, res_time_base_pools_best, t_scaler_best, bulk_A_best, \
		# w_scaler_best, bulk_K_best, bulk_V_best, bulk_xi_best, bulk_I_best, litter_fraction_best = fun_bulk_simu(grid_pred_para.to(device), \
		#                                                                                             torch.tensor(predict_data_x, dtype=torch.float32, device=device), \
		#                                                                                                 torch.tensor(predict_data_z, dtype=torch.float32, device=device),
		#                                                                                                 args.vertical_mixing)
		# # save the bulk simulation results
		# np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Bulk_Simulation/nn_bulk_simu_carbon_input_' + job_id + "_" + str(iepoch) + '.csv', carbon_input_best.detach().cpu().numpy(), delimiter = ',')
		# np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Bulk_Simulation/nn_bulk_simu_cpool_steady_state_' + job_id + "_" + str(iepoch) + '.csv', cpool_steady_state_best.detach().cpu().numpy(), delimiter = ',')
		# np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Bulk_Simulation/nn_bulk_simu_cpools_layer_' + job_id + "_" + str(iepoch) + '.csv', cpools_layer_best.detach().cpu().numpy(), delimiter = ',')
		# np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Bulk_Simulation/nn_bulk_simu_soc_layer_' + job_id + "_" + str(iepoch) + '.csv', soc_layer_best.detach().cpu().numpy(), delimiter = ',')
		# np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Bulk_Simulation/nn_bulk_simu_total_res_time_' + job_id + "_" + str(iepoch) + '.csv', total_res_time_best.detach().cpu().numpy(), delimiter = ',')
		# np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Bulk_Simulation/nn_bulk_simu_total_res_time_base_' + job_id + "_" + str(iepoch) + '.csv', total_res_time_base_best.detach().cpu().numpy(), delimiter = ',')
		# np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Bulk_Simulation/nn_bulk_simu_res_time_base_pools_' + job_id + "_" + str(iepoch) + '.csv', res_time_base_pools_best.detach().cpu().numpy(), delimiter = ',')
		# np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Bulk_Simulation/nn_bulk_simu_t_scaler_' + job_id + "_" + str(iepoch) + '.csv', t_scaler_best.detach().cpu().numpy(), delimiter = ',')
		# np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Bulk_Simulation/nn_bulk_simu_bulk_A_' + job_id + "_" + str(iepoch) + '.csv', bulk_A_best.detach().cpu().numpy(), delimiter = ',')
		# np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Bulk_Simulation/nn_bulk_simu_w_scaler_' + job_id + "_" + str(iepoch) + '.csv', w_scaler_best.detach().cpu().numpy(), delimiter = ',')
		# np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Bulk_Simulation/nn_bulk_simu_bulk_K_' + job_id + "_" + str(iepoch) + '.csv', bulk_K_best.detach().cpu().numpy(), delimiter = ',')
		# np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Bulk_Simulation/nn_bulk_simu_bulk_V_' + job_id + "_" + str(iepoch) + '.csv', bulk_V_best.detach().cpu().numpy(), delimiter = ',')
		# np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Bulk_Simulation/nn_bulk_simu_bulk_xi_' + job_id + "_" + str(iepoch) + '.csv', bulk_xi_best.detach().cpu().numpy(), delimiter = ',')
		# np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Bulk_Simulation/nn_bulk_simu_bulk_I_' + job_id + "_" + str(iepoch) + '.csv', bulk_I_best.detach().cpu().numpy(), delimiter = ',')
		# np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Bulk_Simulation/nn_bulk_simu_litter_fraction_' + job_id + "_" + str(iepoch) + '.csv', litter_fraction_best.detach().cpu().numpy(), delimiter = ',')

		# print('Epoch {} finish evaluating the best model: {:.5f}'.format(iepoch, time.time() - eval_start_time))
		
		
		# np.savetxt(data_dir_output + 'neural_network/val_loss_history_' + time_stamp + '.csv', val_loss_history, delimiter = ',')
		# np.savetxt(data_dir_output + 'neural_network/train_loss_history_' + time_stamp + '.csv', train_loss_history, delimiter = ',')
	# end if iepoch == 0:
	
	# save the training and validation loss history
	loss_file = os.path.join(data_dir_output, "neural_network", job_id, avg_loss_filename)
	if iepoch == 0:  # Write the header if the file doesn't exist yet
		with open(loss_file, mode='w') as f:
			csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			csv_writer.writerow(['epoch'] + [f'{loss}_loss_train' for loss in args.losses] + 
								[f'{loss}_loss_val' for loss in args.losses] +
								['epoch_time', 'cumulative_time', 'best_model_epoch'])
	with open(loss_file, mode='a+') as f:
		csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		best_model_path = data_dir_output + 'neural_network/' + job_id + '/opt_nn_' + job_id + '.pt'
		csv_writer.writerow([iepoch] + [train_loss_history[loss][iepoch, 0] for loss in args.losses] +
							[val_loss_history[loss][iepoch, 0] for loss in args.losses] + 
							[round(train_time, 2), round(hist_time, 2),best_model_epoch.item()])
	
	# NSE file
	with open(os.path.join(data_dir_output, "neural_network", job_id, avg_NSE_filename), "a") as f:
		f.write(f'{iepoch}, {torch.tensor(NSE_record_train).mean():.6f}, {torch.tensor(NSE_record_val).mean():.6f}, {train_time:.2f}, {hist_time:.2f}, {best_model_epoch.item()}\n')
	
	# Ensure all processes reach this point before proceeding
	# dist.barrier()

	# Add a learning rate scheduler
	if args.use_swa and val_NSE.item() < 0.5 and iepoch > swa_start:
		swa_model.update_parameters(model)
		swa_scheduler.step()
	else:
		scheduler.step()

	if args.loss_weighting == "two_stage" and iepoch > args.second_start:
		args.lambdas = args.second_lambdas

	# Add a early stopping condition
	if val_NSE_history[iepoch, :] < best_val_NSE:
		best_val_NSE = val_NSE_history[iepoch, :]
		best_val_loss = val_loss_history[args.losses[0]][iepoch, :]
		epochs_without_improvement = 0
		# Optionally save the model here if it's the best one so far
	else:
		epochs_without_improvement += 1

	# Early stopping condition
	if epochs_without_improvement == patience:
		print("Early stopping due to no improvement after {} epochs.".format(patience))
		break  # exit the epoch loop
	print("Total epoch time", time.time() - epoch_start)

	# If runtimes are over 10*11.30 hours, save checkpoint and exit
	whether_checkpoint = False
	if time.time() - job_begin_time > 41400*10:
		print("Runtime exceeded, saving checkpoint and exiting.")
		# break # at this point, no longer pass the time limit
		whether_checkpoint = True
		checkpoint = {
			'epoch': iepoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'best_val_loss': best_val_loss,
			'best_val_NSE': best_val_NSE,
			'best_model_epoch': best_model_epoch,
			'train_loss_history': train_loss_history,
			'val_loss_history': val_loss_history,
			'val_NSE_history': val_NSE_history,
			'train_indices': train_loc,
			'val_indices': val_loc,
			'test_indices': test_loc,
			'epochs_without_improvement': epochs_without_improvement,
		}
		if args.use_swa:
			checkpoint['swa_model_state_dict'] = swa_model.state_dict()

		torch.save(checkpoint, data_dir_output + 'neural_network/' + job_id + '/checkpoint_' + job_id + '.pt')


		if args.scheduler == 'slurm':
			# Create a file to submit the job again
			with open(job_submit_path + 'Resume' + job_id + '.submit', 'w') as f:
				f.write(f'#!/bin/bash\n')
				f.write(f'#SBATCH -p aida\n')
				f.write(f'#SBATCH -J binn_resume\n')
				f.write(f'#SBATCH -c {args.num_CPU*2}\n')
				f.write(f'#SBATCH -N 1 -n 1\n')
				f.write(f'#SBATCH --mem=50GB\n')
				f.write(f'#SBATCH -t 12:00:00\n')

				f.write(f'source ~/.bashrc\n')
				f.write(f'module load cuda\n')
				f.write(f'conda activate binn\n\n')
				f.write(f'python {" ".join(sys.argv)} --whether_resume 1\n')

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
			# Create a file to submit the job again
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
				f.write(f'python -u /glade/u/home/haodixu/BINN/Server_Script/binns_DDP.py --lr ' + str(args.lr) + ' --weight_decay ' + str(args.weight_decay) + ' --batch_size ' + str(args.batch_size) + \
					' --seed ' + str(args.seed) + ' --n_epochs ' + str(args.n_epochs) + ' --patience ' + str(args.patience) + ' --model ' + str(args.model) + ' --lambda_lipschitz ' + str(args.lambda_lipschitz) + \
					' --note ' + str(args.note) + ' --categorical ' + str(args.categorical) + ' --use_bn ' + ' --embed_dim ' + str(args.embed_dim) + ' --num_CPU ' + str(args.num_CPU) + ' --whether_resume 1\n')

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





# # Ensure all processes reach the end
# dist.barrier()

print("FINISHED TRAINING - now making visualizations")



##################################################
# prediction bv best trained model
##################################################
new_checkpoint = torch.load(data_dir_output + 'neural_network/' + job_id + '/opt_nn_' + job_id + '.pt', map_location=device)
if args.use_swa:
	# Update batchnorm stats of averaged model (required when using SWA)
	misc_utils.update_bn_custom(train_loader, swa_model, device) 
	best_guess_model = swa_model
	best_guess_model.load_state_dict(new_checkpoint['swa_model_state_dict'])
else:
	best_guess_model = model  # Do not need to create a new model
	best_guess_model.load_state_dict(new_checkpoint['model_state_dict'])
print("Loaded model")

with torch.no_grad():
	# Plot loss curves throughout training. Normalize each curve relative to its mean,
	# to make the scales comparable
	for loss in train_loss_history:  # Remove nans first
		train_loss_history[loss] = train_loss_history[loss][~np.isnan(train_loss_history[loss])]
		val_loss_history[loss] = val_loss_history[loss][~np.isnan(val_loss_history[loss])]
	losses = [(train_loss_history[loss] / train_loss_history[loss].mean()).flatten().tolist() for loss in args.losses] + \
		 	 [(val_loss_history[loss] / val_loss_history[loss].mean()).flatten().tolist() for loss in args.losses]
	labels = [f"{loss} loss (train)" for loss in args.losses] + [f"{loss} loss (val)" for loss in args.losses]
	visualization_utils.plot_losses(os.path.join(PLOT_DIR, "losses.png"), losses, labels)

	if whether_break.item() == 1:
		print(f"exiting after training due to NaN encountered in any process.")
		exit(1)

	print("Rank 0 beginning prediction at time {}".format(datetime.now()))
	best_guess_model.eval()
	print("Rank 0 model set to eval at time {}".format(datetime.now()))

	# Get predictions for train examples, compute loss & plot
	best_guess_train_y_hat, best_guess_train_pred_para = best_guess_model(train_x.to(device), train_z.to(device), train_c.to(device), whether_predict=0)
	train_l1_loss, _, _, train_NSE = fun_loss(best_guess_train_y_hat, train_y.to(device), best_guess_train_pred_para, 
												plot_path=os.path.join(PLOT_DIR, "true_vs_predicted_train.png"))
	print(f'Train loss: {train_l1_loss.item():.2f}, Train NSE: {train_NSE.item():.2f}')

	# Get predictions for validation examples, compute loss & plot
	best_guess_val_y_hat, best_guess_val_pred_para = best_guess_model(val_x.to(device), val_z.to(device), val_c.to(device), whether_predict=0)
	val_l1_loss, _, _, val_NSE = fun_loss(best_guess_val_y_hat, val_y.to(device), best_guess_val_pred_para, 
											plot_path=os.path.join(PLOT_DIR, "true_vs_predicted_val.png"))
	print(f'Val loss: {val_l1_loss.item():.2f}, Val NSE: {val_NSE.item():.2f}')

	if test_split_ratio != 0:
		best_guess_test_y_hat, best_guess_test_pred_para = best_guess_model(test_x.to(device), test_z.to(device), test_c.to(device), whether_predict=0)
		test_loss, _, _, test_NSE = fun_loss(best_guess_test_y_hat, test_y.to(device), best_guess_test_pred_para, 
										plot_path=os.path.join(PLOT_DIR, "true_vs_predicted_test.png"))
		print(f'Test loss: {test_loss.item():.2f}, Test NSE: {test_NSE.item():.2f}')

	# @joshuafan: Summary csv file of all results. Create this if it doesn't exist
	if args.n_epochs >= 1:
		results_summary_file = os.path.join(data_dir_output, "neural_network/results_summary.csv")
		if not os.path.isfile(results_summary_file):
			with open(results_summary_file, mode='w') as f:
				csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
				csv_writer.writerow(['job_id', 'command', 'lr', 'weight_decay', 'seed', 'model_path', 'best_val_NSE', 'best_val_loss', 'test_NSE', 'test_loss'])
		command_string = " ".join(sys.argv)

		# Add a row to the summary csv file
		with open(results_summary_file, mode='a+') as f:
			csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			best_model_path = data_dir_output + 'neural_network/' + job_id + '/opt_nn_' + job_id  + '.pt'
			csv_writer.writerow([job_id, command_string, args.lr, args.weight_decay, args.seed, best_model_path, best_val_NSE.item(), best_val_loss.item(), test_NSE.item(), test_loss.item()])
		
	# create folder for the results
	os.makedirs(data_dir_output + 'neural_network/' + job_id + '/Validation', exist_ok=True)
	os.makedirs(data_dir_output + 'neural_network/' + job_id + '/Train', exist_ok=True)
	if test_split_ratio != 0:
		os.makedirs(data_dir_output + 'neural_network/' + job_id + '/Test', exist_ok=True)

	#############
	# Test Data #
	#############

	## predictions and parameters for the test profiles ##
	best_simu_soc = torch.tensor(np.ones((wosis_profile_info.shape[0], 200))*np.nan, dtype = torch.float32, device=device)
	best_pred_para = torch.tensor(np.ones((wosis_profile_info.shape[0], len(para_names)))*np.nan, dtype = torch.float32, device=device)
	upper_depth_all = np.ones((wosis_profile_info.shape[0], 200))*np.nan
	lower_depth_all = np.ones((wosis_profile_info.shape[0], 200))*np.nan


	upper_depth_all[current_data_profile_id, :] = obs_upper_depth_matrix
	lower_depth_all[current_data_profile_id, :] = obs_lower_depth_matrix

	best_simu_soc[test_profile_id, :] = best_guess_test_y_hat
	best_pred_para[test_profile_id, :] = best_guess_test_pred_para


	# save data
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_best_simu_soc_' + job_id + '.csv', best_simu_soc.detach().cpu().numpy(), delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_best_pred_para_' + job_id + '.csv', best_pred_para.detach().cpu().numpy(), delimiter = ',')

	## bulk convergence ##
	# initializz a seperate array to store the prediction results for the test profiles
	# with return of the function: carbon_input, cpool_steady_state, cpools_layer, soc_layer, total_res_time, total_res_time_base, res_time_base_pools, t_scaler, bulk_A, w_scaler, bulk_K, bulk_V, bulk_xi, bulk_I, litter_fraction
	carbon_input_test = np.ones((wosis_profile_info.shape[0], 1))*np.nan
	cpool_steady_state_test = np.ones((wosis_profile_info.shape[0], 140))*np.nan
	cpools_layer_test = np.ones((wosis_profile_info.shape[0], 20))*np.nan
	soc_layer_test = np.ones((wosis_profile_info.shape[0], 20))*np.nan
	total_res_time_test = np.ones((wosis_profile_info.shape[0], 20))*np.nan
	total_res_time_base_test = np.ones((wosis_profile_info.shape[0], 20))*np.nan
	res_time_base_pools_test = np.ones((wosis_profile_info.shape[0], 140))*np.nan
	t_scaler_test = np.ones((wosis_profile_info.shape[0], 20))*np.nan
	bulk_A_test = np.ones((wosis_profile_info.shape[0], 1))*np.nan
	w_scaler_test = np.ones((wosis_profile_info.shape[0], 20))*np.nan
	bulk_K_test = np.ones((wosis_profile_info.shape[0], 1))*np.nan
	bulk_V_test = np.ones((wosis_profile_info.shape[0], 1))*np.nan
	bulk_xi_test = np.ones((wosis_profile_info.shape[0], 1))*np.nan
	bulk_I_test = np.ones((wosis_profile_info.shape[0], 1))*np.nan
	litter_fraction_test = np.ones((wosis_profile_info.shape[0], 1))*np.nan

	carbon_input_test_profile, cpool_steady_state_test_profile, cpools_layer_test_profile, \
		soc_layer_test_profile, total_res_time_test_profile, total_res_time_base_test_profile, res_time_base_pools_test_profile, \
			t_scaler_test_profile, bulk_A_test_profile, w_scaler_test_profile, bulk_K_test_profile, bulk_V_test_profile, bulk_xi_test_profile, \
				bulk_I_test_profile, litter_fraction_test_profile = fun_bulk_simu(best_guess_test_pred_para.to(device), test_x.to(device), test_z.to(device), args.vertical_mixing)

	# store the results
	carbon_input_test[test_profile_id, :] = carbon_input_test_profile.detach().cpu().numpy()
	cpool_steady_state_test[test_profile_id, :] = cpool_steady_state_test_profile.detach().cpu().numpy()
	cpools_layer_test[test_profile_id, :] = cpools_layer_test_profile.detach().cpu().numpy()
	soc_layer_test[test_profile_id, :] = soc_layer_test_profile.detach().cpu().numpy()
	total_res_time_test[test_profile_id, :] = total_res_time_test_profile.detach().cpu().numpy()
	total_res_time_base_test[test_profile_id, :] = total_res_time_base_test_profile.detach().cpu().numpy()
	res_time_base_pools_test[test_profile_id, :] = res_time_base_pools_test_profile.detach().cpu().numpy()
	t_scaler_test[test_profile_id, :] = t_scaler_test_profile.detach().cpu().numpy()
	bulk_A_test[test_profile_id, :] = bulk_A_test_profile.detach().cpu().numpy()
	w_scaler_test[test_profile_id, :] = w_scaler_test_profile.detach().cpu().numpy()
	bulk_K_test[test_profile_id, :] = bulk_K_test_profile.detach().cpu().numpy()
	bulk_V_test[test_profile_id, :] = bulk_V_test_profile.detach().cpu().numpy()
	bulk_xi_test[test_profile_id, :] = bulk_xi_test_profile.detach().cpu().numpy()
	bulk_I_test[test_profile_id, :] = bulk_I_test_profile.detach().cpu().numpy()
	litter_fraction_test[test_profile_id, :] = litter_fraction_test_profile.detach().cpu().numpy()

	# save data
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_bulk_carbon_input_' + job_id + '.csv', carbon_input_test, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_bulk_cpool_steady_state_' + job_id + '.csv', cpool_steady_state_test, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_bulk_cpools_layer_' + job_id + '.csv', cpools_layer_test, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_bulk_soc_layer_' + job_id + '.csv', soc_layer_test, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_bulk_total_res_time_' + job_id + '.csv', total_res_time_test, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_bulk_total_res_time_base_' + job_id + '.csv', total_res_time_base_test, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_bulk_res_time_base_pools_' + job_id + '.csv', res_time_base_pools_test, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_bulk_t_scaler_' + job_id + '.csv', t_scaler_test, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_bulk_bulk_A_' + job_id + '.csv', bulk_A_test, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_bulk_w_scaler_' + job_id + '.csv', w_scaler_test, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_bulk_bulk_K_' + job_id + '.csv', bulk_K_test, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_bulk_bulk_V_' + job_id + '.csv', bulk_V_test, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_bulk_bulk_xi_' + job_id + '.csv', bulk_xi_test, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_bulk_bulk_I_' + job_id + '.csv', bulk_I_test, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Test/nn_test_bulk_litter_fraction_' + job_id + '.csv', litter_fraction_test, delimiter = ',')

	#############
	# Test Maps #
	#############

	# get the latitudes and longitudes of the test profiles by matching ProfileID in env_info with the test_profile_id
	test_lons = np.ones((wosis_profile_info.shape[0]))*np.nan
	test_lats = np.ones((wosis_profile_info.shape[0]))*np.nan
	test_profile_id_all = np.ones((wosis_profile_info.shape[0]))*np.nan
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
	print("test_lons.min(): ", test_lons[test_profile_id_num] .min())
	print("test_lons.max(): ", test_lons[test_profile_id_num] .max())
	print("test_lats.min(): ", test_lats[test_profile_id_num] .min())
	print("test_lats.max(): ", test_lats[test_profile_id_num] .max())

	# get the upper and lower depth of the test profiles
	test_upper_depth = np.ones((wosis_profile_info.shape[0], 200))*np.nan
	test_lower_depth = np.ones((wosis_profile_info.shape[0], 200))*np.nan
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
	scaled_diff = np.ones((wosis_profile_info.shape[0]))*np.nan
	true_test_soc = []
	pred_test_soc = []

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
			# # # print outlier
			# if scaled_diff[i] > 2:
			#     print('outlier: ', test_profile_id_all[i], scaled_diff[i])
			pred_test_soc.append(temp_simu_sum.item())
			true_test_soc.append(temp_obs_sum.item())

	# Plot the scaled difference
	visualization_utils.plot_observations_world_map(test_lons, test_lats, scaled_diff, PLOT_DIR, "test_scaled_diff_" + job_id, us_only=True)

	# TODO Plot parameters
	for i, para_name in enumerate(para_names):  # in range(best_pred_para.shape[1]):
		visualization_utils.plot_observations_world_map(test_lons[test_profile_id_num],
														test_lats[test_profile_id_num],
														best_pred_para[test_profile_id_num, i].detach().cpu().numpy(),
														PLOT_DIR,
														"test_para_{}_{}".format(para_name, job_id), us_only=True)

	# # TODO Choose a location. Compute input similarity to it for all profiles.
	# for profile_idx in [0]:
	# 	profile_id_curr = test_profile_id_num[profile_idx]
	# 	input_curr = best_guess_test_input[profile_idx]
	# 	distances = []
	# 	for i in range(len(best_guess_test_input.shape[0])):
	# 		dist_curr_i = torch.norm(input_curr - best_guess_test_input[i])
	# 		distances.append(dist_curr_i)

	# 	visualization_utils.plot_observations_world_map(test_lons[test_profile_id_num],
	# 										   		    test_lats[test_profile_id_num],
	# 													distances, PLOT_DIR,
	# 													"test_dist_with_profile_lat={}_lon{}".format(test_lons[profile_id_curr], test_lats[profile_id_curr]), us_only=True)



	############
	# Val Data #
	############

	## predictions and parameters for the validation profiles ##

	# initializz a seperate array to store the prediction results for the validation profiles
	val_simu_soc = torch.tensor(np.ones((wosis_profile_info.shape[0], 200))*np.nan, dtype = torch.float32, device=device)
	val_pred_para = torch.tensor(np.ones((wosis_profile_info.shape[0], len(para_names)))*np.nan, dtype = torch.float32, device=device)

	val_simu_soc[val_profile_id, :] = best_guess_val_y_hat
	val_pred_para[val_profile_id, :] = best_guess_val_pred_para

	# save data
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_best_simu_soc_' + job_id + '.csv', val_simu_soc.detach().cpu().numpy(), delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_best_pred_para_' + job_id + '.csv', val_pred_para.detach().cpu().numpy(), delimiter = ',')

	## bulk convergence ##
	carbon_input_val = np.ones((wosis_profile_info.shape[0], 1))*np.nan
	cpool_steady_state_val = np.ones((wosis_profile_info.shape[0], 140))*np.nan
	cpools_layer_val = np.ones((wosis_profile_info.shape[0], 20))*np.nan
	soc_layer_val = np.ones((wosis_profile_info.shape[0], 20))*np.nan
	total_res_time_val = np.ones((wosis_profile_info.shape[0], 20))*np.nan
	total_res_time_base_val = np.ones((wosis_profile_info.shape[0], 20))*np.nan
	res_time_base_pools_val = np.ones((wosis_profile_info.shape[0], 140))*np.nan
	t_scaler_val = np.ones((wosis_profile_info.shape[0], 20))*np.nan
	bulk_A_val = np.ones((wosis_profile_info.shape[0], 1))*np.nan
	w_scaler_val = np.ones((wosis_profile_info.shape[0], 20))*np.nan
	bulk_K_val = np.ones((wosis_profile_info.shape[0], 1))*np.nan
	bulk_V_val = np.ones((wosis_profile_info.shape[0], 1))*np.nan
	bulk_xi_val = np.ones((wosis_profile_info.shape[0], 1))*np.nan
	bulk_I_val = np.ones((wosis_profile_info.shape[0], 1))*np.nan
	litter_fraction_val = np.ones((wosis_profile_info.shape[0], 1))*np.nan

	carbon_input_val_profile, cpool_steady_state_val_profile, cpools_layer_val_profile, \
		soc_layer_val_profile, total_res_time_val_profile, total_res_time_base_val_profile, res_time_base_pools_val_profile, \
			t_scaler_val_profile, bulk_A_val_profile, w_scaler_val_profile, bulk_K_val_profile, bulk_V_val_profile, bulk_xi_val_profile, \
				bulk_I_val_profile, litter_fraction_val_profile = fun_bulk_simu(best_guess_val_pred_para.to(device), val_x.to(device), val_z.to(device), args.vertical_mixing)

	# store the results
	carbon_input_val[val_profile_id, :] = carbon_input_val_profile.detach().cpu().numpy()
	cpool_steady_state_val[val_profile_id, :] = cpool_steady_state_val_profile.detach().cpu().numpy()
	cpools_layer_val[val_profile_id, :] = cpools_layer_val_profile.detach().cpu().numpy()
	soc_layer_val[val_profile_id, :] = soc_layer_val_profile.detach().cpu().numpy()
	total_res_time_val[val_profile_id, :] = total_res_time_val_profile.detach().cpu().numpy()
	total_res_time_base_val[val_profile_id, :] = total_res_time_base_val_profile.detach().cpu().numpy()
	res_time_base_pools_val[val_profile_id, :] = res_time_base_pools_val_profile.detach().cpu().numpy()
	t_scaler_val[val_profile_id, :] = t_scaler_val_profile.detach().cpu().numpy()
	bulk_A_val[val_profile_id, :] = bulk_A_val_profile.detach().cpu().numpy()
	w_scaler_val[val_profile_id, :] = w_scaler_val_profile.detach().cpu().numpy()
	bulk_K_val[val_profile_id, :] = bulk_K_val_profile.detach().cpu().numpy()
	bulk_V_val[val_profile_id, :] = bulk_V_val_profile.detach().cpu().numpy()
	bulk_xi_val[val_profile_id, :] = bulk_xi_val_profile.detach().cpu().numpy()
	bulk_I_val[val_profile_id, :] = bulk_I_val_profile.detach().cpu().numpy()
	litter_fraction_val[val_profile_id, :] = litter_fraction_val_profile.detach().cpu().numpy()

	# save data
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_bulk_carbon_input_' + job_id + '.csv', carbon_input_val, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_bulk_cpool_steady_state_' + job_id + '.csv', cpool_steady_state_val, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_bulk_cpools_layer_' + job_id + '.csv', cpools_layer_val, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_bulk_soc_layer_' + job_id + '.csv', soc_layer_val, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_bulk_total_res_time_' + job_id + '.csv', total_res_time_val, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_bulk_total_res_time_base_' + job_id + '.csv', total_res_time_base_val, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_bulk_res_time_base_pools_' + job_id + '.csv', res_time_base_pools_val, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_bulk_t_scaler_' + job_id + '.csv', t_scaler_val, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_bulk_bulk_A_' + job_id + '.csv', bulk_A_val, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_bulk_w_scaler_' + job_id + '.csv', w_scaler_val, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_bulk_bulk_K_' + job_id + '.csv', bulk_K_val, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_bulk_bulk_V_' + job_id + '.csv', bulk_V_val, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_bulk_bulk_xi_' + job_id + '.csv', bulk_xi_val, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_bulk_bulk_I_' + job_id + '.csv', bulk_I_val, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Validation/nn_val_bulk_litter_fraction_' + job_id + '.csv', litter_fraction_val, delimiter = ',')


	############
	# Val Maps #
	############
	# get the latitudes and longitudes of the validation profiles by matching ProfileID in env_info with the val_profile_id
	val_lons = np.ones((wosis_profile_info.shape[0]))*np.nan
	val_lats = np.ones((wosis_profile_info.shape[0]))*np.nan
	val_profile_id_all = np.ones((wosis_profile_info.shape[0]))*np.nan
	val_profile_id_all[val_profile_id] = val_profile_id
	val_profile_id_num = val_profile_id.numpy().astype(int)
	val_lons[val_profile_id_num] = np.array(env_info.loc[val_profile_id_num, "original_lon"])
	val_lats[val_profile_id_num] = np.array(env_info.loc[val_profile_id_num, "original_lat"])
	# print the range of val_lons and val_lats
	print("val_lons.min(): ", val_lons[val_profile_id_num].min())
	print("val_lons.max(): ", val_lons[val_profile_id_num].max())
	print("val_lats.min(): ", val_lats[val_profile_id_num].min())
	print("val_lats.max(): ", val_lats[val_profile_id_num].max())

	# get the upper and lower depth of the validation profiles
	val_upper_depth = np.ones((wosis_profile_info.shape[0], 200))*np.nan
	val_lower_depth = np.ones((wosis_profile_info.shape[0], 200))*np.nan
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
	scaled_diff = np.ones((wosis_profile_info.shape[0]))*np.nan
	true_test_soc = []
	pred_test_soc = []

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
			#     print('outlier: ', val_profile_id_all[i], scaled_diff[i])
			pred_test_soc.append(temp_simu_sum.item())
			true_test_soc.append(temp_obs_sum.item())

	# Plot the scaled difference
	visualization_utils.plot_observations_world_map(val_lons, val_lats, scaled_diff, PLOT_DIR, "validation_scaled_diff_" + job_id, us_only=True)



	##############
	# Train Data #
	##############

	## predictions and parameters for the training profiles ##

	# initializz a seperate array to store the prediction results for the training profiles
	train_simu_soc = torch.tensor(np.ones((wosis_profile_info.shape[0], 200))*np.nan, dtype = torch.float32, device=device)
	train_pred_para = torch.tensor(np.ones((wosis_profile_info.shape[0], len(para_names)))*np.nan, dtype = torch.float32, device=device)

	train_simu_soc[train_profile_id, :] = best_guess_train_y_hat
	train_pred_para[train_profile_id, :] = best_guess_train_pred_para

	# save data
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_best_simu_soc_' + job_id + '.csv', train_simu_soc.detach().cpu().numpy(), delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_best_pred_para_' + job_id + '.csv', train_pred_para.detach().cpu().numpy(), delimiter = ',')

	## bulk convergence ##
	carbon_input_train = np.ones((wosis_profile_info.shape[0], 1))*np.nan
	cpool_steady_state_train = np.ones((wosis_profile_info.shape[0], 140))*np.nan
	cpools_layer_train = np.ones((wosis_profile_info.shape[0], 20))*np.nan
	soc_layer_train = np.ones((wosis_profile_info.shape[0], 20))*np.nan
	total_res_time_train = np.ones((wosis_profile_info.shape[0], 20))*np.nan
	total_res_time_base_train = np.ones((wosis_profile_info.shape[0], 20))*np.nan
	res_time_base_pools_train = np.ones((wosis_profile_info.shape[0], 140))*np.nan
	t_scaler_train = np.ones((wosis_profile_info.shape[0], 20))*np.nan
	bulk_A_train = np.ones((wosis_profile_info.shape[0], 1))*np.nan
	w_scaler_train = np.ones((wosis_profile_info.shape[0], 20))*np.nan
	bulk_K_train = np.ones((wosis_profile_info.shape[0], 1))*np.nan
	bulk_V_train = np.ones((wosis_profile_info.shape[0], 1))*np.nan
	bulk_xi_train = np.ones((wosis_profile_info.shape[0], 1))*np.nan
	bulk_I_train = np.ones((wosis_profile_info.shape[0], 1))*np.nan
	litter_fraction_train = np.ones((wosis_profile_info.shape[0], 1))*np.nan

	carbon_input_train_profile, cpool_steady_state_train_profile, cpools_layer_train_profile, \
		soc_layer_train_profile, total_res_time_train_profile, total_res_time_base_train_profile, res_time_base_pools_train_profile, \
			t_scaler_train_profile, bulk_A_train_profile, w_scaler_train_profile, bulk_K_train_profile, bulk_V_train_profile, bulk_xi_train_profile, \
				bulk_I_train_profile, litter_fraction_train_profile = fun_bulk_simu(best_guess_train_pred_para.to(device), train_x.to(device), train_z.to(device), args.vertical_mixing)

	# store the results
	carbon_input_train[train_profile_id, :] = carbon_input_train_profile.detach().cpu().numpy()
	cpool_steady_state_train[train_profile_id, :] = cpool_steady_state_train_profile.detach().cpu().numpy()
	cpools_layer_train[train_profile_id, :] = cpools_layer_train_profile.detach().cpu().numpy()
	soc_layer_train[train_profile_id, :] = soc_layer_train_profile.detach().cpu().numpy()
	total_res_time_train[train_profile_id, :] = total_res_time_train_profile.detach().cpu().numpy()
	total_res_time_base_train[train_profile_id, :] = total_res_time_base_train_profile.detach().cpu().numpy()
	res_time_base_pools_train[train_profile_id, :] = res_time_base_pools_train_profile.detach().cpu().numpy()
	t_scaler_train[train_profile_id, :] = t_scaler_train_profile.detach().cpu().numpy()
	bulk_A_train[train_profile_id, :] = bulk_A_train_profile.detach().cpu().numpy()
	w_scaler_train[train_profile_id, :] = w_scaler_train_profile.detach().cpu().numpy()
	bulk_K_train[train_profile_id, :] = bulk_K_train_profile.detach().cpu().numpy()
	bulk_V_train[train_profile_id, :] = bulk_V_train_profile.detach().cpu().numpy()
	bulk_xi_train[train_profile_id, :] = bulk_xi_train_profile.detach().cpu().numpy()
	bulk_I_train[train_profile_id, :] = bulk_I_train_profile.detach().cpu().numpy()
	litter_fraction_train[train_profile_id, :] = litter_fraction_train_profile.detach().cpu().numpy()

	# save data
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_bulk_carbon_input_' + job_id + '.csv', carbon_input_train, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_bulk_cpool_steady_state_' + job_id + '.csv', cpool_steady_state_train, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_bulk_cpools_layer_' + job_id + '.csv', cpools_layer_train, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_bulk_soc_layer_' + job_id + '.csv', soc_layer_train, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_bulk_total_res_time_' + job_id + '.csv', total_res_time_train, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_bulk_total_res_time_base_' + job_id + '.csv', total_res_time_base_train, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_bulk_res_time_base_pools_' + job_id + '.csv', res_time_base_pools_train, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_bulk_t_scaler_' + job_id + '.csv', t_scaler_train, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_bulk_bulk_A_' + job_id + '.csv', bulk_A_train, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_bulk_w_scaler_' + job_id + '.csv', w_scaler_train, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_bulk_bulk_K_' + job_id + '.csv', bulk_K_train, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_bulk_bulk_V_' + job_id + '.csv', bulk_V_train, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_bulk_bulk_xi_' + job_id + '.csv', bulk_xi_train, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_bulk_bulk_I_' + job_id + '.csv', bulk_I_train, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_bulk_litter_fraction_' + job_id + '.csv', litter_fraction_train, delimiter = ',')

	##############
	# Train Maps #
	##############

	# get the latitudes and longitudes of the training profiles by matching ProfileID in env_info with the train_profile_id
	train_lons = np.ones((wosis_profile_info.shape[0]))*np.nan
	train_lats = np.ones((wosis_profile_info.shape[0]))*np.nan
	train_profile_id_all = np.ones((wosis_profile_info.shape[0]))*np.nan
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
	train_upper_depth = np.ones((wosis_profile_info.shape[0], 200))*np.nan
	train_lower_depth = np.ones((wosis_profile_info.shape[0], 200))*np.nan
	train_upper_depth[train_profile_id_num, :] = np.array(upper_depth_all[train_profile_id_num])
	train_lower_depth[train_profile_id_num, :] = np.array(lower_depth_all[train_profile_id_num])

	# save location data for training profiles
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_profile_id_' + job_id + '.csv', train_profile_id_all, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_lons_' + job_id + '.csv', train_lons, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_lats_' + job_id + '.csv', train_lats, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_upper_depth_' + job_id + '.csv', train_upper_depth, delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Train/nn_train_lower_depth_' + job_id + '.csv', train_lower_depth, delimiter = ',')

	# Plot maps to show the scaled difference between the predicted and observed SOC values for training profiles
	scaled_diff = np.ones((wosis_profile_info.shape[0]))*np.nan

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
			# # print outlier
			# if scaled_diff[i] > 2:
			#     print('outlier: ', train_profile_id_all[i], scaled_diff[i])

	# Plot the scaled difference
	visualization_utils.plot_observations_world_map(train_lons, train_lats, scaled_diff, PLOT_DIR, "train_scaled_diff_" + job_id, us_only=True)
	print("-----------------Model Test Finished at " + str(datetime.now()) + "-----------------")

	#########################
	# Grid prediction
	#########################
	# Predict the SOC values based on Grid environmental information using the best model
	grid_simu_soc, grid_pred_para = best_guess_model(torch.tensor(predict_data_x, dtype=torch.float32, device=device), 
												    torch.tensor(predict_data_z, dtype=torch.float32, device=device), 
												    torch.tensor(predict_data_c, dtype=torch.float32, device=device),
												    whether_predict = 1)
	# Save the predicted SOC values, parameters and location data into csv files
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_simu_soc_' + job_id + '.csv', grid_simu_soc.detach().cpu().numpy(), delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_pred_para_' + job_id + '.csv', grid_pred_para.detach().cpu().numpy(), delimiter = ',')
	# save grid_env_info_US['Original_Lat'] and grid_env_info_US['Original_Lon'] to csv files
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_lons_' + job_id + '.csv', grid_env_info_US['original_lon'], delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_lats_' + job_id + '.csv', grid_env_info_US['original_lat'], delimiter = ',')

	# Check grid shapes
	print("Predict data x", predict_data_x.shape)
	print("Grid simu soc", grid_simu_soc.shape)
	print("Grid pred para", grid_pred_para.shape)
	print("Grid env info US", grid_env_info_US.shape)

	# Map of each grid covariate
	for i, covariate in enumerate(var4nn):
		visualization_utils.plot_observations_world_map(grid_env_info_US["original_lon"],
												        grid_env_info_US["original_lat"], 
														predict_data_x[:, i, 0, 0], PLOT_DIR,
														"grid_covariate_{}_{}".format(covariate, job_id), us_only=True)

	# Map of each grid parameter (predictions)
	for i, para_name in enumerate(para_names):  # in range(best_pred_para.shape[1]):
		visualization_utils.plot_observations_world_map(grid_env_info_US["original_lon"],
												        grid_env_info_US["original_lat"], 
														grid_pred_para[:, i].detach().cpu().numpy(), PLOT_DIR,
														"grid_para_{}_{}".format(para_name, job_id), us_only=True)

	# Bulk simulation for the grid data
	carbon_input_pred, cpool_steady_state_pred, cpools_layer_pred, soc_layer_pred, total_res_time_pred, \
		total_res_time_base_pred, res_time_base_pools_pred, t_scaler_pred, bulk_A_pred, \
		w_scaler_pred, bulk_K_pred, bulk_V_pred, bulk_xi_pred, bulk_I_pred, litter_fraction_pred = fun_bulk_simu(grid_pred_para.to(device), \
																										torch.tensor(predict_data_x, dtype=torch.float32, device=device), \
																											torch.tensor(predict_data_z, dtype=torch.float32, device=device), args.vertical_mixing)
	# Save the bulk simulation results into csv files
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_bulk_carbon_input_' + job_id + '.csv', carbon_input_pred.detach().cpu().numpy(), delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_bulk_cpool_steady_state_' + job_id + '.csv', cpool_steady_state_pred.detach().cpu().numpy(), delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_bulk_cpools_layer_' + job_id + '.csv', cpools_layer_pred.detach().cpu().numpy(), delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_bulk_soc_layer_' + job_id + '.csv', soc_layer_pred.detach().cpu().numpy(), delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_bulk_total_res_time_' + job_id + '.csv', total_res_time_pred.detach().cpu().numpy(), delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_bulk_total_res_time_base_' + job_id + '.csv', total_res_time_base_pred.detach().cpu().numpy(), delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_bulk_res_time_base_pools_' + job_id + '.csv', res_time_base_pools_pred.detach().cpu().numpy(), delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_bulk_t_scaler_' + job_id + '.csv', t_scaler_pred.detach().cpu().numpy(), delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_bulk_bulk_A_' + job_id + '.csv', bulk_A_pred.detach().cpu().numpy(), delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_bulk_w_scaler_' + job_id + '.csv', w_scaler_pred.detach().cpu().numpy(), delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_bulk_bulk_K_' + job_id + '.csv', bulk_K_pred.detach().cpu().numpy(), delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_bulk_bulk_V_' + job_id + '.csv', bulk_V_pred.detach().cpu().numpy(), delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_bulk_bulk_xi_' + job_id + '.csv', bulk_xi_pred.detach().cpu().numpy(), delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_bulk_bulk_I_' + job_id + '.csv', bulk_I_pred.detach().cpu().numpy(), delimiter = ',')
	np.savetxt(data_dir_output + 'neural_network/' + job_id + '/Prediction/nn_grid_bulk_litter_fraction_' + job_id + '.csv', litter_fraction_pred.detach().cpu().numpy(), delimiter = ',')


	print("-----------------Model Prediction Finished at " + str(datetime.now()) + "-----------------")




	# if __name__ == '__main__':
	# 	print(f"INSIDE MAIN, num_CPU={args.num_CPU}")
	# 	# multiprocessing.set_start_method('spawn', force=True)  # NOTE changed @joshuafan

	# 	# Number of CPUs requester
	# 	world_size = args.num_CPU
	# 	processes = []
	# 	for rank in range(world_size):
	# 		p = Process(target=worker, args=(rank, world_size))
	# 		p.start()
	# 		processes.append(p)

	# 	for p in processes:
	# 		p.join()

		
