"""
Takes a trained model and produces plots that attempt to visualize/interpret its predictions.
"""

##################################################################################
# Boilerplate to load in dataset and trained model. Copied from binns_DDP.py.    #
##################################################################################
import csv
import functools
import math
import sys
import time
import random
import warnings
import subprocess
import argparse
import misc_utils
from mlp import mlp_wrapper
from sklearn.model_selection import KFold

from mlp import GNN_BINN, Spatial_BINN, mlp_wrapper, nn_only, BINN_Hybrid
from torch.optim.swa_utils import AveragedModel, SWALR
from pe_gcn_model import GridCellSpatialRelationEncoder
from spatial_utils import *
from losses import binns_loss, compute_param_matching_loss, compute_param_violation_loss
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
np.set_printoptions(legacy="1.25")

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
from multiprocessing import Process
from scipy.io import loadmat
import netCDF4 as ncread 
import mat73
from matplotlib import pyplot as plt

###################################
# Import CLM5 process-based model #
###################################
# fun_model_simu predicts at user-specified depths. fun_model_prediction predicts at 20 default layers.
from fun_matrix_clm5_vectorized import fun_model_simu, fun_model_prediction

# fun_bulk_simu returns additional components (quantities describing physical processes)
from fun_matrix_clm5_vectorized_bulk_converge import fun_bulk_simu

################################################
# Data Directories (CHANGE THIS!!!)
################################################
data_dir_input = '../ENSEMBLE/INPUT_DATA/'
data_dir_output = '../OUTPUT_DATA/'

# NAM model per input-output pair
# TRAINED_MODEL_PATH = "/mnt/beegfs/bulk/mirror/jyf6/datasets/BINNS/OUTPUT_DATA/neural_network/20250305-233914_NAM_DEBUGGING_lr=1e-03_fold=1_seed=1/opt_nn_20250305-233914_NAM_DEBUGGING_lr=1e-03_fold=1_seed=1.pt"

# NAM Joint (one model predicts everything, given feature). TODO RENAME
TRAINED_MODEL_PATH = "/mnt/beegfs/bulk/mirror/jyf6/datasets/BINNS/OUTPUT_DATA/neural_network/20250321-002807_NAM_DEBUG_lr=1e-03_fold=1_seed=1/opt_nn_20250321-002807_NAM_DEBUG_lr=1e-03_fold=1_seed=1.pt"

# TRAINED_MODEL_PATH = "/mnt/beegfs/bulk/mirror/jyf6/datasets/BINNS/OUTPUT_DATA/neural_network/20250321-005926_NAM_DEBUG_TEN_lr=1e-03_fold=1_seed=1/opt_nn_20250321-005926_NAM_DEBUG_TEN_lr=1e-03_fold=1_seed=1.pt"

# BINN-Synthetic 
# TRAINED_MODEL_PATH = "/mnt/beegfs/bulk/mirror/jyf6/datasets/BINNS/OUTPUT_DATA/neural_network/20250310-171603_BINN_SYNTHETIC_lr=1e-02_fold=1_seed=1/opt_nn_20250310-171603_BINN_SYNTHETIC_lr=1e-02_fold=1_seed=1.pt"

# KAN
# TRAINED_MODEL_PATH = "/mnt/beegfs/bulk/mirror/jyf6/datasets/BINNS/OUTPUT_DATA/neural_network/20250319-111214_KAN_lr=1e-03_fold=1_seed=1/opt_nn_20250319-111214_KAN_lr=1e-03_fold=1_seed=1.pt"

# Original BINN (hardsigmoid)


PLOT_DIR = os.path.join(os.path.dirname(TRAINED_MODEL_PATH), 'visualizations')
device = "cuda:" + str(os.environ["CUDA_VISIBLE_DEVICES"].split(',')[0]) if torch.cuda.is_available() else "cpu"

############################################################
# Load the model checkpoint (so we can see its arguments)
############################################################
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

checkpoint_path = TRAINED_MODEL_PATH
checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
args = checkpoint['args']  # Recently added, older models might not have this
set_seeds(args.seed)

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

#-------------------------------
# CLM5 constants
#-------------------------------
# Parameter names
# Assuming we are predicting all 21 original CLM5 parameters.
# Not supporting other options yet.
para_names = ['diffus', 'cryo', 'q10', 'efolding', 'taucwd', 'taul1', 'taul2', 'tau4s1', 'tau4s2', 'tau4s3', 'fl1s1', 'fl2s1', 'fl3s2', 'fs1s2', 'fs1s3', 'fs2s1', 'fs2s3', 'fs3s1', 'fcwdl2', 'w-scaling', 'beta']
para_index = np.arange(0, len(para_names))

# Min/max values
PARA_MIN_MAX = np.array([[3e-5, 5e-4], 
				         [3e-5, 16*1e-4],
						 [1.2, 3],
						 [0.1, 1],
						 [1, 6],
						 [0.0001, 0.11],
						 [0.1, 0.3],
						 [0.0001, 0.5],
						 [1, 10],
						 [20, 400],
						 [0.1, 0.8],
						 [0.2, 0.8],
						 [0.2, 0.8],
						 [0.0001, 0.4],
						 [0.0001, 0.1],
						 [0.1, 0.74],
						 [0.0001, 0.1],
						 [0.0001, 0.9],
						 [0.5, 1],
						 [0.0001, 5],
						 [0.5, 0.9999]])


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
	# Choose random subset of profiles for testing.
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
# Initialize numpy array for PRODA soc simulation data
PRODA_soc_simu = np.ones((len(current_data_profile_id), 200))*np.nan

if args.synthetic_labels:
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


################################################################
# Data splitting
################################################################
# Load train/val/test split from checkpoint.
train_loc = checkpoint['train_indices']
val_loc = checkpoint['val_indices']
test_loc = checkpoint['test_indices']

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

# # Exclude all rows with nan values
# grid_env_info = grid_env_info.dropna(axis=0, how='any')

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

# # Drop rows with nan values
# nan_loc = np.nanmean(predict_data_x[:, 0:len(var4nn), 0, 0], axis = 1) + \
# 			np.sum(predict_data_x[:, 0:12, 0, 1], axis = 1) + \
# 			np.sum(predict_data_x[:, 0:12, 0, 2], axis = 1) + \
# 			np.sum(predict_data_x[:, 0:12, 0, 3], axis = 1) + \
# 			np.sum(predict_data_x[:, 0:12, 0, 4], axis = 1) + \
# 			np.sum(predict_data_x[:, 0:12, 0, 5], axis = 1) + \
# 			np.sum(predict_data_x[:, 0:12, 0, 6], axis = 1) + \
# 			np.sum(predict_data_x[:, 0:12, 0, 7], axis = 1) + \
# 			np.sum(predict_data_x[:, 0:20, 0:12, 8], axis = (1, 2)) + \
# 			np.sum(predict_data_x[:, 0:20, 0:12, 9], axis = (1, 2)) + \
# 			np.sum(predict_data_x[:, 0:20, 0:12, 10], axis = (1, 2)) + \
# 			np.sum(predict_data_x[:, 0:20, 0:12, 11], axis = (1, 2)) + \
# 			np.sum(predict_data_x[:, 0:20, 0:12, 12], axis = (1, 2))
# valid_profile_loc = np.where(np.isnan(nan_loc) == False)[0]
# predict_data_x = predict_data_x[valid_profile_loc, :, :, :]
# predict_data_z = predict_data_z[valid_profile_loc]
# grid_env_info_US = grid_env_info_US.iloc[valid_profile_loc, :]

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
print("Original grid PRODA para shape", grid_PRODA_para.shape)

# # Filter to the 'grid profile IDs' inside the US bounding box. TODO Not needed anymore
# grid_PRODA_para = grid_PRODA_para[grid_PRODA_para['profile_id'].isin(grid_US_profiles)]
# print("Grid PRODA para shape after filter to US", grid_PRODA_para.shape)

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



##############################################################################
# Construct and load model
##############################################################################
if args.model in ["new_mlp", "lipmlp", "senn", "nam", "nam_joint", "nag", "kan"]:
	model_class = mlp_wrapper
elif args.model == 'nn_only':
	model_class = nn_only
else:
	raise ValueError("Unsupported model")
model_kwargs = checkpoint["model_kwargs"]
model = model_class(**model_kwargs).to(device)

# Load model from checkpoint
# (If resuming training, we would also need to load the optimizer state dict, but we are not doing that here)
state_dict = checkpoint['model_state_dict']
torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, prefix="module.")  # Remove module. from state_dict if DDP 
model.load_state_dict(state_dict)
model.eval()
print("MODEL", model)

# Initialize dataset. For now only using validation dataset.
val_dataset = MergeDataset(val_x, val_y, val_z, val_c, val_profile_id, val_proda_para)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)


##############################################################################
# Obtain all predictions from this dataset
##############################################################################
with torch.no_grad():
	# Get all results: batch-by-batch approach. TODO implement this for all data splits
	# all_val_pred_para = []
	# all_val_proda_para = []
	# all_val_coords = []
	# all_val_pred_soc = []
	# all_val_true_soc = []
	# all_val_z = []
	# all_val_new_input = []  # List the numeric features first, right before passing through NN
	# all_val_f_out = []  # For Neural Additive Model only, contribution of each feature towards each output

	# for batch_info in val_loader:
	# 	batch_x, batch_y, batch_z, batch_c, batch_profile_id, batch_proda_para = batch_info
	# 	batch_x = batch_x.to(device)
	# 	batch_y = batch_y.to(device)
	# 	batch_z = batch_z.to(device)
	# 	batch_c = batch_c.to(device)
	# 	batch_proda_para = batch_proda_para.to(device)
	# 	batch_profile_id = batch_profile_id.to(device)

	# 	# Obtain predicted SOC and parameters.
	# 	# batch_y_hat is [batch, 200] where 200 is the max number of observations per site (usually less; invalid entries are nan)
	# 	# batch_pred_para is [batch, 21] where 21 is the number of CLM5 parameters.
	# 	batch_y_hat, batch_pred_para = model(batch_x, batch_z, batch_c, whether_predict=0, one_param_only=args.one_param_only, PRODA_para=batch_proda_para)

	# 	# Record predicted parameters, true/predicted SOC
	# 	all_val_pred_para.append(batch_pred_para)
	# 	all_val_proda_para.append(batch_proda_para)
	# 	all_val_coords.append(batch_c)
	# 	all_val_z.append(batch_z)
	# 	all_val_pred_soc.append(batch_y_hat)
	# 	all_val_true_soc.append(batch_y)
	# 	all_val_new_input.append(model.new_input)

	# 	# If NAM record the feature contributions as well
	# 	if args.model in ["nam", "nam_joint"]:
	# 		batch_f_out = model.mlp.f_out  # [batch, n_features, n_params]
	# 		all_val_f_out.append(batch_f_out)

	# # Concatenate all batches' results
	# all_val_pred_para = torch.cat(all_val_pred_para, dim=0)
	# all_val_proda_para = torch.cat(all_val_proda_para, dim=0)
	# all_val_coords = torch.cat(all_val_coords, dim=0)
	# all_val_z = torch.cat(all_val_z, dim=0)
	# all_val_pred_soc = torch.cat(all_val_pred_soc, dim=0)
	# all_val_true_soc = torch.cat(all_val_true_soc, dim=0)
	# all_val_new_input = torch.cat(all_val_new_input, dim=0)
	# all_val_f_out = torch.cat(all_val_f_out, dim=0)  # [n_examples, n_features, n_params]


	# Get all results (putting everything in one batch)
	# Get predictions for train examples, compute loss & plot
	train_y_hat, train_pred_para = model(train_x.to(device), train_z.to(device), train_c.to(device),
										 whether_predict=0, PRODA_para=train_proda_para.to(device))
	train_mae, train_smooth_l1_loss, train_mse, _, train_NSE = binns_loss(train_y_hat, train_y.to(device), train_pred_para)
	print(f'Train - MSE: {train_mse.item():.2f}, MAE: {train_mae.item():.2f}, NSE: {train_NSE.item():.2f}')
	if args.model in ["nam", "nam_joint"]:
		train_f_out = model.mlp.f_out
		train_new_input = model.new_input

	# Get predictions for val examples, compute loss & plot
	val_y_hat, val_pred_para = model(val_x.to(device), val_z.to(device), val_c.to(device),
									 whether_predict=0, PRODA_para=val_proda_para.to(device))
	val_mae, val_smooth_l1_loss, val_mse, _, val_NSE = binns_loss(val_y_hat, val_y.to(device), val_pred_para)
	print(f'Val - MSE: {val_mse.item():.2f}, MAE: {val_mae.item():.2f}, NSE: {val_NSE.item():.2f}')
	if args.model in ["nam", "nam_joint"]:
		val_f_out = model.mlp.f_out
		val_new_input = model.new_input

	# Get predictions for test examples, compute loss & plot
	test_y_hat, test_pred_para = model(test_x.to(device), test_z.to(device), test_c.to(device),
									   whether_predict=0, PRODA_para=test_proda_para.to(device))
	test_mae, test_smooth_l1_loss, test_mse, _, test_NSE = binns_loss(test_y_hat, test_y.to(device), test_pred_para)
	print(f'Test - MSE: {test_mse.item():.2f}, MAE: {test_mae.item():.2f}, NSE: {test_NSE.item():.2f}')
	if args.model in ["nam", "nam_joint"]:
		test_f_out = model.mlp.f_out
		test_new_input = model.new_input

	# Get predictions for grid points
	grid_simu_soc, grid_pred_para = model(predict_data_x.to(device), predict_data_z.to(device), predict_data_c.to(device),
										  whether_predict = 1, PRODA_para=grid_PRODA_para.to(device))
	if args.model in ["nam", "nam_joint"]:
		grid_f_out = model.mlp.f_out
		grid_new_input = model.new_input

	# KAN-specific visualizations
	if args.model == "kan":
		pruned_model = model.mlp  # .prune()
		pruned_model.plot(scale=5.0, in_vars=var4nn, out_vars=para_names, varscale=0.1)
		plt.savefig(os.path.join(PLOT_DIR, "kan_plot.png"))
		plt.close()
		exit(1)

	# NAM-specific visualziations
	if args.model in ["nam", "nam_joint"]:
		# Feature importance
		# NAM arranges features as [numeric features, categorical features], and the categorical
		# features are listed in the same order as var_idx_to_emb.keys()
		# NOTE: the spatial positional encoding is not supported!
		NAM_FEATURE_ORDER = [var4nn[int(i)] for i in (model.non_categorical_indices + list(model.var_idx_to_emb.keys()))]
		# print("NAM FEATURE ORDER", NAM_FEATURE_ORDER)
		variances = torch.var(val_f_out, dim=0)  # [n_features, n_params]
		visualization_utils.plot_matrix(variances.cpu().detach().numpy(), row_labels=NAM_FEATURE_ORDER, col_labels=para_names,
										filename=os.path.join(PLOT_DIR, "nam_feature_importance.png"), 
										title="Feature contributions")

		# Loop through each output parameter. Plot shape function of 5 most influential features
		N_ROWS = 5
		fig, axeslist = plt.subplots(N_ROWS, len(para_names), figsize=(5*len(para_names), 2*N_ROWS))
		for para_idx in range(len(para_names)):
			important_feature_idx = torch.topk(variances[:, para_idx], N_ROWS).indices

			# for feat_idx in range(len(NAM_FEATURE_ORDER)):  # All features
			# 	row = feat_idx
			for row, feat_idx in enumerate(important_feature_idx):  # Important features
				feat_name = NAM_FEATURE_ORDER[feat_idx]
				
				feat_idx_original = list(env_info.columns).index(feat_name)
				scaling_x_min = col_max_min[feat_idx_original, 0]
				scaling_x_max = col_max_min[feat_idx_original, 1]
				scaling_y_min = PARA_MIN_MAX[para_idx, 0]
				scaling_y_max = PARA_MIN_MAX[para_idx, 1]

				ax = axeslist[row, para_idx]
				if feat_idx < len(model.non_categorical_indices):  # Numeric feature
					if args.model == "nam":  # Separate model for each input-output pair
						feat_nn = model.mlp.feature_nns[feat_idx * len(para_names) + para_idx].cpu()
					elif args.model == "nam_joint":  # Single model for each input, extract the correct output
						feat_nn_all = model.mlp.feature_nns[feat_idx].cpu()
						feat_nn = lambda x: feat_nn_all(x)[:, para_idx:para_idx+1]
					feat_vals = grid_new_input[:, feat_idx].cpu().detach().numpy()  # Values of the feature
					visualization_utils.plot_shape_function(feat_nn, feat_vals, ax=ax, title=feat_name, xlabel=feat_name, ylabel="Change to " + para_names[para_idx], divide_by=6 * model.mlp.divide_by,
											 				scaling_x_min=scaling_x_min, scaling_x_max=scaling_x_max, scaling_y_min=scaling_y_min, scaling_y_max=scaling_y_max)  # 1/6 is the slope of hardsigmoid, so divide by 6
				else:  # Categorical feature, plot map of grid contributions
					f_out = grid_f_out[:, feat_idx, para_idx]
					visualization_utils.plot_observations_world_map(predict_data_c[:, 0], predict_data_c[:, 1], f_out, None, 
																	feat_name, title=feat_name, us_only=True, ax=ax)

		# Column headers. Source: https://stackoverflow.com/a/25814386
		pad = 15
		for ax, para_name in zip(axeslist[0], para_names):
			ax.annotate(para_name, xy=(0.5, 1), xytext=(0, pad), xycoords='axes fraction', textcoords='offset points',
                        size='large', ha='center', va='baseline')

		fig.tight_layout()
		fig.subplots_adjust(top=0.90)
		plt.savefig(os.path.join(PLOT_DIR, "nam_shape_functions.png"))
		plt.close()


	# FINAL SUMMARY MAPS
	# Scatters of true-vs-predicted SOC (grid).
	# Each row represents a layer (or all layers), each column represents a split (train/val/test)
	titles = ["Train: All Depths", "Val: All Depths", "Test: All Depths"]
	y_hats = [train_y_hat.flatten(), val_y_hat.flatten(), test_y_hat.flatten()]  # predictions
	ys = [train_y.flatten().to(device), val_y.flatten().to(device), test_y.flatten().to(device)]  # labels
	LAYER_BOUNDARIES = [0, 0.1, 0.3, 1.0, 50.0]
	for i in range(len(LAYER_BOUNDARIES) - 1):  # Loop through layers
		layer_loc_train = (train_z >= LAYER_BOUNDARIES[i]) & (train_z < LAYER_BOUNDARIES[i+1])  # True for observations within this layer that are non-nan
		layer_loc_val = (val_z >= LAYER_BOUNDARIES[i]) & (val_z < LAYER_BOUNDARIES[i+1]) 
		layer_loc_test = (test_z >= LAYER_BOUNDARIES[i]) & (test_z < LAYER_BOUNDARIES[i+1]) 
		y_hats.extend([train_y_hat[layer_loc_train], val_y_hat[layer_loc_val], test_y_hat[layer_loc_test]])
		ys.extend([train_y[layer_loc_train].to(device), val_y[layer_loc_val].to(device), test_y[layer_loc_test].to(device)])
		layer_str = f'{LAYER_BOUNDARIES[i]}-{LAYER_BOUNDARIES[i+1]}m'
		titles.extend([f'Train: {layer_str}', f'Val: {layer_str}', f'Test: {layer_str}'])
	visualization_utils.plot_true_vs_predicted_multiple(os.path.join(PLOT_DIR, f"FINAL_POSTTRAINING_scatters.png"), y_hats, ys, titles, cols=3)
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
		train_pred_soc = torch.where(layer_loc_train.to(device), train_y_hat, torch.nan)  # Same for predictions
		train_pred_soc = torch.nanmean(train_pred_soc, dim=1)

		# Repeat above for val data
		layer_loc_val = (val_z >= LAYER_BOUNDARIES[i]) & (val_z < LAYER_BOUNDARIES[i+1])  # True for observations within this layer that are non-nan
		val_true_soc = torch.where(layer_loc_val.to(device), val_y.to(device), torch.nan)  # Create tensor: only observations in this layer, nan elsewhere
		val_true_soc = torch.nanmean(val_true_soc, dim=1)  # For each site, average over observations in this layer. If none, return nan.
		val_pred_soc = torch.where(layer_loc_val.to(device), val_y_hat, torch.nan)  # Same for predictions
		val_pred_soc = torch.nanmean(val_pred_soc, dim=1)

		# Repeat above for test data
		layer_loc_test = (test_z >= LAYER_BOUNDARIES[i]) & (test_z < LAYER_BOUNDARIES[i+1])  # True for observations within this layer that are non-nan
		test_true_soc = torch.where(layer_loc_test.to(device), test_y.to(device), torch.nan)  # Create tensor: only observations in this layer, nan elsewhere
		test_true_soc = torch.nanmean(test_true_soc, dim=1)  # For each site, average over observations in this layer. If none, return nan.
		test_pred_soc = torch.where(layer_loc_test.to(device), test_y_hat, torch.nan)  # Same for predictions
		test_pred_soc = torch.nanmean(test_pred_soc, dim=1)

		# Collect results
		lons_list.extend([train_c[:, 0], train_c[:, 0], val_c[:, 0], val_c[:, 0], test_c[:, 0], test_c[:, 0]])
		lats_list.extend([train_c[:, 1], train_c[:, 1], val_c[:, 1], val_c[:, 1], test_c[:, 1], test_c[:, 1]])
		values_list.extend([train_true_soc, train_pred_soc, val_true_soc, val_pred_soc, test_true_soc, test_pred_soc])
		layer_str = f'{LAYER_BOUNDARIES[i]}-{LAYER_BOUNDARIES[i+1]}m'
		vars_list.extend([f'True SOC - Train: {layer_str}', f'Predicted SOC - Train: {layer_str}',
							f'True SOC - Val: {layer_str}', f'Predicted SOC - Val: {layer_str}',
							f'True SOC - Test: {layer_str}', f'Predicted SOC - Test: {layer_str}',])
	visualization_utils.plot_map_grid(os.path.join(PLOT_DIR, f"FINAL_POSTTRAINING_soc_maps.png"),
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
			lons_list.extend([train_c[:, 0], train_c[:, 0], val_c[:, 0], val_c[:, 0], test_c[:, 0], test_c[:, 0], predict_data_c[:, 0], predict_data_c[:, 0]])
			lats_list.extend([train_c[:, 1], train_c[:, 1], val_c[:, 1], val_c[:, 1], test_c[:, 1], test_c[:, 1], predict_data_c[:, 1], predict_data_c[:, 1]])
			values_list.extend([train_proda_para[:, para_idx].to(device), train_pred_para[:, para_idx],
								val_proda_para[:, para_idx].to(device), val_pred_para[:, para_idx],
								test_proda_para[:, para_idx].to(device), test_pred_para[:, para_idx],
								grid_PRODA_para[:, para_idx].to(device), grid_pred_para[:, para_idx]])
			para_name = para_names[para_idx]
			vars_list.extend([f'PRODA para {para_name} - Train', f'Predicted para {para_name} - Train',
								f'PRODA para {para_name} - Val', f'Predicted para {para_name} - Val',
								f'PRODA para {para_name} - Test', f'Predicted para {para_name} - Test',
								f'PRODA para {para_name} - Grid', f'Predicted para {para_name} - Grid'])
		visualization_utils.plot_map_grid(os.path.join(PLOT_DIR, f"FINAL_POSTTRAINING_para_maps.png"),
				lons_list, lats_list, values_list, vars_list, us_only=True, cols=2)

		# Also plot scatters
		y_hats = []
		ys = []
		titles = []
		for para_idx in para_index:  # Only plot parameters that were predicted by NN
			y_hats.extend([train_pred_para[:, para_idx], val_pred_para[:, para_idx], test_pred_para[:, para_idx], grid_pred_para[:, para_idx]])
			ys.extend([train_proda_para[:, para_idx], val_proda_para[:, para_idx], test_proda_para[:, para_idx], grid_PRODA_para[:, para_idx].to(device)])
			para_name = para_names[para_idx]
			titles.extend([f'Train: {para_name}', f'Val: {para_name}', f'Test: {para_name}', f'Grid: {para_name}'])
		visualization_utils.plot_true_vs_predicted_multiple(os.path.join(PLOT_DIR, f"FINAL_POSTTRAINING_para_scatters.png"), y_hats, ys, titles, cols=4)

	print("-----------------Model Prediction Finished at " + str(datetime.now()) + "-----------------")
