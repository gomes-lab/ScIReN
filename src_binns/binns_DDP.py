# Distributed Data Parallel (DDP) training script for BINN
# Initializer script for DDP, automatically resubmit jobs after 11.5 hours (call DDP_resume.py)
# Import the required libraries
import sys
import time
import warnings
import subprocess

# sys.path.append('C:/Users/hx293/Research_Data/BINN/')
sys.path.append('/glade/u/home/haodixu/BINN')
# sys.path.append(r'/User/homes/ftao/Projects/BINNS/src_binns')

# Set HDF5_DISABLE_VERSION_CHECK to suppress version mismatch error
import os
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2'
import psutil
import gc

from datetime import datetime
from pandas import DataFrame as df
import numpy as np
from scipy.interpolate import pchip_interpolate
# parallel computing
# import concurrent.futures
# import torch.multiprocessing as mp
# initialize the multiprocessing for pytorch
# mp.set_start_method('spawn', force=True)

import os
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import multiprocessing
from multiprocessing import Process
# set random seed
torch.manual_seed(0)


from scipy.io import loadmat
import netCDF4 as ncread 
import mat73

from matplotlib import pyplot as plt
from collections import OrderedDict

#####################################
# Import Different Versions of CLM5 #
#####################################

# from fun_matrix_clm5 import fun_model_simu
from fun_matrix_clm5_vectorized import fun_model_simu
# from fun_matrix_clm5_GPU import fun_model_simu
# from fun_matrix_clm5_parallel import fun_model_simu
# from fun_matrix_clm5_parallel_update_V_matrix import fun_model_simu
import visualization_utils

if torch.cuda.is_available():
	dev = 'cuda'
else:
	dev = 'cpu'
# @joshuafan changed
# dev = 'cpu'
device = torch.device(dev) 
print(datetime.now(), '------------device: ', device, '------------')

# print the number of cores
cpu_count = multiprocessing.cpu_count()
thread_count = torch.get_num_threads()
# print("Number of CPUs: ", torch.cpu.device_count())  # No idea why it is not working on NCAR server
print("Number of Cores: ", cpu_count)
print("Number of threads: ", thread_count)
# set the number of threads
# torch.set_num_threads(thread_count-1)
print(datetime.now(), '------------number of cores: ', cpu_count, '------------')

print(datetime.now(), '------------all packages loaded------------')

# time_stamp = f'{datetime.date(datetime.now())}'
time_stamp = str(datetime.now()).replace(':', '_').replace(' ', '_').replace('.', '_')
start_time = time.time()

# Get job id
job_id = os.environ.get('PBS_JOBID')
job_id = job_id.split('.')[0]

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

# @joshuafan changed
# pathway
# data_dir_input = '/Users/phoenix/Google_Drive/Tsinghua_Luo/Projects/DATAHUB/ENSEMBLE/INPUT_DATA/'
# data_dir_output = '/Users/phoenix/Google_Drive/Tsinghua_Luo/Projects/DATAHUB/BINNS/OUTPUT_DATA/'
# data_dir_input = 'C:/Users/hx293/Research_Data/BINN/ENSEMBLE/INPUT_DATA/'
# data_dir_output = 'C:/Users/hx293/Unsync_Data/BINN_output/'
# server path
data_dir_input = '/glade/u/home/haodixu/BINN/ENSEMBLE/INPUT_DATA/'
data_dir_output = '/glade/u/home/haodixu/BINN/BINNS/OUTPUT_DATA/'
os.makedirs(os.path.join(data_dir_output, "neural_network"), exist_ok=True)
PLOT_DIR = os.path.join(data_dir_output, "visualizations")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(os.path.join(data_dir_output, "neural_network", job_id), exist_ok=True)

# constants
month_num = 12 
soil_cpool_num = 7
soil_decom_num = 20

#-------------------------------
# wosis data
#-------------------------------
# load wosis data

# layer_info: "profile_id, date, upper_depth, lower_depth, node_depth, soc_layer_weight, soc_stock, bulk_denstiy, is_pedo"
nc_data_middle = ncread.Dataset(data_dir_input + 'wosis_2019_snap_shot/soc_profile_wosis_2019_snapshot_hugelius_mishra.nc') # wosis profile info
wosis_profile_info = nc_data_middle['soc_profile_info'][:].data.transpose()
nc_data_middle.close()

nc_data_middle = ncread.Dataset(data_dir_input + 'wosis_2019_snap_shot/soc_data_integrate_wosis_2019_snapshot_hugelius_mishra.nc') # wosis SOC info
wosis_soc_info = nc_data_middle['data_soc_integrate'][:].data.transpose()
nc_data_middle.close()

#-------------------------------
# CLM5 constants
#-------------------------------
para_names = ['diffus', 'cryo', 'q10', 'efolding', 'taucwd', 'taul1', 'taul2', 'tau4s1', 'tau4s2', 'tau4s3', 'fl1s1', 'fl2s1', 'fl3s2', 'fs1s2', 'fs1s3', 'fs2s1', 'fs2s3', 'fs3s1', 'fcwdl2', 'w-scaling', 'beta']
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
    (wosis_profile_info[:, 3] >= -124.763068) & 
    (wosis_profile_info[:, 3] <= -66.949895) & 
    (wosis_profile_info[:, 4] >= 24.5) & 
    (wosis_profile_info[:, 4] <= 49.384358)
)[0]
profile_collection = np.reshape(profile_collection, [profile_collection.shape[0], 1])

profile_range = np.arange(0, len(profile_collection))

print(datetime.now(), '------------all input data loaded------------')

#---------------------------------------------------
# wrap up soc data for NN
#---------------------------------------------------
obs_soc_matrix = np.ones([len(profile_collection), 200])*np.nan  # Each row is a profile. Each non-nan column is an SOC observation
obs_depth_matrix = np.ones([len(profile_collection), 200])*np.nan  # Each row is a profile. Each column represents the depth of the corresponding SOC observation in "obs_soc_matrix"

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
	# exclude nan values
	valid_soc_loc = np.where((np.isnan(wosis_layer_obs) == False) & (np.isnan(wosis_layer_depth) == False))
	# valid layer number
	num_layers = len(valid_soc_loc[0])
	
	if num_layers > 0:
		wosis_layer_depth = wosis_layer_depth[valid_soc_loc]/100 # convert unit from cm to m
		wosis_layer_obs = wosis_layer_obs[valid_soc_loc]
		
		obs_depth_matrix[iprofile_hat, 0:num_layers] = wosis_layer_depth
		obs_soc_matrix[iprofile_hat, 0:num_layers] = wosis_layer_obs
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

print(datetime.now(), '------------soc data prepared------------')
########################################################
# neural network (BINNS)
########################################################
batch_size = 32
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
					['USDA_Suborder'], ['WRB_Subgroup'], ['Koppen_Climate_2018']]

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
		print("Var {} Nans {}".format(var, np.count_nonzero(np.isnan(env_info[:, idx]))))
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
env_info["original_lon"] = original_lons
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


var_idx_to_emb = dict()
for group in categorical_vars:
	n_categories = int(np.nanmax(env_info[group]) + 1)
	print("Variable {}: num categories {}".format(group, n_categories))
	print("Unique values", env_info[group].value_counts(sort=True))
	emb = nn.Embedding(num_embeddings=n_categories, embedding_dim=5).to(device)
	for var in group:
		idx = var4nn.index(var)
		var_idx_to_emb[idx] = emb


# For each embedding layer, extract which variables are passed through it,
# and the number of classes for each variable


#---------------------------------------------------
# training data
#---------------------------------------------------
current_data_x = np.ones((len(profile_collection), 60, 12, 13))*np.nan
current_data_x[:, 0:60, 0, 0] = np.array(env_info.loc[profile_collection[:, 0], var4nn])
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

# for col_idx, col_name in enumerate(var4nn):
# 	envir_var_values = current_data_x[:, col_idx, 0, 0]
# 	visualization_utils.plot_observations_world_map(lons, lats, envir_var_values, PLOT_DIR, col_name)

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
			np.sum(current_data_x[:, 0:60, 0, 0], axis = 1) + \
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

valid_profile_loc = np.where(np.isnan(nan_loc) == False)[0]

current_data_y = current_data_y[valid_profile_loc, :]
current_data_z = current_data_z[valid_profile_loc, :]
current_data_x = current_data_x[valid_profile_loc, :, :, :]
current_data_profile_id = profile_collection[valid_profile_loc, 0]


# Train, validation, test split
if test_split_ratio == 0:
	train_loc = np.random.choice(np.arange(0, len(current_data_x[:, 0])), size = round((1-nn_split_ratio)*len(current_data_x[:, 0])), replace = False)
	val_loc = np.setdiff1d(np.arange(0, len(current_data_x[:, 0])), train_loc)

	train_y = torch.tensor(current_data_y[train_loc, :], dtype = torch.float32)
	val_y = torch.tensor(current_data_y[val_loc, :], dtype = torch.float32)

	train_z = torch.tensor(current_data_z[train_loc, :], dtype = torch.float32)
	val_z = torch.tensor(current_data_z[val_loc, :], dtype = torch.float32)

	train_x = torch.tensor(current_data_x[train_loc, :, :, :], dtype = torch.float32)
	val_x = torch.tensor(current_data_x[val_loc, :, :, :], dtype = torch.float32)

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

	train_x = torch.tensor(current_data_x[train_loc, :, :, :], dtype=torch.float32)
	val_x = torch.tensor(current_data_x[val_loc, :, :, :], dtype=torch.float32)
	test_x = torch.tensor(current_data_x[test_loc, :, :, :], dtype=torch.float32)

	train_profile_id = torch.tensor(current_data_profile_id[train_loc], dtype=torch.long)
	val_profile_id = torch.tensor(current_data_profile_id[val_loc], dtype=torch.long)
	test_profile_id = torch.tensor(current_data_profile_id[test_loc], dtype=torch.long)




# test
# torch.autograd.set_detect_anomaly(True)

# pred_para = torch.rand((len(train_loc), len(para_names)), requires_grad = True)
# soc_simu = fun_model_simu(pred_para[0:32, :], train_y[0:32, :, :, :])
# soc_true =  train_y[0:32, :, 0, 0]
# 
# 
# soc_simu_vector = torch.reshape(soc_simu, [1, -1])
# soc_true_vector = torch.reshape(soc_true, [1, -1])
# 
# valid_loc = torch.where(torch.isnan(soc_simu_vector+soc_true_vector) == False) 
# 
# soc_simu_vector = soc_simu_vector[valid_loc]
# soc_true_vector = soc_true_vector[valid_loc]
# 
# modeling_inefficiency = torch.sum((soc_simu_vector - soc_true_vector)**2)/torch.sum((soc_true_vector - torch.mean(soc_true_vector))**2)
# 
# modeling_inefficiency.backward(retain_graph=True)

print(datetime.now(), '------------nn data prepared------------')
#---------------------------------------------------
# constants for NN
#---------------------------------------------------
nn_training_name = job_id + '_' + model_name

writer = SummaryWriter(data_dir_output + 'tensorboard/' + nn_training_name)

#---------------------------------------------------
# define the loss function                          
#---------------------------------------------------
def binns_loss(y_pred, y_true):
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

	# modeling inefficiency
	modeling_inefficiency = torch.sum((soc_simu_vector - soc_true_vector)**2)/torch.sum((soc_true_vector - torch.mean(soc_true_vector))**2)
	# modeling_inefficiency = torch.sum((soc_simu_vector - soc_true_vector)**2)/len(soc_true_vector) 
	
	# print(modeling_inefficiency)

	##########################
	# Regularization Penalty #
	##########################
	# lambda_reg = 0.001
	# L1_reg = sum(p.abs().sum() for p in model.parameters())
	# l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
	loss = modeling_inefficiency # + lambda_reg * l2_reg
	return loss
	
	# return modeling_inefficiency
# end binns loss


#---------------------------------------------------
# NN by PyTorch
#---------------------------------------------------
# define model
class nn_model(nn.Module):
	def __init__(self, var_idx_to_emb):
		super().__init__()

		# Dict from categorical variable index -> Embedding layer we use
		self.var_idx_to_emb = var_idx_to_emb

		# List of non-categorical variable indices
		self.non_categorical_indices = list(set(list(range(len(var4nn)))).difference(var_idx_to_emb.keys()))
		print("Categorical indices", var_idx_to_emb.keys())
		print("Noncategorical indices", self.non_categorical_indices)
		new_input_size = len(self.non_categorical_indices)
		for idx, emb in self.var_idx_to_emb.items():
			new_input_size += emb.embedding_dim

		# Neural network layers
		# first layer
		self.l1 = nn.Linear(new_input_size, 256)
		torch.nn.init.xavier_uniform_(self.l1.weight)
		nn.init.zeros_(self.l1.bias)
		
		# second layer
		self.l2 = nn.Linear(256, 512)
		torch.nn.init.xavier_uniform_(self.l2.weight)
		nn.init.zeros_(self.l2.bias)

		# third layer
		self.l3 = nn.Linear(512, 512)
		torch.nn.init.xavier_uniform_(self.l3.weight)
		nn.init.zeros_(self.l3.bias)

		# fourth layer
		self.l4 = nn.Linear(512, 256)
		torch.nn.init.xavier_uniform_(self.l4.weight)
		nn.init.zeros_(self.l4.bias)

		# fifth layer
		self.l5 = nn.Linear(256, 21)
		torch.nn.init.xavier_uniform_(self.l5.weight)
		nn.init.zeros_(self.l5.bias)

		# Dropout layers
		self.dropout = nn.Dropout(0.3)

		# # Transform from 256 to 512
		# self.transform_h1_to_h2 = nn.Linear(256, 512)
		# torch.nn.init.xavier_uniform_(self.transform_h1_to_h2.weight)
		# nn.init.zeros_(self.transform_h1_to_h2.bias)

		# # Transform from 512 to 256
		# self.transform_h3_to_h4 = nn.Linear(512, 256)
		# torch.nn.init.xavier_uniform_(self.transform_h3_to_h4.weight)
		# nn.init.zeros_(self.transform_h3_to_h4.bias)

	def forward(self, input_var, wosis_depth):
		predictor = input_var[:, :, 0, 0]
		forcing = input_var[:, :, :, :]
		obs_depth = wosis_depth

		# Compute embeddings for all categorical variables
		embs = []
		for idx, embedding_layer in self.var_idx_to_emb.items():
			emb = embedding_layer(predictor[:, idx].int())
			embs.append(emb)
		all_embs = torch.concatenate(embs, dim=1)
		new_input = torch.concatenate([predictor[:, self.non_categorical_indices], all_embs], dim=1)

		# hidden layers
		h1 = nn.functional.relu(self.l1(new_input))
		h1 = self.dropout(h1)
		# transformed_h1 = self.transform_h1_to_h2(h1)
		h2 = nn.functional.relu(self.l2(h1)) # + transformed_h1 # residual connection
		h2 = self.dropout(h2)
		h3 = nn.functional.relu(self.l3(h2)) + h2 # residual connection
		h3 = self.dropout(h3)
		# transformed_h3 = self.transform_h3_to_h4(h3)
		h4 = nn.functional.relu(self.l4(h3)) # + transformed_h3 # residual connection
		h4 = self.dropout(h4)
		h5 = torch.sigmoid(self.l5(h4)/100) # hardtanh(self.l5(h4))

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

		simu_soc = fun_model_simu(h5, forcing, obs_depth)

		
		return simu_soc, h5
# end nn_model

# Helper function to combine the training data into a single tensor
class MergeDataset(Dataset):
    def __init__(self, data_x, data_y, data_z, profile_id):
        self.data_x = data_x
        self.data_y = data_y
        self.data_z = data_z
        self.profile_id = profile_id

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx], self.data_z[idx], self.profile_id[idx]


# Start training
def worker(rank, world_size):
	# Filename to store average losses
	avg_loss_filename = 'avg_loss_' + nn_training_name + '.txt'
	# Initialize the process group
	os.environ['RANK'] = str(rank)
	os.environ['WORLD_SIZE'] = str(world_size)
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = '12355'

	# Initialize distributed environment
	dist.init_process_group('gloo', rank=rank, world_size=world_size)

	# Initialize model
	global model
	model = nn_model(var_idx_to_emb).to(device)
	model = DDP(model)


	# Loss and optimizer
	optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
	fun_loss = binns_loss

	# Initialize datasets
	train_dataset = MergeDataset(train_x, train_y, train_z, train_profile_id)
	val_dataset = MergeDataset(val_x, val_y, val_z, val_profile_id)

	# Use DistributedSampler for distributed training
	train_sampler = DistributedSampler(train_dataset)
	val_sampler = DistributedSampler(val_dataset)

	# Data loaders with DistributedSampler
	train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)
	# train_loader = DataLoader([[train_x[i], train_y[i], train_z[i], train_profile_id[i]] for i in range(train_y.shape[0])], shuffle = True, batch_size = batch_size, num_workers=4)
	# val_loader = DataLoader([[val_x[i], val_y[i], val_z[i], val_profile_id[i]] for i in range(val_y.shape[0])], shuffle = True, batch_size = batch_size, num_workers=4)

	# training and validation loop
	num_epoch = 5000

	# record the loss history
	train_loss_history = np.ones((num_epoch, 1))*np.nan
	val_loss_history = np.ones((num_epoch, 1))*np.nan

	# Early stopping parameters
	best_val_loss = float('inf')  
	patience = 500
	epochs_without_improvement = 0

	for iepoch in range(num_epoch):
		# -------------------------------------training
		loss_record_train = list()
		ibatch = 0
		epoch_start = time.time()
		for batch_info in train_loader:
			batch_x, batch_y, batch_z, batch_profile_id = batch_info
			ibatch = ibatch + 1
		
			# batch_size = batch_x.size(0)
			# batch_x = batch_x.view(batch_size, -1).to(device)
			batch_x = batch_x.to(device)
			batch_y = batch_y.to(device)
			#------------ 1 forward
			batch_y_hat, batch_pred_para = model(batch_x, batch_z)

			# num_cores = os.cpu_count()
			# print(f'Number of cores: {num_cores}')
			# print(psutil.cpu_percent(interval=None, percpu=True))
			# print(psutil.virtual_memory())

			# record the predicted para and modelled soc
			# middle_simu_soc[batch_profile_id, :] = batch_y_hat
			# middle_pred_para[batch_profile_id, :] = batch_pred_para
			
			#------------ 2 compute the objective function
			obj = fun_loss(batch_y_hat, batch_y)
			
			# print(batch_y_hat)
			# print(f'{datetime.now()} Epoch {iepoch + 1} batch {ibatch}, train loss: {obj.item():.2f}')

			#------------ 3 cleaning gradients
			model.zero_grad()
			
			#------------ 4 accumulate partical derivatives of objective respect to parameters
			obj.backward()
			
			#------------ 5 step in the opposite direction of the gradient
			# with torch.no_grad(): para = pata - eta*para.grad # eta is learning rate
			optimizer.step()
			
			loss_record_train.append(obj.item())

			# writer.add_scalar('training loss', obj.item(), iepoch)
			# record prediction
		# end for batch_info in train_loader:
		
		# record the loss history
		# train_loss_history[iepoch, :] = loss_record_train

		# training time
		train_time = time.time() - epoch_start
		
		# print(f'Epoch {iepoch + 1}, Rank {rank}, train loss: {torch.tensor(loss_record_train).mean():.1f}, time: {(time.time()-epoch_start):.2f}')
		# print(f"-----------------Epoch {iepoch + 1} - Rank {rank} - Model Weights: {model.module.l1.weight.data} - {model.module.l2.weight.data} - {model.module.l3.weight.data} - {model.module.l4.weight.data} - {model.module.l5.weight.data}-----------------")


		# writer.add_scalar('training loss', torch.tensor(loss_record_train).mean(), iepoch+1)

		# -------------------------------------validation
		loss_record_val = list()
		ibatch = 0
		for batch_info in val_loader:
			batch_x, batch_y, batch_z, batch_profile_id = batch_info
			ibatch = ibatch + 1
			# batch_size = batch_x.size(0)
			# batch_x = batch_x.view(batch_size, -1).to(device)
			batch_x = batch_x.to(device)
			batch_y = batch_y.to(device)
			# 1 forward
			with torch.no_grad():
				batch_y_hat, batch_pred_para = model(batch_x, batch_z)
				# record the predicted para and modelled soc
				# middle_simu_soc[batch_profile_id, :] = batch_y_hat
				# middle_pred_para[batch_profile_id, :] = batch_pred_para
			# 2 compute the objective function
			
			obj = fun_loss(batch_y_hat, batch_y)

			# if validation loss is nan, print out the batch info
			if np.isnan(obj.item()):
				print(batch_y_hat)
			
			loss_record_val.append(obj.item())
			# print(f'{datetime.now()}, Epoch {iepoch + 1}, Rank {rank}, batch {ibatch}, validation loss: {obj.item():.2f}')
		# end for batch_info in val_loader: 

		# Gather losses from all processes
		all_train_losses = [torch.tensor(0.0) for _ in range(world_size)]
		all_val_losses = [torch.tensor(0.0) for _ in range(world_size)]
		all_train_times = [torch.tensor(0.0) for _ in range(world_size)]

		dist.all_gather(all_train_losses, torch.tensor(loss_record_train).mean())
		dist.all_gather(all_val_losses, torch.tensor(loss_record_val).mean())
		dist.all_gather(all_train_times, torch.tensor(train_time))

		# record the loss history
		train_loss_history[iepoch, :] = torch.stack(all_train_losses).mean()
		val_loss_history[iepoch, :] = torch.stack(all_val_losses).mean()

		if rank == 0:
			writer.add_scalars('loss', {'training': torch.stack(all_train_losses).mean(), 'validation': torch.stack(all_val_losses).mean()}, iepoch+1)
			print(f'Epoch {iepoch + 1}, train loss: {torch.stack(all_train_losses).mean():.2f}, validation loss: {torch.stack(all_val_losses).mean():.2f}, time: {torch.stack(all_train_times).mean():.2f}')
		elif rank == 1:
			with open(os.path.join(data_dir_output, "neural_network", job_id, avg_loss_filename), "a") as f:
				f.write(f'{iepoch + 1}, {torch.stack(all_train_losses).mean():.2f}, {torch.stack(all_val_losses).mean():.2f}, {torch.stack(all_train_times).mean():.2f}\n')
		elif rank == 2: 
			if iepoch == 0:
				# best_simu_soc = middle_simu_soc
				# best_pred_para = middle_pred_para
				print(f'Best model updated at epoch {iepoch + 1}')
			elif train_loss_history[iepoch, :] <= train_loss_history[(iepoch-1), :] and val_loss_history[iepoch, :] <= best_val_loss:
				# best_simu_soc = middle_simu_soc
				# best_pred_para = middle_pred_para
				
				print(f'Best model updated at epoch {iepoch + 1}')
				
				# save prediction and model
				# np.savetxt(data_dir_output + 'neural_network/nn_best_pred_para_' + time_stamp + '.csv', best_pred_para.detach().numpy(), delimiter = ',')
				# np.savetxt(data_dir_output + 'neural_network/nn_best_simu_soc_' + time_stamp + '.csv', best_simu_soc.detach().numpy(), delimiter = ',')
				# torch.save(model, data_dir_output + 'neural_network/' + job_id + '/opt_nn_' + job_id  + '.pt')
				print(f'Best model updated at epoch {iepoch + 1}')
				
				# save prediction and model
				# np.savetxt(data_dir_output + 'neural_network/nn_best_pred_para_' + time_stamp + '.csv', best_pred_para.detach().numpy(), delimiter = ',')
				# np.savetxt(data_dir_output + 'neural_network/nn_best_simu_soc_' + time_stamp + '.csv', best_simu_soc.detach().numpy(), delimiter = ',')
				checkpoint_best_model = {
					'epoch': iepoch,
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'best_val_loss': best_val_loss,
					'train_loss_history': train_loss_history,
					'val_loss_history': val_loss_history,
					'train_indices': train_loc,
					'val_indices': val_loc,
					'test_indices': test_loc,
					'epochs_without_improvement': epochs_without_improvement,
				}
				torch.save(checkpoint_best_model, data_dir_output + 'neural_network/' + job_id + '/opt_nn_' + job_id  + '.pt')
				
				# np.savetxt(data_dir_output + 'neural_network/val_loss_history_' + time_stamp + '.csv', val_loss_history, delimiter = ',')
				# np.savetxt(data_dir_output + 'neural_network/train_loss_history_' + time_stamp + '.csv', train_loss_history, delimiter = ',')
			# end if iepoch == 0:
		
		# Add a early stopping condition
		if val_loss_history[iepoch, :] < best_val_loss:
			best_val_loss = val_loss_history[iepoch, :]
			epochs_without_improvement = 0
			# Optionally save the model here if it's the best one so far
		else:
			epochs_without_improvement += 1

		# Early stopping condition
		if epochs_without_improvement == patience:
			print("Rank {}: Early stopping due to no improvement after {} epochs.".format(rank, patience))
			break  # exit the epoch loop
		
		# If runtimes are over 11.50 hours, save checkpoint and exit
		whether_checkpoint = False
		if time.time() - start_time > 41400:
			if rank == 0:
				print("Rank {}: Runtime exceeded, saving checkpoint and exiting.".format(rank))
				whether_checkpoint = True
				checkpoint = {
					'epoch': iepoch,
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'best_val_loss': best_val_loss,
					'train_loss_history': train_loss_history,
					'val_loss_history': val_loss_history,
					'train_indices': train_loc,
					'val_indices': val_loc,
					'test_indices': test_loc,
					'epochs_without_improvement': epochs_without_improvement,
				}
				torch.save(checkpoint, data_dir_output + 'neural_network/' + job_id + '/checkpoint_' + job_id + '.pt')
				# submit the job again
				submit_command = ['qsub', 
					  '-v', f"PREVIOUS_JOB_ID={job_id}",
					  '/glade/u/home/haodixu/BINN/PBS_Submit/Checkpoint_DDP/Resume.submit']
				# Submit the job and get the new job ID
				try:
					submit_output = subprocess.check_output(submit_command, universal_newlines=True)
					new_job_id = submit_output.strip()
					print(f"New job submitted. New Job ID is {new_job_id}")
				except subprocess.CalledProcessError as e:
					print(f"Failed to submit job: {e.output}")
			break



	print(f"Rank {rank} finished processing data.")

	if whether_checkpoint:
		print("Rank {}: Exiting after saving checkpoint.".format(rank))
		return
	
	##################################################
	# prediction bv best trained model
	##################################################
	# best_guess_model = torch.load(data_dir_output + 'neural_network/' + job_id + '/opt_nn_' + job_id + '.pt').to(device)
	# best_guess_model.eval()
	new_checkpoint = torch.load(data_dir_output + 'neural_network/' + job_id + '/opt_nn_' + job_id + '.pt')
	# best_guess_model = torch.load(data_dir_output + 'neural_network/' + job_id + '/opt_nn_' + job_id + '.pt').to(device)
	best_guess_model = nn_model(var_idx_to_emb).to(device)
	best_guess_model = DDP(best_guess_model)
	best_guess_model.load_state_dict(new_checkpoint['model_state_dict'])
	best_guess_model.eval()
	
	if rank == 0:
		with torch.no_grad():
			best_guess_val_y_hat, best_guess_val_pred_para = best_guess_model(val_x, val_z)
			best_guess_train_y_hat, best_guess_train_pred_para = best_guess_model(train_x, train_z)
			if test_split_ratio != 0:
				best_guess_test_y_hat, best_guess_test_pred_para = best_guess_model(test_x, test_z)
				test_loss = fun_loss(best_guess_test_y_hat, test_y)
				print(f'Test loss: {test_loss.item():.2f}')

		# end with torch.no_grad():

		# write prediction results
		binn_obs_soc = np.ones((wosis_profile_info.shape[0], 200))*np.nan
		best_simu_soc = torch.tensor(np.ones((wosis_profile_info.shape[0], 200))*np.nan, dtype = torch.float32)
		best_pred_para = torch.tensor(np.ones((wosis_profile_info.shape[0], len(para_names)))*np.nan, dtype = torch.float32)

		binn_obs_soc[current_data_profile_id, :] = current_data_y

		best_simu_soc[test_profile_id, :] = best_guess_test_y_hat
		best_pred_para[test_profile_id, :] = best_guess_test_pred_para

		# save data
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/nn_best_simu_soc_test_' + job_id + '.csv', best_simu_soc.detach().numpy(), delimiter = ',')

		# get the latitudes and longitudes of the test profiles by matching ProfileID in env_info with the test_profile_id
		test_lons = np.ones((wosis_profile_info.shape[0]))*np.nan
		test_lats = np.ones((wosis_profile_info.shape[0]))*np.nan
		test_profile_id_all = np.ones((wosis_profile_info.shape[0]))*np.nan
		test_profile_id_all[test_profile_id] = test_profile_id
		test_profile_id_num = test_profile_id.numpy().astype(int)
		test_lons[test_profile_id_num] = np.array(env_info.loc[test_profile_id_num, "original_lon"])
		test_lats[test_profile_id_num] = np.array(env_info.loc[test_profile_id_num, "original_lat"])

		# initialize the scaled difference
		scaled_diff = np.ones((wosis_profile_info.shape[0]))*np.nan

		# for each location, calculate the difference between the predicted and observed SOC values
		for i in range(binn_obs_soc.shape[0]):
			if np.isnan(binn_obs_soc[i, :]).all() or np.isnan(best_simu_soc[i, :]).all():
				continue
			else: 
				# Get the predicted and observed SOC values for this profile
				obs_soc = binn_obs_soc[i, :]
				simu_soc = best_simu_soc[i, :]
				temp_simu_sum = 0
				temp_obs_sum = 0
				for j in range(len(simu_soc)):
					if np.isnan(obs_soc[j]) or np.isnan(simu_soc[j]):
						continue
					else:
						if j >= 25:
							print('outlier: ', test_profile_id_all[i], j, obs_soc[j], simu_soc[j])
							continue
						# Calculate the scaled difference
						temp_simu_sum += simu_soc[j] # * dz[j]
						temp_obs_sum += obs_soc[j] # * dz[j]
				scaled_diff[i] = temp_obs_sum/temp_simu_sum
				# print outlier
				if scaled_diff[i] > 2:
					print('outlier: ', test_profile_id_all[i], scaled_diff[i])

		# Plot the scaled difference
		visualization_utils.plot_observations_world_map(test_lons, test_lats, scaled_diff, PLOT_DIR, "test_scaled_diff_" + job_id)

		best_simu_soc[val_profile_id, :] = best_guess_val_y_hat
		best_pred_para[val_profile_id, :] = best_guess_val_pred_para

		# initializz a seperate array to store the prediction results for the validation profiles
		val_simu_soc = torch.tensor(np.ones((wosis_profile_info.shape[0], 200))*np.nan, dtype = torch.float32)
		val_pred_para = torch.tensor(np.ones((wosis_profile_info.shape[0], len(para_names)))*np.nan, dtype = torch.float32)

		val_simu_soc[val_profile_id, :] = best_guess_val_y_hat
		val_pred_para[val_profile_id, :] = best_guess_val_pred_para

		# save data
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/nn_obs_soc_' + job_id + '.csv', binn_obs_soc, delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/nn_best_simu_soc_test_val_' + job_id + '.csv', best_simu_soc.detach().numpy(), delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/nn_best_pred_para_' + job_id + '.csv', best_pred_para.detach().numpy(), delimiter = ',')

		best_simu_soc[train_profile_id, :] = best_guess_train_y_hat
		# best_pred_para[train_profile_id, :] = best_guess_train_pred_para

		# save data
		np.savetxt(data_dir_output + 'neural_network/' + job_id + '/nn_best_simu_soc_all_' + job_id + '.csv', best_simu_soc.detach().numpy(), delimiter = ',')

		# get the latitudes and longitudes of the validation profiles by matching ProfileID in env_info with the val_profile_id
		val_lons = np.ones((wosis_profile_info.shape[0]))*np.nan
		val_lats = np.ones((wosis_profile_info.shape[0]))*np.nan
		val_profile_id_all = np.ones((wosis_profile_info.shape[0]))*np.nan
		val_profile_id_all[val_profile_id] = val_profile_id
		val_profile_id_num = val_profile_id.numpy().astype(int)
		val_lons[val_profile_id_num] = np.array(env_info.loc[val_profile_id_num, "original_lon"])
		val_lats[val_profile_id_num] = np.array(env_info.loc[val_profile_id_num, "original_lat"])

		# Plot maps to show the scaled difference between the predicted and observed SOC values for validation profiles
		# convert nan to 0
		# binn_obs_soc[np.isnan(binn_obs_soc)] = 0
		# best_simu_soc[torch.isnan(best_simu_soc)] = 0
		# initialize the scaled difference
		scaled_diff = np.ones((wosis_profile_info.shape[0]))*np.nan

		# for each location, calculate the difference between the predicted and observed SOC values
		for i in range(val_simu_soc.shape[0]):
			if np.isnan(val_simu_soc[i, :]).all() or np.isnan(binn_obs_soc[i, :]).all():
				continue
			else: 
				# Get the predicted and observed SOC values for this profile
				obs_soc = binn_obs_soc[i, :]
				# print(obs_soc)
				# print(obs_soc.dtype)
				simu_soc = val_simu_soc[i, :]
				temp_simu = 0
				temp_obs_sum = 0
				for j in range(len(simu_soc)):
					if np.isnan(obs_soc[j]) or np.isnan(simu_soc[j]):
						continue
					else:
						if j >= 25:
							print('outlier: ', val_profile_id_all[i], j, obs_soc[j], simu_soc[j])
							continue
						# Calculate the scaled difference
						temp_simu += simu_soc[j] # * dz[j]
						temp_obs_sum += obs_soc[j] # * dz[j]
				scaled_diff[i] = temp_obs_sum/temp_simu
				# print outlier
				if scaled_diff[i] > 2:
					print('outlier: ', val_profile_id_all[i], scaled_diff[i])


		# Plot the scaled difference
		visualization_utils.plot_observations_world_map(val_lons, val_lats, scaled_diff, PLOT_DIR, "validation_scaled_diff_" + job_id)

		

		

		print("-----------------Model Test Finished at " + str(datetime.now()) + "-----------------")





if __name__ == '__main__':
	# Number of CPUs requester
	world_size = 128
	processes = []
	for rank in range(world_size):
		p = Process(target=worker, args=(rank, world_size))
		p.start()
		processes.append(p)

	for p in processes:
		p.join()

	
