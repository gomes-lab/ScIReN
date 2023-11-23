import sys
import time
import warnings
# @joshuafan changed
# sys.path.append('/mnt/beegfs/bulk/mirror/jyf6/datasets/BINNS/src_binns')
# sys.path.append(r'/User/homes/ftao/Projects/BINNS/src_binns')

from datetime import datetime
from pandas import DataFrame as df
import numpy as np
from scipy.interpolate import pchip_interpolate

import os
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
from torch.nn.utils.parametrizations import spectral_norm

import argparse
import csv
import random
from scipy.io import loadmat
import netCDF4 as ncread 
import mat73

from matplotlib import pyplot as plt

from fun_matrix_clm5 import fun_model_simu
import visualization_utils

# Set device
if torch.cuda.is_available():
	dev = 'cuda'
else:
	dev = 'cpu'
device = torch.device(dev) 
print(datetime.now(), '------------device: ', device, '------------')

print(datetime.now(), '------------all packages loaded------------')

################################################
# Command-line arguments
################################################
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-2)
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--note", type=str, default="", help="Optional name to give to the model")
args = parser.parse_args()

# Set random seeds to try to ensure reproducibility
random.seed(args.seed)
np.random.seed(args.seed) # set the random seed of numpy
torch.manual_seed(args.seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Current timestamp and model name
run_name = time.strftime("%Y%m%d-%H%M%S")  # Convert datetime to string: https://stackoverflow.com/questions/10607688/how-to-create-a-file-name-with-the-current-date-time-in-python
if args.note != "":
	run_name += ("_" + args.note)

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
# data_dir_input = '/Users/ft254/DATAHUB/ENSEMBLE/INPUT_DATA/'
# data_dir_output = '/Users/ft254/DATAHUB/BINNS/OUTPUT_DATA/'
data_dir_input = '/mnt/beegfs/bulk/mirror/jyf6/datasets/BINNS/INPUT_DATA/'
data_dir_output = '/mnt/beegfs/bulk/mirror/jyf6/datasets/BINNS/OUTPUT_DATA/'
os.makedirs(os.path.join(data_dir_output, "neural_network"), exist_ok=True)
PLOT_DIR = os.path.join(data_dir_output, "visualizations")
os.makedirs(PLOT_DIR, exist_ok=True)

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

profile_collection = np.reshape(sample_profile_id[:, 0:20], [2000, 1])

# if use the whole dataset
# profile_collection = np.range(0, wosis_profile_info.shape[0])

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
nn_split_ratio = 0.2
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

col_max_min = loadmat(data_dir_input + 'data4nn/world_grid_envinfo_present_cesm2_clm5_cen_vr_v2_whole_time_col_max_min.mat')
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
	print("Unique values", env_info[group].value_counts().sort_index())
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

# Plot SOC observation labels within each layer. If a profile has multiple observations 
# in a layer, pick the first one
layer_top = 0
for layer_idx in range(len(zisoi)):
	layer_bottom = zisoi[layer_idx]
	this_layer_y = np.ones((current_data_y.shape[0])) * np.nan

	# Loop through all profiles
	for j in range(current_data_y.shape[0]):
		# Get depth of each SOC observation
		depths = current_data_z[j]

		# Select SOC observations whose depth falls within the current layer
		this_layer_this_profile_y = current_data_y[j, (~np.isnan(depths)) & (depths >= layer_top) & (depths < layer_bottom)]
		if len(this_layer_this_profile_y) > 1:
			continue
			print("Oddly enough this profile had more than 2 observations in the same soil layer")
			print("Layer", layer_top, "to", layer_bottom)
			print("Observation depths", depths)
		elif len(this_layer_this_profile_y) == 0:
			continue
		else:
			this_layer_y[j] = this_layer_this_profile_y[0]
	layer_name = "Layer {} ({:.2f}-{:.2f} m)".format(layer_idx, layer_top, layer_bottom)
	col_name = "soc_layer{}_{:.2f}-{:.2f}m".format(layer_idx, layer_top, layer_bottom)
	# visualization_utils.plot_observations_world_map(lons, lats, this_layer_y, PLOT_DIR, col_name)
	layer_top = layer_bottom


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

# train and validation split
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
# train_y, val_y = random_split(current_data_y, [round(current_data_x.shape[0]*0.8), (current_data_x.shape[0] - round(current_data_x.shape[0]*0.8))], generator=torch.Generator().manual_seed(42))
# train_y = torch.tensor(current_data_y[train_y.indices], dtype = torch.float32)
# val_y = torch.tensor(current_data_y[val_y.indices], dtype = torch.float32)

# train_x, val_x = random_split(current_data_x, [round(current_data_x.shape[0]*0.8), (current_data_x.shape[0] - round(current_data_x.shape[0]*0.8))], generator=torch.Generator().manual_seed(42))
# train_x = torch.tensor(current_data_x[train_x.indices], dtype = torch.float32)
# val_x = torch.tensor(current_data_x[val_x.indices], dtype = torch.float32)

# data loader
train_loader = DataLoader([[train_x[i], train_y[i], train_z[i], train_profile_id[i]] for i in range(train_y.shape[0])], shuffle = True, batch_size = batch_size)

val_loader = DataLoader([[val_x[i], val_y[i], val_z[i], val_profile_id[i]] for i in range(val_y.shape[0])], shuffle = True, batch_size = batch_size)

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
nn_training_name = 'exp_pc_binns_1'

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
	
	return modeling_inefficiency
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
		self.l1 = spectral_norm(nn.Linear(new_input_size, 256))
		self.l2 = spectral_norm(nn.Linear(256, 512))
		self.l3 = spectral_norm(nn.Linear(512, 512))
		self.l4 = spectral_norm(nn.Linear(512, 256))
		self.l5 = nn.Linear(256, 21)

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
		h2 = nn.functional.relu(self.l2(h1))
		h3 = nn.functional.relu(self.l3(h2))
		h4 = nn.functional.relu(self.l4(h3))
		h5 = torch.sigmoid(self.l5(h4)/100) # hardtanh(self.l5(h4))
		# biogeochemical model
		simu_soc = fun_model_simu(h5, forcing, obs_depth)
		return simu_soc, h5

model = nn_model(var_idx_to_emb).to(device)


# nn_model = nn.Sequential(
#	nn.Linear(len(var4nn), 256),
#	nn.ReLU(),
#	nn.Linear(256, 512),
#	nn.ReLU(),
#	nn.Linear(512, 512),
#	nn.ReLU(),
#	nn.Linear(512, 256),
#	nn.ReLU(),
#	nn.Linear(256, 21),
#	nn.Hardtanh()
#)
#
#model = nn_model.to(device)

# optimizer - @joshuafan changed
# optimizer = torch.optim.Adadelta(model.parameters())
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# loss
fun_loss = binns_loss

print(datetime.now(), '------------neural network set, training started------------')

# training and validation loop
num_epoch = 50

# best_simu_soc = torch.tensor(np.ones((wosis_profile_info.shape[0], 200))*np.nan, dtype = torch.float32)
# best_pred_para = torch.tensor(np.ones((wosis_profile_info.shape[0], len(para_names)))*np.nan, dtype = torch.float32)
# middle_simu_soc = torch.tensor(np.ones((wosis_profile_info.shape[0], 200))*np.nan, dtype = torch.float32)
# middle_pred_para = torch.tensor(np.ones((wosis_profile_info.shape[0], len(para_names)))*np.nan, dtype = torch.float32)

train_loss_history = np.ones((num_epoch, int(np.ceil(train_y.shape[0]/batch_size))))*np.nan
val_loss_history = np.ones((num_epoch, int(np.ceil(val_y.shape[0]/batch_size))))*np.nan

# Save best validation loss, best epoch, best model path
best_val_loss = float('inf')
best_epoch = -1
best_model_path = data_dir_output + 'neural_network/opt_nn_' + run_name + '.pt'

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
		batch_x = batch_x.to(device)  # [batch_size, 60, 12, 13]. 60 is number of environmental variables (?), 12 is number of months, 13 is number of monthly forcing variables.
		batch_y = batch_y.to(device)  # [batch_size, 200]. 200 is max possible SOC observations per location (most have much fewer)

		#------------ 1 forward
		batch_y_hat, batch_pred_para = model(batch_x, batch_z)

		# record the predicted para and modelled soc
		# middle_simu_soc[batch_profile_id, :] = batch_y_hat
		# middle_pred_para[batch_profile_id, :] = batch_pred_para
		
		#------------ 2 compute the objective function
		obj = fun_loss(batch_y_hat, batch_y)
		
		# print(batch_y_hat)
		print(f'{datetime.now()} Epoch {iepoch + 1} batch {ibatch}, train loss: {obj.item():.2f}')

		#------------ 3 cleaning gradients
		model.zero_grad()
		
		#------------ 4 accumulate partical derivatives of objective respect to parameters
		obj.backward()
		
		#------------ 5 step in the opposite direction of the gradient
		# with torch.no_grad(): para = pata - eta*para.grad # eta is learning rate
		optimizer.step()
		
		loss_record_train.append(obj.item())
		# record prediction
	# end for batch_info in train_loader:
	
	# record the loss history
	train_loss_history[iepoch, :] = loss_record_train
	
	print(f'Epoch {iepoch + 1}, train loss: {torch.tensor(loss_record_train).mean():.1f}, time: {(time.time()-epoch_start):.2f}')

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
		
		loss_record_val.append(obj.item())
		print(f'{datetime.now()}, Epoch {iepoch + 1} batch {ibatch}, validation loss: {obj.item():.2f}')
	# end for batch_info in val_loader: 

	val_loss_history[iepoch, :] = loss_record_val
	print(f'Epoch {iepoch + 1}, validation loss: {torch.tensor(loss_record_val).mean():.2f}')

	# Average validation loss across all batches
	val_loss = val_loss_history[iepoch, :].mean()

	#----------------------------------- find the best prediction
	# if iepoch == 0:
	# 	# best_simu_soc = middle_simu_soc
	# 	# best_pred_para = middle_pred_para
		
	# 	print(f'Best model updated at epoch {iepoch + 1}')
	if val_loss < best_val_loss:  # train_loss_history[iepoch, :].mean() <= train_loss_history[(iepoch-1), :].mean():
		# best_simu_soc = middle_simu_soc
		# best_pred_para = middle_pred_para
		best_val_loss = val_loss
		best_epoch = iepoch + 1
		
		print(f'Best model updated at epoch {iepoch + 1}')
		
		# save prediction and model
		# np.savetxt(data_dir_output + 'neural_network/nn_best_pred_para_' + run_name + '.csv', best_pred_para.detach().numpy(), delimiter = ',')
		# np.savetxt(data_dir_output + 'neural_network/nn_best_simu_soc_' + run_name + '.csv', best_simu_soc.detach().numpy(), delimiter = ',')
		torch.save(model.state_dict(), best_model_path)
		np.savetxt(data_dir_output + 'neural_network/val_loss_history_' + run_name + '.csv', val_loss_history, delimiter = ',')
		np.savetxt(data_dir_output + 'neural_network/train_loss_history_' + run_name + '.csv', train_loss_history, delimiter = ',')
	# end if iepoch == 0:


# save loss history at the end
np.savetxt(data_dir_output + 'neural_network/val_loss_history_' + run_name + '.csv', val_loss_history, delimiter = ',')
np.savetxt(data_dir_output + 'neural_network/train_loss_history_' + run_name + '.csv', train_loss_history, delimiter = ',')


##################################################
# prediction bv best trained model
##################################################
print(f'Best validation loss: {best_val_loss:.3f} (epoch {best_epoch})')
model.load_state_dict(torch.load(best_model_path, map_location=device))
best_guess_model = model
best_guess_model.eval()

with torch.no_grad():
	best_guess_val_y_hat, best_guess_val_pred_para = best_guess_model(val_x.to(device), val_z.to(device))
	best_guess_train_y_hat, best_guess_train_pred_para = best_guess_model(train_x.to(device), train_z.to(device))
# end with torch.no_grad():

# write prediction results
binn_obs_soc = np.ones((wosis_profile_info.shape[0], 200))*np.nan
best_simu_soc = torch.tensor(np.ones((wosis_profile_info.shape[0], 200))*np.nan, dtype = torch.float32, device=device)
best_pred_para = torch.tensor(np.ones((wosis_profile_info.shape[0], len(para_names)))*np.nan, dtype = torch.float32, device=device)

binn_obs_soc[current_data_profile_id, :] = current_data_y

best_simu_soc[val_profile_id, :] = best_guess_val_y_hat
best_simu_soc[train_profile_id, :] = best_guess_train_y_hat

best_pred_para[val_profile_id, :] = best_guess_val_pred_para
best_pred_para[train_profile_id, :] = best_guess_train_pred_para

# save data
np.savetxt(data_dir_output + 'neural_network/nn_obs_soc_' + run_name + '.csv', binn_obs_soc, delimiter = ',')
np.savetxt(data_dir_output + 'neural_network/nn_best_simu_soc_' + run_name + '.csv', best_simu_soc.detach().cpu().numpy(), delimiter = ',')
np.savetxt(data_dir_output + 'neural_network/nn_best_pred_para_' + run_name + '.csv', best_pred_para.detach().cpu().numpy(), delimiter = ',')

# Summary csv file of all results. Create this if it doesn't exist
results_summary_file = os.path.join(data_dir_output, "neural_network/results_summary.csv")
if not os.path.isfile(results_summary_file):
	with open(results_summary_file, mode='w') as f:
		csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		csv_writer.writerow(['run_name', 'git_commit', 'command', 'lr', 'weight_decay', 'seed', 'model_path', 'val_loss'])
git_commit = visualization_utils.get_git_revision_hash()
command_string = " ".join(sys.argv)

# Add a row to the summary csv file
with open(results_summary_file, mode='a+') as f:
	csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	csv_writer.writerow([run_name, git_commit, command_string, args.lr, args.weight_decay, args.seed, best_model_path, best_val_loss])







