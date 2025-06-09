import sys
import random
sys.path.append(r'C:/Users/hx293/Research_Data/BINN/')

# Set HDF5_DISABLE_VERSION_CHECK to suppress version mismatch error
import os
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2'

import warnings
import time
from datetime import datetime
from pandas import DataFrame as df
import pandas as pd
import numpy as np
import numbers
from scipy.interpolate import pchip_interpolate

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from scipy.io import loadmat
import netCDF4 as ncread 
import mat73

from matplotlib import pyplot as plt

# from fun_matrix_clm5 import fun_model_simu
from fun_matrix_COMPAS_sensitivity import fun_model_sensitivity

if torch.cuda.is_available():
	dev = 'cuda'
else:
	dev = 'cpu'
# dev = 'cpu'
device = torch.device(dev) 

time_stamp = f'{datetime.date(datetime.now())}'

print(datetime.now(), '------------device: ', device, '------------')

print(datetime.now(), '------------all packages loaded------------')

random_seed = 111

# @joshuafan: Set random seeds to try to ensure reproducibility
random.seed(random_seed)
np.random.seed(random_seed) # set the random seed of numpy
torch.manual_seed(random_seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

dev = 'cpu'
# @joshuafan changed
# dev = 'cpu'
device = torch.device(dev) 
print(datetime.now(), '------------device: ', device, '------------')

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
data_dir_input = 'D:/BINN/ENSEMBLE/INPUT_DATA/'
data_dir_output = 'D:/BINN/OUTPUT_DATA/Sensitivity_Test/'
os.makedirs(data_dir_output, exist_ok=True)

################################
## Sensitivity Test Constants ##
################################
sensitivity_test_num = 100


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
# CLM5 constants
#-------------------------------
para_names = ['diffus', 'cryo', 'q10', 'efolding', 'taucwd', 'taul1', 'taul2', 'tau4doc', 'tau4mic', 'tau4poc', 'tau4maom','fl1_DOC', 'fl1_MIC', 'fl2_MIC', 'fl2_MAOM', 'fMIC_DOC', 'fMIC_POC',  'fPOC_DOC', 'CUEl1', 'CUEl2', 'CUEDOC', 'w-scaling', 'beta']
# para_names = ['diffus', 'cryo', 'q10', 'efolding', 'taucwd', 'taul1', 'taul2', 'tau4s1', 'tau4s2', 'tau4s3', 'fl1s1', 'fl2s1', 'fl3s2', 'fs1s2', 'fs1s3', 'fs2s1', 'fs2s3', 'fs3s1', 'fcwdl2', 'w-scaling']
para_index = np.arange(len(para_names))
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
sample_profile_id = loadmat(data_dir_input + 'POM_MAOM/eligible_profile_loc_1_cesm2_clm5_cen_vr_v2_whole_time.mat')
sample_profile_id = sample_profile_id['eligible_loc_1']
# convert the number to be starting from 0 in python world
sample_profile_id = sample_profile_id - 1

profile_collection = sample_profile_id

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
# Depth of each observation
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
	# profile id in WOSIS data
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
		# exclude nan values and negative values in wosis_layer_obs
		valid_soc_loc = np.where((np.isnan(wosis_layer_obs) == False) & (np.isnan(wosis_layer_depth) == False) & (np.isnan(wosis_layer_upper_depth) == False) & (np.isnan(wosis_layer_lower_depth) == False) & (wosis_layer_obs > 0))
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
	# Profile with POM and MAOM data
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
# Save obs_POM_matrix into csv file
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
'R_Squared', \
'Ald_0_20',	'Ald_20_40', 'Ald_40_60', 'Ald_60_80', 'Ald_80_100', 'Alo_0_20', 'Alo_20_40', 'Alo_40_60', 'Alo_60_80', 'Alo_80_100',\
'Fed_0_20', 'Fed_20_40', 'Fed_40_60', 'Fed_60_80', 'Fed_80_100', 'Feo_0_20', 'Feo_20_40', 'Feo_40_60', 'Feo_60_80', 'Feo_80_100'
]

categorical_vars = [['ESA_Land_Cover'], ['Texture_USDA_0cm', 'Texture_USDA_30cm', 'Texture_USDA_100cm'], 
					['USDA_Suborder'], ['WRB_Subgroup']]  # Variables inside a sub-list share the same categories
categorical_vars_flattened = [item for sublist in categorical_vars for item in sublist]

# Environment info data for WOSIS profiles
env_info = np.genfromtxt(data_dir_input + 'wosis_2019_snap_shot/env_info_SOC_with_Al_Fe.csv', delimiter = ',', skip_header = 1)
# Environment info data for POM and MAOM profiles
env_info_POM_MAOM = np.genfromtxt(data_dir_input + 'POM_MAOM/env_info_POM_MAOM_with_Al_Fe.csv', delimiter = ',', skip_header = 1)
# Merge the two datasets
env_info = np.vstack((env_info, env_info_POM_MAOM))

original_lons = env_info[:, 3].copy()
original_lats = env_info[:, 4].copy()

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
		# print("Var {} Nans {}".format(var, np.count_nonzero(np.isnan(env_info[:, idx]))))
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
# 'Veg_Cover', \
'BIO1', \
# 'BIO2', 'BIO3', 'BIO4', 'BIO5', 'BIO6', 'BIO7', 'BIO8', 'BIO9', 'BIO10', 'BIO11', \
'BIO12', \
# 'BIO13', 'BIO14', 'BIO15', 'BIO16', 'BIO17', 'BIO18', 'BIO19', \
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
# 'Koppen_Climate_2018', \
'cesm2_npp', 'cesm2_npp_std', \
# 'cesm2_gpp', 'cesm2_gpp_std', \
'cesm2_vegc', \
'nbedrock', \
# 'R_Squared', \
'Ald_0_20',	'Ald_20_40', 'Ald_40_60', 'Ald_60_80', 'Ald_80_100', 'Alo_0_20', 'Alo_20_40', 'Alo_40_60', 'Alo_60_80', 'Alo_80_100',\
'Fed_0_20', 'Fed_20_40', 'Fed_40_60', 'Fed_60_80', 'Fed_80_100', 'Feo_0_20', 'Feo_20_40', 'Feo_40_60', 'Feo_60_80', 'Feo_80_100'
]

##########################################
# If using another var4nn list
##########################################
var4nn = ["BIO1", "BIO12", "Clay_Content_avg", "Sand_Content_avg", "Bulk_Density_avg", "SWC_v_Wilting_Point_avg", "pH_Water_avg", "CEC_avg", "cesm2_npp", "cesm2_vegc"]
categorical_vars = []

# Create new columns for the average values of the three layers
env_info["Clay_Content_avg"] = (env_info["Clay_Content_0cm"] + env_info["Clay_Content_30cm"] + env_info["Clay_Content_100cm"]) / 3
env_info["Sand_Content_avg"] = (env_info["Sand_Content_0cm"] + env_info["Sand_Content_30cm"] + env_info["Sand_Content_100cm"]) / 3
env_info["Bulk_Density_avg"] = (env_info["Bulk_Density_0cm"] + env_info["Bulk_Density_30cm"] + env_info["Bulk_Density_100cm"]) / 3
env_info["SWC_v_Wilting_Point_avg"] = (env_info["SWC_v_Wilting_Point_0cm"] + env_info["SWC_v_Wilting_Point_30cm"] + env_info["SWC_v_Wilting_Point_100cm"]) / 3
env_info["pH_Water_avg"] = (env_info["pH_Water_0cm"] + env_info["pH_Water_30cm"] + env_info["pH_Water_100cm"]) / 3
env_info["CEC_avg"] = (env_info["CEC_0cm"] + env_info["CEC_30cm"] + env_info["CEC_100cm"]) / 3



#---------------------------------------------------
# training data
#---------------------------------------------------
current_data_x = np.ones((len(profile_collection),  max(len(var4nn), 20), 12, 13))*np.nan
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

# Include SOC, POM, and MAOM data in current_data_y
current_data_y = np.ones((len(profile_collection), 3, 200))*np.nan
current_data_y[:, 0, :] = obs_soc_matrix
current_data_y[:, 1, :] = obs_POM_matrix
current_data_y[:, 2, :] = obs_MAOM_matrix

# Check if there're any negative values in the data
print("Negative values in current_data_y", np.sum(current_data_y < 0))


current_data_z = obs_depth_matrix

lons = np.array(env_info.loc[profile_collection[:, 0], "original_lon"])
lats = np.array(env_info.loc[profile_collection[:, 0], "original_lat"])


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
			np.sum(model_force_soil_water_profile, axis = (1, 2)) \
			# + np.nanmean(current_data_y[:, 0, :], axis = 1) 

valid_profile_loc = np.where(np.isnan(nan_loc) == False)[0]

# Sample 1000 profiles from the valid profiles
valid_profile_loc = np.random.choice(valid_profile_loc, size=min(1000, len(valid_profile_loc)), replace=False)

current_data_y = current_data_y[valid_profile_loc, :, :]
current_data_z = current_data_z[valid_profile_loc, :]
current_data_x = current_data_x[valid_profile_loc, :, :, :]
current_data_profile_id = profile_collection[valid_profile_loc, 0]
obs_upper_depth_matrix = obs_upper_depth_matrix[valid_profile_loc, :]
obs_lower_depth_matrix = obs_lower_depth_matrix[valid_profile_loc, :]
print("Shape of current data x", current_data_x.shape)
print("Shape of current data y", current_data_y.shape)
print("Shape of current data z", current_data_z.shape)
print("Shape of obs upper depth matrix", obs_upper_depth_matrix.shape)
print("Shape of obs lower depth matrix", obs_lower_depth_matrix.shape)


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
def worker():
	start_time = time.time()
	warnings.filterwarnings("ignore")
	# Define parameter names
	para_name = ['diffus', 'cryo', 'q10', 'efolding', 'taucwd', 'taul1', 'taul2', 'tau4doc', 'tau4mic', 'tau4poc', 'tau4maom','fl1_MIC', 'fl2_MIC', 'fMIC_DOC', 'fMIC_POC', 'fDOC_MAOM', 'CUEl1', 'CUEl2', 'CUEDOC', 'w-scaling', 'beta']
	# Define the prior range for each parameter in the order of para_name
	prior_range = [[3e-5, 5e-4], [3e-5, 16e-4], [1.2, 3], [0.1, 1], [1, 6], [0.0001, 0.11], [0.1, 0.3], [0.0001, 1], [0.0001, 1], [1, 10], [1, 200], [0.0001, 0.9], [0.0001, 0.9], [0.3, 0.8], [0.0001, 0.2], [0.0001, 0.2], [0.0001, 0.6], [0.0001, 0.4], [0.1, 0.99], [0.0001, 5], [0.5, 0.9999]]
	para_size = len(para_name)
	# Initialize datasets
	dataset = MergeDataset(current_data_x, current_data_y, current_data_z, current_data_profile_id)
	# Load datasets into dataloaders
	dist_loader = DataLoader(dataset, batch_size=1, shuffle=True)

	# Define number of batches for sensitivity analysis
	num_batches = 32  # Number of batches to run for sensitivity analysis

	# Initialize the tensor to store the sensitivity
	sensitivity_soc_all = torch.zeros([para_size, num_batches], requires_grad=False, device=device)
	sensitivity_soc_0_30 = torch.zeros([para_size, num_batches], requires_grad=False, device=device)
	sensitivity_soc_30_100 = torch.zeros([para_size, num_batches], requires_grad=False, device=device)
	sensitivity_soc_100_ = torch.zeros([para_size, num_batches], requires_grad=False, device=device)

	sensitivity_soc_all_POM = torch.zeros([para_size, num_batches], requires_grad=False, device=device)
	sensitivity_soc_0_30_POM = torch.zeros([para_size, num_batches], requires_grad=False, device=device)
	sensitivity_soc_30_100_POM = torch.zeros([para_size, num_batches], requires_grad=False, device=device)
	sensitivity_soc_100_POM = torch.zeros([para_size, num_batches], requires_grad=False, device=device)

	sensitivity_soc_all_MAOM = torch.zeros([para_size, num_batches], requires_grad=False, device=device)
	sensitivity_soc_0_30_MAOM = torch.zeros([para_size, num_batches], requires_grad=False, device=device)
	sensitivity_soc_30_100_MAOM = torch.zeros([para_size, num_batches], requires_grad=False, device=device)
	sensitivity_soc_100_MAOM = torch.zeros([para_size, num_batches], requires_grad=False, device=device)

	for batch_idx in range(num_batches):
		# Randomly select one profile from dist_loader
		batch_x, batch_y, batch_z, batch_profile_id = next(iter(dist_loader))
		batch_x = batch_x.to(device)
		batch_y = batch_y.to(device)
		batch_z = batch_z.to(device)
		batch_profile_id = batch_profile_id.to(device)
		
		#####################
		# Observed Variable #
		#####################
		# Initialize the parameter list
		base_run_step = 100
		obs_para = torch.zeros([len(para_name)], requires_grad=False, device=device)
		# Initialize the tensor to store the observed SOC
		obs_soc_all = torch.zeros([base_run_step], requires_grad=False, device=device)
		obs_soc_0_30 = torch.zeros([base_run_step], requires_grad=False, device=device)
		obs_soc_30_100 = torch.zeros([base_run_step], requires_grad=False, device=device)
		obs_soc_100_ = torch.zeros([base_run_step], requires_grad=False, device=device)

		obs_POM_all = torch.zeros([base_run_step], requires_grad=False, device=device)
		obs_POM_0_30 = torch.zeros([base_run_step], requires_grad=False, device=device)
		obs_POM_30_100 = torch.zeros([base_run_step], requires_grad=False, device=device)
		obs_POM_100_ = torch.zeros([base_run_step], requires_grad=False, device=device)

		obs_MAOM_all = torch.zeros([base_run_step], requires_grad=False, device=device)
		obs_MAOM_0_30 = torch.zeros([base_run_step], requires_grad=False, device=device)
		obs_MAOM_30_100 = torch.zeros([base_run_step], requires_grad=False, device=device)
		obs_MAOM_100_ = torch.zeros([base_run_step], requires_grad=False, device=device)

		for idx_obs in range(base_run_step):
			# Freely select parameters from prior range
			for ipara in range(len(para_name)):
				# obs_para[ipara] = torch.tensor(np.random.normal(prior_range[ipara][0], prior_range[ipara][1]), requires_grad=False, device=device)
				obs_para[ipara] = torch.tensor(np.random.uniform(0, 1) * (prior_range[ipara][1] - prior_range[ipara][0]) + prior_range[ipara][0], requires_grad=False, device=device)
			# end for parameter selection
			
			# Initialize the model
			temp_obs_soc, temp_obs_POM, temp_obs_MAOM = fun_model_sensitivity(obs_para, batch_x)

			# Calculate the sum if not all nan, otherwise set to nan
			obs_soc_all[idx_obs] = (torch.nansum(temp_obs_soc) if not torch.all(torch.isnan(temp_obs_soc)) else torch.tensor(np.nan, device=device))
			obs_soc_0_30[idx_obs] = (torch.nansum(temp_obs_soc[0, 0:6]) if not torch.all(torch.isnan(temp_obs_soc[0, 0:6])) else torch.tensor(np.nan, device=device))
			obs_soc_30_100[idx_obs] = (torch.nansum(temp_obs_soc[0, 6:9]) if not torch.all(torch.isnan(temp_obs_soc[0, 6:9])) else torch.tensor(np.nan, device=device))
			obs_soc_100_[idx_obs] = (torch.nansum(temp_obs_soc[0, 9:]) if not torch.all(torch.isnan(temp_obs_soc[0, 9:])) else torch.tensor(np.nan, device=device))

			obs_POM_all[idx_obs] = (torch.nansum(temp_obs_POM) if not torch.all(torch.isnan(temp_obs_POM)) else torch.tensor(np.nan, device=device))
			obs_POM_0_30[idx_obs] = (torch.nansum(temp_obs_POM[0, 0:6]) if not torch.all(torch.isnan(temp_obs_POM[0, 0:6])) else torch.tensor(np.nan, device=device))
			obs_POM_30_100[idx_obs] = (torch.nansum(temp_obs_POM[0, 6:9]) if not torch.all(torch.isnan(temp_obs_POM[0, 6:9])) else torch.tensor(np.nan, device=device))
			obs_POM_100_[idx_obs] = (torch.nansum(temp_obs_POM[0, 9:]) if not torch.all(torch.isnan(temp_obs_POM[0, 9:])) else torch.tensor(np.nan, device=device))

			obs_MAOM_all[idx_obs] = (torch.nansum(temp_obs_MAOM) if not torch.all(torch.isnan(temp_obs_MAOM)) else torch.tensor(np.nan, device=device))
			obs_MAOM_0_30[idx_obs] = (torch.nansum(temp_obs_MAOM[0, 0:6]) if not torch.all(torch.isnan(temp_obs_MAOM[0, 0:6])) else torch.tensor(np.nan, device=device))
			obs_MAOM_30_100[idx_obs] = (torch.nansum(temp_obs_MAOM[0, 6:9]) if not torch.all(torch.isnan(temp_obs_MAOM[0, 6:9])) else torch.tensor(np.nan, device=device))
			obs_MAOM_100_[idx_obs] = (torch.nansum(temp_obs_MAOM[0, 9:]) if not torch.all(torch.isnan(temp_obs_MAOM[0, 9:])) else torch.tensor(np.nan, device=device))

			# if any values over 1e15, set to 1e15
			obs_soc_all[idx_obs] = torch.clamp(obs_soc_all[idx_obs], max=1e15)
			obs_soc_0_30[idx_obs] = torch.clamp(obs_soc_0_30[idx_obs], max=1e15)
			obs_soc_30_100[idx_obs] = torch.clamp(obs_soc_30_100[idx_obs], max=1e15)
			obs_soc_100_[idx_obs] = torch.clamp(obs_soc_100_[idx_obs], max=1e15)

			obs_POM_all[idx_obs] = torch.clamp(obs_POM_all[idx_obs], max=1e15)
			obs_POM_0_30[idx_obs] = torch.clamp(obs_POM_0_30[idx_obs], max=1e15)
			obs_POM_30_100[idx_obs] = torch.clamp(obs_POM_30_100[idx_obs], max=1e15)
			obs_POM_100_[idx_obs] = torch.clamp(obs_POM_100_[idx_obs], max=1e15)

			obs_MAOM_all[idx_obs] = torch.clamp(obs_MAOM_all[idx_obs], max=1e15)
			obs_MAOM_0_30[idx_obs] = torch.clamp(obs_MAOM_0_30[idx_obs], max=1e15)
			obs_MAOM_30_100[idx_obs] = torch.clamp(obs_MAOM_30_100[idx_obs], max=1e15)
			obs_MAOM_100_[idx_obs] = torch.clamp(obs_MAOM_100_[idx_obs], max=1e15)


		# end for idx_obs in range(100)
		# Remove nan, inf, and zero
		obs_soc_all = obs_soc_all[~torch.isnan(obs_soc_all) & ~torch.isinf(obs_soc_all) & (obs_soc_all >= 0)]
		obs_soc_0_30 = obs_soc_0_30[~torch.isnan(obs_soc_0_30) & ~torch.isinf(obs_soc_0_30) & (obs_soc_0_30 >= 0)]
		obs_soc_30_100 = obs_soc_30_100[~torch.isnan(obs_soc_30_100) & ~torch.isinf(obs_soc_30_100) & (obs_soc_30_100 >= 0)]
		obs_soc_100_ = obs_soc_100_[~torch.isnan(obs_soc_100_) & ~torch.isinf(obs_soc_100_) & (obs_soc_100_ >= 0)]

		obs_POM_all = obs_POM_all[~torch.isnan(obs_POM_all) & ~torch.isinf(obs_POM_all) & (obs_POM_all >= 0)]
		obs_POM_0_30 = obs_POM_0_30[~torch.isnan(obs_POM_0_30) & ~torch.isinf(obs_POM_0_30) & (obs_POM_0_30 >= 0)]
		obs_POM_30_100 = obs_POM_30_100[~torch.isnan(obs_POM_30_100) & ~torch.isinf(obs_POM_30_100) & (obs_POM_30_100 >= 0)]
		obs_POM_100_ = obs_POM_100_[~torch.isnan(obs_POM_100_) & ~torch.isinf(obs_POM_100_) & (obs_POM_100_ >= 0)]

		obs_MAOM_all = obs_MAOM_all[~torch.isnan(obs_MAOM_all) & ~torch.isinf(obs_MAOM_all) & (obs_MAOM_all >= 0)]
		obs_MAOM_0_30 = obs_MAOM_0_30[~torch.isnan(obs_MAOM_0_30) & ~torch.isinf(obs_MAOM_0_30) & (obs_MAOM_0_30 >= 0)]
		obs_MAOM_30_100 = obs_MAOM_30_100[~torch.isnan(obs_MAOM_30_100) & ~torch.isinf(obs_MAOM_30_100) & (obs_MAOM_30_100 >= 0)]
		obs_MAOM_100_ = obs_MAOM_100_[~torch.isnan(obs_MAOM_100_) & ~torch.isinf(obs_MAOM_100_) & (obs_MAOM_100_ >= 0)]
		# Calculate the variance of the observed SOC
		obs_soc_all_var = torch.var(obs_soc_all)
		obs_soc_0_30_var = torch.var(obs_soc_0_30)
		obs_soc_30_100_var = torch.var(obs_soc_30_100)
		obs_soc_100_var = torch.var(obs_soc_100_)

		obs_POM_all_var = torch.var(obs_POM_all)
		obs_POM_0_30_var = torch.var(obs_POM_0_30)
		obs_POM_30_100_var = torch.var(obs_POM_30_100)
		obs_POM_100_var = torch.var(obs_POM_100_)

		obs_MAOM_all_var = torch.var(obs_MAOM_all)
		obs_MAOM_0_30_var = torch.var(obs_MAOM_0_30)
		obs_MAOM_30_100_var = torch.var(obs_MAOM_30_100)
		obs_MAOM_100_var = torch.var(obs_MAOM_100_)

		# Print the variance of the observed SOC
		print('Calculating sensitivity for batch: ', batch_idx, ' out of ', num_batches)
		print('---------------------SOC---------------------')
		print('Variance of the observed SOC at all layers: ', obs_soc_all_var)
		print('Variance of the observed SOC at 0-30cm: ', obs_soc_0_30_var)
		print('Variance of the observed SOC at 30-100cm: ', obs_soc_30_100_var)
		print('Variance of the observed SOC at 100-: ', obs_soc_100_var)
		print('---------------------POM---------------------')
		print('Variance of the observed POM at all layers: ', obs_POM_all_var)
		print('Variance of the observed POM at 0-30cm: ', obs_POM_0_30_var)
		print('Variance of the observed POM at 30-100cm: ', obs_POM_30_100_var)
		print('Variance of the observed POM at 100-: ', obs_POM_100_var)
		print('---------------------MAOM---------------------')
		print('Variance of the observed MAOM at all layers: ', obs_MAOM_all_var)
		print('Variance of the observed MAOM at 0-30cm: ', obs_MAOM_0_30_var)
		print('Variance of the observed MAOM at 30-100cm: ', obs_MAOM_30_100_var)
		print('Variance of the observed MAOM at 100-: ', obs_MAOM_100_var)

		print('Observed SOC variance calculated in ', time.time() - start_time, ' seconds')

		####################
		# Sensitivity Test #
		####################
		# initialize the parameter list
		test_para = obs_para.clone().detach()

		# Loop through all parameters
		test_run_step = 100
		test_start_time = time.time()
		for ipara in range(len(para_name)):
			para_cal_start_time = time.time()
			# Initialize the tensor to store the sensitivity
			temp_sensitivity_all = torch.zeros([test_run_step], requires_grad=False, device=device)
			temp_sensitivity_0_30 = torch.zeros([test_run_step], requires_grad=False, device=device)
			temp_sensitivity_30_100 = torch.zeros([test_run_step], requires_grad=False, device=device)
			temp_sensitivity_100_ = torch.zeros([test_run_step], requires_grad=False, device=device)

			temp_sensitivity_all_POM = torch.zeros([test_run_step], requires_grad=False, device=device)
			temp_sensitivity_0_30_POM = torch.zeros([test_run_step], requires_grad=False, device=device)
			temp_sensitivity_30_100_POM = torch.zeros([test_run_step], requires_grad=False, device=device)
			temp_sensitivity_100_POM = torch.zeros([test_run_step], requires_grad=False, device=device)

			temp_sensitivity_all_MAOM = torch.zeros([test_run_step], requires_grad=False, device=device)
			temp_sensitivity_0_30_MAOM = torch.zeros([test_run_step], requires_grad=False, device=device)
			temp_sensitivity_30_100_MAOM = torch.zeros([test_run_step], requires_grad=False, device=device)
			temp_sensitivity_100_MAOM = torch.zeros([test_run_step], requires_grad=False, device=device)

			for idx_test in range(test_run_step):
				# Initialize the parameter for sensitivity test
				# test_para[ipara] = torch.tensor(np.random.normal(prior_range[ipara][0], prior_range[ipara][1]), requires_grad=False, device=device)
				test_para[ipara] = torch.tensor(np.random.uniform(0, 1) * (prior_range[ipara][1] - prior_range[ipara][0]) + prior_range[ipara][0], requires_grad=False, device=device)
				
				# Initialize the tensor to store the simulated SOC
				temp_soc_all = torch.zeros([test_run_step], requires_grad=False, device=device)
				temp_soc_0_30 = torch.zeros([test_run_step], requires_grad=False, device=device)
				temp_soc_30_100 = torch.zeros([test_run_step], requires_grad=False, device=device)
				temp_soc_100_ = torch.zeros([test_run_step], requires_grad=False, device=device)

				temp_POM_all = torch.zeros([test_run_step], requires_grad=False, device=device)
				temp_POM_0_30 = torch.zeros([test_run_step], requires_grad=False, device=device)
				temp_POM_30_100 = torch.zeros([test_run_step], requires_grad=False, device=device)
				temp_POM_100_ = torch.zeros([test_run_step], requires_grad=False, device=device)

				temp_MAOM_all = torch.zeros([test_run_step], requires_grad=False, device=device)
				temp_MAOM_0_30 = torch.zeros([test_run_step], requires_grad=False, device=device)
				temp_MAOM_30_100 = torch.zeros([test_run_step], requires_grad=False, device=device)
				temp_MAOM_100_ = torch.zeros([test_run_step], requires_grad=False, device=device)

				# loop through other parameters to calculate the sensitivity
				for idx_temp in range(test_run_step):
					# Freely select parameters from prior range
					for ipara_temp in range(len(para_name)):
						if ipara_temp != ipara:
							# test_para[ipara_temp] = torch.tensor(np.random.normal(prior_range[ipara_temp][0], prior_range[ipara_temp][1]), requires_grad=False, device=device)
							test_para[ipara_temp] = torch.tensor(np.random.uniform(0, 1) * (prior_range[ipara_temp][1] - prior_range[ipara_temp][0]) + prior_range[ipara_temp][0], requires_grad=False, device=device)
						# end if ipara_temp != ipara
					# end for ipara_temp in range(len(para_name))
					
					# Initialize the model
					temp_soc, temp_POM, temp_MAOM = fun_model_sensitivity(test_para, batch_x)

					temp_soc_all[idx_temp] = (torch.nansum(temp_soc) if not torch.all(torch.isnan(temp_soc)) else torch.tensor(np.nan, device=device))
					temp_soc_0_30[idx_temp] = (torch.nansum(temp_soc[0, 0:6]) if not torch.all(torch.isnan(temp_soc[0, 0:6])) else torch.tensor(np.nan, device=device))
					temp_soc_30_100[idx_temp] = (torch.nansum(temp_soc[0, 6:9]) if not torch.all(torch.isnan(temp_soc[0, 6:9])) else torch.tensor(np.nan, device=device))
					temp_soc_100_[idx_temp] = (torch.nansum(temp_soc[0, 9:]) if not torch.all(torch.isnan(temp_soc[0, 9:])) else torch.tensor(np.nan, device=device))

					temp_POM_all[idx_temp] = (torch.nansum(temp_POM) if not torch.all(torch.isnan(temp_POM)) else torch.tensor(np.nan, device=device))
					temp_POM_0_30[idx_temp] = (torch.nansum(temp_POM[0, 0:6]) if not torch.all(torch.isnan(temp_POM[0, 0:6])) else torch.tensor(np.nan, device=device))
					temp_POM_30_100[idx_temp] = (torch.nansum(temp_POM[0, 6:9]) if not torch.all(torch.isnan(temp_POM[0, 6:9])) else torch.tensor(np.nan, device=device))
					temp_POM_100_[idx_temp] = (torch.nansum(temp_POM[0, 9:]) if not torch.all(torch.isnan(temp_POM[0, 9:])) else torch.tensor(np.nan, device=device))

					temp_MAOM_all[idx_temp] = (torch.nansum(temp_MAOM) if not torch.all(torch.isnan(temp_MAOM)) else torch.tensor(np.nan, device=device))
					temp_MAOM_0_30[idx_temp] = (torch.nansum(temp_MAOM[0, 0:6]) if not torch.all(torch.isnan(temp_MAOM[0, 0:6])) else torch.tensor(np.nan, device=device))
					temp_MAOM_30_100[idx_temp] = (torch.nansum(temp_MAOM[0, 6:9]) if not torch.all(torch.isnan(temp_MAOM[0, 6:9])) else torch.tensor(np.nan, device=device))
					temp_MAOM_100_[idx_temp] = (torch.nansum(temp_MAOM[0, 9:]) if not torch.all(torch.isnan(temp_MAOM[0, 9:])) else torch.tensor(np.nan, device=device))
				# end for idx_temp in range(100)
				
				# Calculate the mean of the simulated SOC and store it
				temp_sensitivity_all[idx_test] = (torch.nanmean(temp_soc_all) if not torch.all(torch.isnan(temp_soc_all)) else torch.tensor(np.nan, device=device))
				temp_sensitivity_0_30[idx_test] = (torch.nanmean(temp_soc_0_30) if not torch.all(torch.isnan(temp_soc_0_30)) else torch.tensor(np.nan, device=device))
				temp_sensitivity_30_100[idx_test] = (torch.nanmean(temp_soc_30_100) if not torch.all(torch.isnan(temp_soc_30_100)) else torch.tensor(np.nan, device=device))
				temp_sensitivity_100_[idx_test] = (torch.nanmean(temp_soc_100_) if not torch.all(torch.isnan(temp_soc_100_)) else torch.tensor(np.nan, device=device))

				temp_sensitivity_all_POM[idx_test] = (torch.nanmean(temp_POM_all) if not torch.all(torch.isnan(temp_POM_all)) else torch.tensor(np.nan, device=device))
				temp_sensitivity_0_30_POM[idx_test] = (torch.nanmean(temp_POM_0_30) if not torch.all(torch.isnan(temp_POM_0_30)) else torch.tensor(np.nan, device=device))
				temp_sensitivity_30_100_POM[idx_test] = (torch.nanmean(temp_POM_30_100) if not torch.all(torch.isnan(temp_POM_30_100)) else torch.tensor(np.nan, device=device))
				temp_sensitivity_100_POM[idx_test] = (torch.nanmean(temp_POM_100_) if not torch.all(torch.isnan(temp_POM_100_)) else torch.tensor(np.nan, device=device))

				temp_sensitivity_all_MAOM[idx_test] = (torch.nanmean(temp_MAOM_all) if not torch.all(torch.isnan(temp_MAOM_all)) else torch.tensor(np.nan, device=device))
				temp_sensitivity_0_30_MAOM[idx_test] = (torch.nanmean(temp_MAOM_0_30) if not torch.all(torch.isnan(temp_MAOM_0_30)) else torch.tensor(np.nan, device=device))
				temp_sensitivity_30_100_MAOM[idx_test] = (torch.nanmean(temp_MAOM_30_100) if not torch.all(torch.isnan(temp_MAOM_30_100)) else torch.tensor(np.nan, device=device))
				temp_sensitivity_100_MAOM[idx_test] = (torch.nanmean(temp_MAOM_100_) if not torch.all(torch.isnan(temp_MAOM_100_)) else torch.tensor(np.nan, device=device))

				# if any values over 1e15, set to 1e15
				temp_sensitivity_all[idx_test] = torch.clamp(temp_sensitivity_all[idx_test], max=1e15)
				temp_sensitivity_0_30[idx_test] = torch.clamp(temp_sensitivity_0_30[idx_test], max=1e15)
				temp_sensitivity_30_100[idx_test] = torch.clamp(temp_sensitivity_30_100[idx_test], max=1e15)
				temp_sensitivity_100_[idx_test] = torch.clamp(temp_sensitivity_100_[idx_test], max=1e15)

				temp_sensitivity_all_POM[idx_test] = torch.clamp(temp_sensitivity_all_POM[idx_test], max=1e15)
				temp_sensitivity_0_30_POM[idx_test] = torch.clamp(temp_sensitivity_0_30_POM[idx_test], max=1e15)
				temp_sensitivity_30_100_POM[idx_test] = torch.clamp(temp_sensitivity_30_100_POM[idx_test], max=1e15)
				temp_sensitivity_100_POM[idx_test] = torch.clamp(temp_sensitivity_100_POM[idx_test], max=1e15)

				temp_sensitivity_all_MAOM[idx_test] = torch.clamp(temp_sensitivity_all_MAOM[idx_test], max=1e15)
				temp_sensitivity_0_30_MAOM[idx_test] = torch.clamp(temp_sensitivity_0_30_MAOM[idx_test], max=1e15)
				temp_sensitivity_30_100_MAOM[idx_test] = torch.clamp(temp_sensitivity_30_100_MAOM[idx_test], max=1e15)
				temp_sensitivity_100_MAOM[idx_test] = torch.clamp(temp_sensitivity_100_MAOM[idx_test], max=1e15)

				# end for idx_test in range(100)
			# Remove nan, inf, and zero
			temp_sensitivity_all = temp_sensitivity_all[~torch.isnan(temp_sensitivity_all) & ~torch.isinf(temp_sensitivity_all) & (temp_sensitivity_all >= 0)]
			temp_sensitivity_0_30 = temp_sensitivity_0_30[~torch.isnan(temp_sensitivity_0_30) & ~torch.isinf(temp_sensitivity_0_30) & (temp_sensitivity_0_30 >= 0)]
			temp_sensitivity_30_100 = temp_sensitivity_30_100[~torch.isnan(temp_sensitivity_30_100) & ~torch.isinf(temp_sensitivity_30_100) & (temp_sensitivity_30_100 >= 0)]
			temp_sensitivity_100_ = temp_sensitivity_100_[~torch.isnan(temp_sensitivity_100_) & ~torch.isinf(temp_sensitivity_100_) & (temp_sensitivity_100_ >= 0)]

			temp_sensitivity_all_POM = temp_sensitivity_all_POM[~torch.isnan(temp_sensitivity_all_POM) & ~torch.isinf(temp_sensitivity_all_POM) & (temp_sensitivity_all_POM >= 0)]
			temp_sensitivity_0_30_POM = temp_sensitivity_0_30_POM[~torch.isnan(temp_sensitivity_0_30_POM) & ~torch.isinf(temp_sensitivity_0_30_POM) & (temp_sensitivity_0_30_POM >= 0)]
			temp_sensitivity_30_100_POM = temp_sensitivity_30_100_POM[~torch.isnan(temp_sensitivity_30_100_POM) & ~torch.isinf(temp_sensitivity_30_100_POM)	 & (temp_sensitivity_30_100_POM >= 0)]
			temp_sensitivity_100_POM = temp_sensitivity_100_POM[~torch.isnan(temp_sensitivity_100_POM) & ~torch.isinf(temp_sensitivity_100_POM) & (temp_sensitivity_100_POM >= 0)]
			
			temp_sensitivity_all_MAOM = temp_sensitivity_all_MAOM[~torch.isnan(temp_sensitivity_all_MAOM) & ~torch.isinf(temp_sensitivity_all_MAOM) & (temp_sensitivity_all_MAOM >= 0)]
			temp_sensitivity_0_30_MAOM = temp_sensitivity_0_30_MAOM[~torch.isnan(temp_sensitivity_0_30_MAOM) & ~torch.isinf(temp_sensitivity_0_30_MAOM) & (temp_sensitivity_0_30_MAOM >= 0)]
			temp_sensitivity_30_100_MAOM = temp_sensitivity_30_100_MAOM[~torch.isnan(temp_sensitivity_30_100_MAOM) & ~torch.isinf(temp_sensitivity_30_100_MAOM) & (temp_sensitivity_30_100_MAOM >= 0)]
			temp_sensitivity_100_MAOM = temp_sensitivity_100_MAOM[~torch.isnan(temp_sensitivity_100_MAOM) & ~torch.isinf(temp_sensitivity_100_MAOM) & (temp_sensitivity_100_MAOM >= 0)]
			
			# Calculate the variance of the sensitivity
			sensitivity_soc_all[ipara, batch_idx] = torch.var(temp_sensitivity_all) / obs_soc_all_var
			sensitivity_soc_0_30[ipara, batch_idx] = torch.var(temp_sensitivity_0_30) / obs_soc_0_30_var
			sensitivity_soc_30_100[ipara, batch_idx] = torch.var(temp_sensitivity_30_100) / obs_soc_30_100_var
			sensitivity_soc_100_[ipara, batch_idx] = torch.var(temp_sensitivity_100_) / obs_soc_100_var

			sensitivity_soc_all_POM[ipara, batch_idx] = torch.var(temp_sensitivity_all_POM) / obs_POM_all_var
			sensitivity_soc_0_30_POM[ipara, batch_idx] = torch.var(temp_sensitivity_0_30_POM) / obs_POM_0_30_var
			sensitivity_soc_30_100_POM[ipara, batch_idx] = torch.var(temp_sensitivity_30_100_POM) / obs_POM_30_100_var
			sensitivity_soc_100_POM[ipara, batch_idx] = torch.var(temp_sensitivity_100_POM) / obs_POM_100_var

			sensitivity_soc_all_MAOM[ipara, batch_idx] = torch.var(temp_sensitivity_all_MAOM) / obs_MAOM_all_var
			sensitivity_soc_0_30_MAOM[ipara, batch_idx] = torch.var(temp_sensitivity_0_30_MAOM) / obs_MAOM_0_30_var
			sensitivity_soc_30_100_MAOM[ipara, batch_idx] = torch.var(temp_sensitivity_30_100_MAOM) / obs_MAOM_30_100_var
			sensitivity_soc_100_MAOM[ipara, batch_idx] = torch.var(temp_sensitivity_100_MAOM) / obs_MAOM_100_var



		print('Sensitivity test for batch: ', batch_idx, ' finished in ', time.time() - test_start_time, ' seconds')
		# end for ipara in range(len(para_name))
	
	# Save the sensitivity to excel file
	pd.DataFrame(sensitivity_soc_all.cpu().detach().numpy()).to_excel(data_dir_output + '/Sensitivity_soc_all.xlsx')
	pd.DataFrame(sensitivity_soc_0_30.cpu().detach().numpy()).to_excel(data_dir_output + '/Sensitivity_soc_0_30.xlsx')
	pd.DataFrame(sensitivity_soc_30_100.cpu().detach().numpy()).to_excel(data_dir_output + '/Sensitivity_soc_30_100.xlsx')
	pd.DataFrame(sensitivity_soc_100_.cpu().detach().numpy()).to_excel(data_dir_output + '/Sensitivity_soc_100_.xlsx')

	pd.DataFrame(sensitivity_soc_all_POM.cpu().detach().numpy()).to_excel(data_dir_output + '/Sensitivity_soc_all_POM.xlsx')
	pd.DataFrame(sensitivity_soc_0_30_POM.cpu().detach().numpy()).to_excel(data_dir_output + '/Sensitivity_soc_0_30_POM.xlsx')
	pd.DataFrame(sensitivity_soc_30_100_POM.cpu().detach().numpy()).to_excel(data_dir_output + '/Sensitivity_soc_30_100_POM.xlsx')
	pd.DataFrame(sensitivity_soc_100_POM.cpu().detach().numpy()).to_excel(data_dir_output + '/Sensitivity_soc_100_POM.xlsx')

	pd.DataFrame(sensitivity_soc_all_MAOM.cpu().detach().numpy()).to_excel(data_dir_output + '/Sensitivity_soc_all_MAOM.xlsx')
	pd.DataFrame(sensitivity_soc_0_30_MAOM.cpu().detach().numpy()).to_excel(data_dir_output + '/Sensitivity_soc_0_30_MAOM.xlsx')
	pd.DataFrame(sensitivity_soc_30_100_MAOM.cpu().detach().numpy()).to_excel(data_dir_output + '/Sensitivity_soc_30_100_MAOM.xlsx')
	pd.DataFrame(sensitivity_soc_100_MAOM.cpu().detach().numpy()).to_excel(data_dir_output + '/Sensitivity_soc_100_MAOM.xlsx')

	# Calculate the mean sensitivity for each parameter
	sensitivity_soc_all = torch.mean(sensitivity_soc_all, dim=1)
	sensitivity_soc_0_30 = torch.mean(sensitivity_soc_0_30, dim=1)
	sensitivity_soc_30_100 = torch.mean(sensitivity_soc_30_100, dim=1)
	sensitivity_soc_100_ = torch.mean(sensitivity_soc_100_, dim=1)

	sensitivity_soc_all_POM = torch.mean(sensitivity_soc_all_POM, dim=1)
	sensitivity_soc_0_30_POM = torch.mean(sensitivity_soc_0_30_POM, dim=1)
	sensitivity_soc_30_100_POM = torch.mean(sensitivity_soc_30_100_POM, dim=1)
	sensitivity_soc_100_POM = torch.mean(sensitivity_soc_100_POM, dim=1)

	sensitivity_soc_all_MAOM = torch.mean(sensitivity_soc_all_MAOM, dim=1)
	sensitivity_soc_0_30_MAOM = torch.mean(sensitivity_soc_0_30_MAOM, dim=1)
	sensitivity_soc_30_100_MAOM = torch.mean(sensitivity_soc_30_100_MAOM, dim=1)
	sensitivity_soc_100_MAOM = torch.mean(sensitivity_soc_100_MAOM, dim=1)

	# Create bar plot for sensitivity of each parameter
	for i_type in ['soc', 'POM', 'MAOM']:
		if i_type == 'soc':
			sensitivity_all = sensitivity_soc_all
			sensitivity_0_30 = sensitivity_soc_0_30
			sensitivity_30_100 = sensitivity_soc_30_100
			sensitivity_100_ = sensitivity_soc_100_
		elif i_type == 'POM':
			sensitivity_all = sensitivity_soc_all_POM
			sensitivity_0_30 = sensitivity_soc_0_30_POM
			sensitivity_30_100 = sensitivity_soc_30_100_POM
			sensitivity_100_ = sensitivity_soc_100_POM
		elif i_type == 'MAOM':
			sensitivity_all = sensitivity_soc_all_MAOM
			sensitivity_0_30 = sensitivity_soc_0_30_MAOM
			sensitivity_30_100 = sensitivity_soc_30_100_MAOM
			sensitivity_100_ = sensitivity_soc_100_MAOM

		# Create a horizonal bar plot for sensitivity of each parameter	
		fig, ax = plt.subplots()
		ax.barh(para_name, sensitivity_all.cpu().detach().numpy())
		ax.set_xlabel('Sensitivity')
		ax.set_ylabel('Parameter')
		ax.set_title('Sensitivity of each parameter to ' + i_type + ' at all layers')
		# ax.set_xlim(0, 1)
		plt.savefig(data_dir_output + '/Sensitivity_' + i_type + '_all.png')
		plt.close()
		print('Sensitivity of each parameter to ' + i_type + ' at all layers saved')

		fig, ax = plt.subplots()
		ax.barh(para_name, sensitivity_0_30.cpu().detach().numpy())
		ax.set_xlabel('Sensitivity')
		ax.set_ylabel('Parameter')
		ax.set_title('Sensitivity of each parameter to ' + i_type + ' at 0-30cm')
		# ax.set_xlim(0, 1)
		plt.savefig(data_dir_output + '/Sensitivity_' + i_type + '_0_30.png')
		plt.close()
		print('Sensitivity of each parameter to ' + i_type + ' at 0-30cm saved')

		fig, ax = plt.subplots()
		ax.barh(para_name, sensitivity_30_100.cpu().detach().numpy())
		ax.set_xlabel('Sensitivity')
		ax.set_ylabel('Parameter')
		ax.set_title('Sensitivity of each parameter to ' + i_type + ' at 30-100cm')
		# ax.set_xlim(0, 1)
		plt.savefig(data_dir_output + '/Sensitivity_' + i_type + '_30_100.png')
		plt.close()
		print('Sensitivity of each parameter to ' + i_type + ' at 30-100cm saved')

		fig, ax = plt.subplots()
		ax.barh(para_name, sensitivity_100_.cpu().detach().numpy())
		ax.set_xlabel('Sensitivity')
		ax.set_ylabel('Parameter')
		ax.set_title('Sensitivity of each parameter to ' + i_type + ' at 100-')
		# ax.set_xlim(0, 1)
		plt.savefig(data_dir_output + '/Sensitivity_' + i_type + '_100.png')
		plt.close()
		print('Sensitivity of each parameter to ' + i_type + ' at 100- saved')
	



	print('Sensitivity test finished within: ', time.time() - start_time, ' seconds')
# end worker

# Start the worker
if __name__ == '__main__':
	worker()