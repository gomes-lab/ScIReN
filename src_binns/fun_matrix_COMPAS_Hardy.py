import time
import numpy as np
import torch
import traceback
import math


# Simulate the soil carbon profile using the CLM5 model at the depth of the observation layers
def fun_model_simu(tensor_para, tensor_frocing_steady_state, tensor_obs_layer_depth):
	start_time = time.time()
	device = tensor_para.device
	# convert tensor to numpy
	para = tensor_para
	# para = (tensor_para - (-1)) /(1 - (-1)) # conversion from Hardttanh [-1, 1] to [0, 1]
	frocing_steady_state = tensor_frocing_steady_state 
	obs_layer_depth = tensor_obs_layer_depth

	# depth of the node                                                   
	zsoi = torch.tensor([1.000000000000000E-002, 4.000000000000000E-002, 9.000000000000000E-002, \
		0.160000000000000, 0.260000000000000, 0.400000000000000, \
		0.580000000000000, 0.800000000000000, 1.06000000000000, \
		1.36000000000000, 1.70000000000000, 2.08000000000000, \
		2.50000000000000, 2.99000000000000, 3.58000000000000, \
		4.27000000000000, 5.06000000000000, 5.95000000000000, \
		6.94000000000000, 8.03000000000000, 9.79500000000000, \
		13.3277669529664, 19.4831291701244, 28.8707244343160, \
		41.9984368640029]).to(device)
	

	n_soil_layer = 20

	# final ouputs of simulation
	profile_num = para.shape[0]
	simu_ouput = (torch.ones((profile_num, 3, 200))*np.nan).to(device)
	# calculate soc solution for each profile
	for iprofile in range(0, profile_num):
		profile_para = para[iprofile, :]
		profile_force_steady_state = frocing_steady_state[iprofile, :, :, :]
		profile_obs_layer_depth = obs_layer_depth[iprofile, :]
		valid_layer_loc = torch.where(torch.isnan(profile_obs_layer_depth) == False)[0]

		if torch.isnan(torch.sum(profile_para)) == False and \
			torch.isnan(torch.sum(profile_force_steady_state[0:12, 0, 1:8])) == False and \
			torch.isnan(torch.sum(profile_force_steady_state[0:20, 0:12, 8:13])) == False:
			
			# print(profile_para)
			# model simulation
			profile_simu_soc, profile_simu_POM, profile_simu_MAOM, profile_simu_DOC, profile_simu_MIC = matrix_fun_COMPAS(profile_para, profile_force_steady_state)
			
			for ilayer in range(0, len(valid_layer_loc)):
				layer_depth = profile_obs_layer_depth[valid_layer_loc[ilayer]]
				depth_diff = zsoi[0:n_soil_layer] - layer_depth
				if len(torch.where(depth_diff == 0)[0]) == 0:
					if depth_diff[0] > 0:
						node_depth_upper_loc = 0
						node_depth_lower_loc = 0
					elif depth_diff[-1] < 0:
						node_depth_upper_loc = n_soil_layer - 1
						node_depth_lower_loc = n_soil_layer - 1
					else:
						node_depth_upper_loc = torch.where(depth_diff[:-1]*depth_diff[1:]<0)[0]
						node_depth_lower_loc = node_depth_upper_loc + 1
					# end if depth_diff[0] > 0:
				else:
					node_depth_upper_loc = torch.where(depth_diff == 0)
					node_depth_lower_loc = node_depth_upper_loc
				#end if len(torch.where(depth_diff == 0)[0]) == 0:
				if node_depth_lower_loc == node_depth_upper_loc:
					simu_ouput[iprofile, 0, valid_layer_loc[ilayer]] = profile_simu_soc[node_depth_lower_loc]
					simu_ouput[iprofile, 1, valid_layer_loc[ilayer]] = profile_simu_POM[node_depth_lower_loc]
					simu_ouput[iprofile, 2, valid_layer_loc[ilayer]] = profile_simu_MAOM[node_depth_lower_loc]
				else:
					simu_ouput[iprofile, 0, valid_layer_loc[ilayer]] = \
					profile_simu_soc[node_depth_lower_loc] \
					+ (profile_simu_soc[node_depth_upper_loc] - profile_simu_soc[node_depth_lower_loc]) \
					/(zsoi[node_depth_upper_loc] - zsoi[node_depth_lower_loc]) \
					*(layer_depth - zsoi[node_depth_lower_loc])

					simu_ouput[iprofile, 1, valid_layer_loc[ilayer]] = \
					profile_simu_POM[node_depth_lower_loc] \
					+ (profile_simu_POM[node_depth_upper_loc] - profile_simu_POM[node_depth_lower_loc]) \
					/(zsoi[node_depth_upper_loc] - zsoi[node_depth_lower_loc]) \
					*(layer_depth - zsoi[node_depth_lower_loc])

					simu_ouput[iprofile, 2, valid_layer_loc[ilayer]] = \
					profile_simu_MAOM[node_depth_lower_loc] \
					+ (profile_simu_MAOM[node_depth_upper_loc] - profile_simu_MAOM[node_depth_lower_loc]) \
					/(zsoi[node_depth_upper_loc] - zsoi[node_depth_lower_loc]) \
					*(layer_depth - zsoi[node_depth_lower_loc])

			# end for
		# end if 
	#end for iprofile
	# print("fun_model_simu", time.time()-start_time)
	return simu_ouput
	
# end def fun_model_simu

# Model prediction of the soil carbon profile using the CLM5 model
def fun_model_prediction(tensor_para, tensor_frocing_steady_state):
	device = tensor_para.device
	# convert tensor to numpy
	para = tensor_para
	# para = (tensor_para - (-1)) /(1 - (-1)) # conversion from Hardttanh [-1, 1] to [0, 1]
	frocing_steady_state = tensor_frocing_steady_state 

	# depth of the node                                                   
	zsoi = torch.tensor([1.000000000000000E-002, 4.000000000000000E-002, 9.000000000000000E-002, \
		0.160000000000000, 0.260000000000000, 0.400000000000000, \
		0.580000000000000, 0.800000000000000, 1.06000000000000, \
		1.36000000000000, 1.70000000000000, 2.08000000000000, \
		2.50000000000000, 2.99000000000000, 3.58000000000000, \
		4.27000000000000, 5.06000000000000, 5.95000000000000, \
		6.94000000000000, 8.03000000000000, 9.79500000000000, \
		13.3277669529664, 19.4831291701244, 28.8707244343160, \
		41.9984368640029]).to(device)
	

	n_soil_layer = 20

	# final ouputs of simulation
	profile_num = para.shape[0]
	simu_ouput = (torch.ones((profile_num, 5, 200))*np.nan).to(device)

	# calculate soc solution for each profile
	for iprofile in range(0, profile_num):
		profile_para = para[iprofile, :]
		profile_force_steady_state = frocing_steady_state[iprofile, :, :, :]

		if torch.isnan(torch.sum(profile_para)) == False and \
			torch.isnan(torch.sum(profile_force_steady_state[0:12, 0, 1:8])) == False and \
			torch.isnan(torch.sum(profile_force_steady_state[0:20, 0:12, 8:13])) == False:
			
			# print(profile_para)
			# model simulation
			profile_simu_soc, profile_simu_POM, profile_simu_MAOM, profile_simu_DOC, profile_simu_MIC = matrix_fun_COMPAS(profile_para, profile_force_steady_state)
			
			# save simulation results
			simu_ouput[iprofile, 0, 0:20] = profile_simu_soc
			simu_ouput[iprofile, 1, 0:20] = profile_simu_POM
			simu_ouput[iprofile, 2, 0:20] = profile_simu_MAOM
			simu_ouput[iprofile, 3, 0:20] = profile_simu_DOC
			simu_ouput[iprofile, 4, 0:20] = profile_simu_MIC

	#end for iprofile
	return simu_ouput

#######################################################
# forward simulation for clm5
#######################################################
def matrix_fun_COMPAS(tensor_para, tensor_forcing_steady_state):
	device = tensor_para.device

	para = tensor_para
	forcing_steady_state = tensor_forcing_steady_state

	global month_num, normalize_q10_to_century_tfunc, kelvin_to_celsius, m_to_cm, n_soil_layer, npool, npool_vr, days_per_year, timestep_num, use_beta

	month_num = 12
	normalize_q10_to_century_tfunc = False
	kelvin_to_celsius = 273.15
	m_to_cm = 100
	n_soil_layer = 20
	npool = 7
	npool_vr = 140
	days_per_year = 365
	timestep_num = month_num
	use_beta = True
	
	global max_altdepth_cryoturbation, max_depth_cryoturb
	max_altdepth_cryoturbation = 2
	max_depth_cryoturb = 3

	global dz, zisoi, zisoi_0, zsoi, dz_node
	# width between two interfaces
	dz = torch.tensor([2.000000000000000E-002, 4.000000000000000E-002, 6.000000000000000E-002, \
		8.000000000000000E-002, 0.120000000000000, 0.160000000000000, \
		0.200000000000000, 0.240000000000000, 0.280000000000000, \
		0.320000000000000, 0.360000000000000, 0.400000000000000, \
		0.440000000000000, 0.540000000000000, 0.640000000000000, \
		0.740000000000000, 0.840000000000000, 0.940000000000000, \
		1.04000000000000, 1.14000000000000, 2.39000000000000, \
		4.67553390593274, 7.63519052838329, 11.1400000000000, \
		15.1154248593737]).to(device)
	
	# depth of the interface
	zisoi = torch.tensor([2.000000000000000E-002, 6.000000000000000E-002, \
		0.120000000000000, 0.200000000000000, 0.320000000000000, \
		0.480000000000000, 0.680000000000000, 0.920000000000000, \
		1.20000000000000, 1.52000000000000, 1.88000000000000, \
		2.28000000000000, 2.72000000000000, 3.26000000000000, \
		3.90000000000000, 4.64000000000000, 5.48000000000000, \
		6.42000000000000, 7.46000000000000, 8.60000000000000, \
		10.9900000000000, 15.6655339059327, 23.3007244343160, \
		34.4407244343160, 49.5561492936897]).to(device)

	zisoi_0 = 0;

	# depth of the node                                                   
	zsoi = torch.tensor([1.000000000000000E-002, 4.000000000000000E-002, 9.000000000000000E-002, \
		0.160000000000000, 0.260000000000000, 0.400000000000000, \
		0.580000000000000, 0.800000000000000, 1.06000000000000, \
		1.36000000000000, 1.70000000000000, 2.08000000000000, \
		2.50000000000000, 2.99000000000000, 3.58000000000000, \
		4.27000000000000, 5.06000000000000, 5.95000000000000, \
		6.94000000000000, 8.03000000000000, 9.79500000000000, \
		13.3277669529664, 19.4831291701244, 28.8707244343160, \
		41.9984368640029]).to(device)

	# depth between two node
	dz_node = zsoi - torch.cat((torch.tensor([0.0]).to(device), zsoi[:-1]), axis = 0)

	# construct a diagonal matrix that contains dz for each layer (20 layers) and 7 pools
	dz_matrix = torch.diag(-1*torch.ones(npool_vr)).to(device)
	# fill the diagonal matrix with dz for each pool (7 pools)
	dz_matrix.diagonal()[0:20] = dz[0:20]
	dz_matrix.diagonal()[20:40] = dz[0:20]
	dz_matrix.diagonal()[40:60] = dz[0:20]
	dz_matrix.diagonal()[60:80] = dz[0:20]
	dz_matrix.diagonal()[80:100] = dz[0:20]
	dz_matrix.diagonal()[100:120] = dz[0:20]
	dz_matrix.diagonal()[120:140] = dz[0:20]
	dz_matrix_diagonal = dz_matrix.diagonal().view(npool_vr, 1)

	
	#---------------------------------------------------
	# steady state forcing
	#---------------------------------------------------
	input_vector_cwd = forcing_steady_state[0:12, 0, 1]
	input_vector_litter1 = forcing_steady_state[0:12, 0, 2]
	input_vector_litter2 = forcing_steady_state[0:12, 0, 3]
	input_vector_litter3 = forcing_steady_state[0:12, 0, 4]
	altmax_lastyear_profile = forcing_steady_state[0:12, 0, 5]
	altmax_current_profile = forcing_steady_state[0:12, 0, 6]
	nbedrock = forcing_steady_state[0:12, 0, 7].type(torch.int)

	xio = forcing_steady_state[0:20, 0:12, 8]
	xin = forcing_steady_state[0:20, 0:12, 9]
	sand_vector = forcing_steady_state[0:20, 0:12, 10]
	soil_temp_profile = forcing_steady_state[0:20, 0:12, 11]
	soil_water_profile = forcing_steady_state[0:20, 0:12, 12]



	def scale(val, lo, hi):
		return val * (hi - lo) + lo

	#---------------------------------------------------
	# define parameters to be optimised
	#---------------------------------------------------
	bio = scale(para[0], 3e-5, 5e-4)
	cryo = scale(para[1], 3e-5, 16e-4)
	q10 = scale(para[2], 1.2, 3)
	fq10 = q10
	efolding = scale(para[3], 0.1, 1)
	tau4cwd = scale(para[4], 1, 6)
	tau4l1 = scale(para[5], 0, 0.11)
	tau4l2 = scale(para[6], 0.1, 0.3)
	tau4doc = scale(para[7], 0.0001, 1)
	tau4mic = scale(para[8], 0.0001, 1)
	tau4poc = scale(para[9], 1, 10)
	tau4maom = scale(para[10], 1, 200)

	# f42 = scale(para[11], 0.1, 0.5)
	# x52 = scale(para[12], 0.05, 0.5)
	x52 = scale(para[11], 0.0001, 0.9)
	x53 = scale(para[12], 0.0001, 0.9)
	# f43 = scale(para[13], 0.0001, 0.4)
	f45 = scale(para[13], 0.3, 0.8)
	f65 = scale(para[14], 0.0001, 0.2)
	x74 = scale(para[15], 0.0001, 0.2)
	# f46 = scale(para[16], 0.1, 0.8)
	f46 = 1

	e52 = scale(para[16], 0.0001, 0.6)
	e53 = scale(para[17], 0.0001, 0.4)
	e54 = scale(para[18], 0.1, 0.99)
	w_scaling = scale(para[19], 0.0001, 5)
	beta = scale(para[20], 0.5, 0.9999)


	adv = 0

	#####################################################
	# steady state solutions
	#####################################################
	# Environmental Scalars
	xit = (torch.ones(n_soil_layer, timestep_num)*np.nan).to(device)
	xiw = (torch.ones(n_soil_layer, timestep_num)*np.nan).to(device)
	xio = xio
	xin = xin

	for imonth in range(month_num):
		for ilayer in range(n_soil_layer):
			if soil_temp_profile[ilayer, imonth] >= (0 + kelvin_to_celsius):
				xit[ilayer, imonth] = q10 ** ((soil_temp_profile[ilayer, imonth] - (kelvin_to_celsius + 25)) / 10)
			else:
				xit[ilayer, imonth] = q10 ** ((273.15 - 298.15) / 10) * (fq10 ** ((soil_temp_profile[ilayer, imonth] - (0 + kelvin_to_celsius)) / 10))


	catanf_30 = catanf(torch.tensor(30.0).to(device))
	normalization_tref = torch.tensor(15).to(device)
	if normalize_q10_to_century_tfunc == True:
		# scale all decomposition rates by a constant to compensate for offset between original CENTURY temp func and Q10
		normalization_factor = (catanf(normalization_tref)/catanf_30) / (q10**((normalization_tref-25)/10))
		xit[:, itimestep] = xit[:, itimestep]*normalization_factor

	xiw = soil_water_profile * w_scaling
	xiw = torch.clamp(xiw, max=1.0)


	#---------------------------------------------------
	# steady state tridiagnal matrix, A matrix, K matrix
	#---------------------------------------------------
	sand_vector_mean = torch.mean(sand_vector, axis = 1)

	a_ma = a_matrix(x52, x53, x74, f45, f65, f46, e52, e53, e54, sand_vector) # f42, f43, 

	kk_ma_middle = (torch.zeros([npool_vr, npool_vr, timestep_num])*np.nan).to(device) 
	tri_ma_middle = (torch.zeros([npool_vr, npool_vr, timestep_num])*np.nan).to(device) 

	
	for itimestep in range(timestep_num):
		# decomposition matrix
		timesteply_xit = xit[:, itimestep]
		timesteply_xiw = xiw[:, itimestep]
		timesteply_xio = xio[:, itimestep]
		timesteply_xin = xin[:, itimestep]

		# K matrix
		kk_ma_middle[:, :, itimestep] = kk_matrix(timesteply_xit, timesteply_xiw, timesteply_xio, timesteply_xin, efolding, tau4cwd, tau4l1, tau4l2, tau4doc, tau4mic, tau4poc, tau4maom)
		# tri matrix
		timesteply_nbedrock = nbedrock[itimestep]
		timesteply_altmax_current_profile = altmax_current_profile[itimestep]
		timesteply_altmax_lastyear_profile = altmax_lastyear_profile[itimestep]
		tri_ma_middle[:, :, itimestep] = tri_matrix(timesteply_nbedrock, timesteply_altmax_current_profile, timesteply_altmax_lastyear_profile, bio, adv, cryo)
		# tri_ma_middle[:, :, itimestep] = tri_matrix(timesteply_nbedrock, timesteply_altmax_current_profile, timesteply_altmax_lastyear_profile, bio, adv, cryo)
	# end for itimestep
	tri_ma = torch.mean(tri_ma_middle, axis = 2)
	kk_ma = torch.mean(kk_ma_middle, axis = 2)

	#---------------------------------------------------
	# steady state vertical profile, input allocation
	#---------------------------------------------------
	# in the original beta model in Jackson et al 1996, the unit for the depth of the soil is cm (dmax*100)
	vertical_prof = (torch.ones(n_soil_layer)*np.nan).to(device) 
	if torch.mean(altmax_lastyear_profile) > 0:
		for j in range(n_soil_layer): #1:n_soil_layer
			if j == 0: # first layer
				vertical_prof[j] = (beta**((zisoi_0)*m_to_cm) - beta**(zisoi[j]*m_to_cm))/dz[j]
			else:
				vertical_prof[j] = (beta**((zisoi[j - 1])*m_to_cm) - beta**(zisoi[j]*m_to_cm))/dz[j]
			# end j == 0:
		# end for j in range(n_soil_layer):
	else:
		vertical_prof[0] = 1/dz[0]
		vertical_prof[1:] = 0
	# end if np.mean(altmax_lastyear_profile_steady_state) > 0:
	vertical_input = dz[0:n_soil_layer]*vertical_prof/sum(vertical_prof*dz[0:n_soil_layer])

	#---------------------------------------------------
	# steady state analytical solution of soc
	#---------------------------------------------------
	matrix_in = (torch.ones([npool_vr, 1])*np.nan).to(device)
	# total input amount
	input_tot_cwd = torch.nansum(input_vector_cwd)/days_per_year # (gc/m2/day)
	input_tot_litter1 = torch.nansum(input_vector_litter1)/days_per_year # (gc/m2/day)
	input_tot_litter2 = torch.nansum(input_vector_litter2)/days_per_year # (gc/m2/day)
	input_tot_litter3 = torch.nansum(input_vector_litter3)/days_per_year # (gc/m2/day)


	# redistribution by beta
	matrix_in[0:20, 0] = input_tot_cwd*vertical_input/dz[0:n_soil_layer] # litter input gc/m3/day
	matrix_in[20:40, 0] = input_tot_litter1*vertical_input/dz[0:n_soil_layer]
	matrix_in[40:60, 0] = (input_tot_litter2+input_tot_litter3)*vertical_input/dz[0:n_soil_layer]
	# matrix_in[60:80, 0] = input_tot_litter3*vertical_input/dz[0:n_soil_layer]
	matrix_in[60:140, 0] = 0

	# analytical solution of soc pools
	try:
		# torch 1.7
		# cpool_steady_state = torch.solve((-matrix_in), (torch.matmul(a_ma, kk_ma)-tri_ma)).solution
		# torch 1.11
		# cpool_steady_state = torch.linalg.solve((torch.matmul(a_ma, kk_ma)- tri_ma), (-matrix_in))
		cpool_steady_state = torch.linalg.solve((torch.matmul(a_ma, kk_ma)- tri_ma), (-matrix_in))
		# cpool_steady_state = torch.div(cpool_steady_state, dz_matrix_diagonal)
		# print("Shape of cpool_steady_state after division: ", cpool_steady_state.shape)
	except Exception:
		traceback.print_exc()
		print("Predicted Parameters: ", para)
		# check if the matrix is singular and print the matrix
		# check a_ma
		if torch.isnan(torch.sum(a_ma)):
			print("a_ma contains nan")
		if torch.det(a_ma) == 0:
			print("a_ma is singular")
		# check kk_ma
		if torch.isnan(torch.sum(kk_ma)):
			print("kk_ma contains nan")
		if torch.det(kk_ma) == 0:
			print("kk_ma is singular")
			print(torch.diagonal(kk_ma))
		# check tri_ma
		if torch.isnan(torch.sum(tri_ma)):
			print("tri_ma contains nan")
		if torch.det(tri_ma) == 0:
			print("tri_ma is singular")
			print(torch.diagonal(tri_ma, offset=0))
			print(torch.diagonal(tri_ma, offset=1))
			print(torch.diagonal(tri_ma, offset=-1))
		# check matrix_in
		if torch.isnan(torch.sum(matrix_in)):
			print("matrix_in contains nan")
		if torch.det(matrix_in) == 0:
			print("matrix_in is singular")
			print(matrix_in)
		if torch.det(torch.matmul(a_ma, kk_ma)-tri_ma) == 0:
			print("a_ma*kk_ma - tri_ma is singular")
		

		# cpool_steady_state = torch.linalg.lstsq((torch.matmul(a_ma, kk_ma)-tri_ma), (-matrix_in)).solution 
		cpool_steady_state = (torch.ones([140, 1])*(-1.0)).to(device)*torch.sum(para)/torch.sum(para)

	soc_layer = torch.cat((cpool_steady_state[60:80, :], cpool_steady_state[80:100, :], cpool_steady_state[100:120, :], cpool_steady_state[120:140, :]), dim = 1)
	soc_layer = torch.sum(soc_layer, axis = 1) # unit gC/m3

	POM_layer = cpool_steady_state[100:120, :]
	POM_layer = torch.sum(POM_layer, axis = 1) # unit gC/m3
	MAOM_layer = cpool_steady_state[120:140, :]
	MAOM_layer = torch.sum(MAOM_layer, axis = 1) # unit gC/m3

	DOC_layer = cpool_steady_state[60:80, :]
	DOC_layer = torch.sum(DOC_layer, axis = 1) # unit gC/m3
	MIC_layer = cpool_steady_state[80:100, :]
	MIC_layer = torch.sum(MIC_layer, axis = 1) # unit gC/m3
	
	
	# if soc_layer[-1] > soc_layer[0]:
	# 	soc_layer =  (torch.ones(20)*(-9999.*3))
	# # end if soc_layer[-1] > soc_layer[0]:
	
	# outcome = soc_layer
	return soc_layer, POM_layer, MAOM_layer, DOC_layer, MIC_layer
	# return cpool_steady_state, a_ma, kk_ma, tri_ma, matrix_in
	
#end def fun_forward_simu_clm5


##################################################
# sub-function in matrix equation
##################################################

def a_matrix(x52, x53, x74, f45, f65, f46, E52, E53, E54, sand_vector): # f42, f43
    n_soil_layer = 20
    npool = 7
    npool_vr = 140
    device = sand_vector.device

    a_ma_vr = torch.diag(-1 * torch.ones(npool_vr, device=device))

    fl2cwd = 1.0
    # f62 = 1 - f42 - x52
    f42 = 1 - x52
    f52 = x52 * E52
    f63 = 1 - x53
    f53 = x53 * E53
    f54 = (1-x74) * E54
    f75 = 1 - f45 - f65
    f76 = 1 - f46 
    f47 = 1.0
    f74 = x74

    transfer_fraction = [fl2cwd, f42, f52, f53, f63, f54, f45, f65, f75, f46, f76, f47, f74] # f62: transfer_fraction[3], f43: transfer_fraction[5]

    for j in range(n_soil_layer):
        a_ma_vr[(3 - 1) * n_soil_layer + j, (1 - 1) * n_soil_layer + j] = transfer_fraction[0]
        a_ma_vr[(4 - 1) * n_soil_layer + j, (2 - 1) * n_soil_layer + j] = transfer_fraction[1]
        a_ma_vr[(5 - 1) * n_soil_layer + j, (2 - 1) * n_soil_layer + j] = transfer_fraction[2]
        a_ma_vr[(5 - 1) * n_soil_layer + j, (3 - 1) * n_soil_layer + j] = transfer_fraction[3]
        a_ma_vr[(6 - 1) * n_soil_layer + j, (3 - 1) * n_soil_layer + j] = transfer_fraction[4]
        a_ma_vr[(5 - 1) * n_soil_layer + j, (4 - 1) * n_soil_layer + j] = transfer_fraction[5]
        a_ma_vr[(4 - 1) * n_soil_layer + j, (5 - 1) * n_soil_layer + j] = transfer_fraction[6]
        a_ma_vr[(6 - 1) * n_soil_layer + j, (5 - 1) * n_soil_layer + j] = transfer_fraction[7]
        a_ma_vr[(7 - 1) * n_soil_layer + j, (5 - 1) * n_soil_layer + j] = transfer_fraction[8]
        a_ma_vr[(4 - 1) * n_soil_layer + j, (6 - 1) * n_soil_layer + j] = transfer_fraction[9]
        a_ma_vr[(7 - 1) * n_soil_layer + j, (6 - 1) * n_soil_layer + j] = transfer_fraction[10]
        a_ma_vr[(4 - 1) * n_soil_layer + j, (7 - 1) * n_soil_layer + j] = transfer_fraction[11]
        a_ma_vr[(7 - 1) * n_soil_layer + j, (4 - 1) * n_soil_layer + j] = transfer_fraction[12]

    return a_ma_vr
	
# end def a_matrix

def kk_matrix(xit, xiw, xio, xin, efolding, tau4cwd, tau4l1, tau4l2, tau4doc, tau4mic, tau4poc, tau4maom):
	device = xit.device

	n_soil_layer = 20
	days_per_year = 365

	# env scalars
	n_scalar = xin[:n_soil_layer] # nitrogen 
	t_scalar = xit[:n_soil_layer] # temperature
	w_scalar = xiw[:n_soil_layer] # water
	o_scalar = xio[:n_soil_layer] # oxygen

	kl1 = 1 / (days_per_year * tau4l1)
	kl2 = 1 / (days_per_year * tau4l2)
	kdoc = 1 / (days_per_year * tau4doc)
	kmic = 1 / (days_per_year * tau4mic)
	kpoc = 1 / (days_per_year * tau4poc)
	kmaom = 1 / (days_per_year * tau4maom)
	kcwd = 1 / (days_per_year * tau4cwd)

	decomp_depth_efolding = efolding
	depth_scalar = torch.exp(-zsoi/decomp_depth_efolding)[:n_soil_layer] 
	xi_tw = t_scalar*w_scalar*o_scalar

	diagonal_vector = torch.concatenate([kcwd * xi_tw * depth_scalar,
									     kl1 * xi_tw * depth_scalar * n_scalar,
										 kl2 * xi_tw * depth_scalar * n_scalar,
										 kdoc * xi_tw * depth_scalar,
										 kmic * xi_tw * depth_scalar,
										 kpoc * xi_tw * depth_scalar,
										 kmaom * xi_tw * depth_scalar], dim=0).to(device)
	assert diagonal_vector.shape == (140, )
	return torch.diag(diagonal_vector)



# end def kk_matrix


def tri_matrix(nbedrock, altmax, altmax_lastyear, som_diffus, som_adv_flux, cryoturb_diffusion_k):
    	
	device = som_diffus.device

	nlevdecomp = n_soil_layer
	epsilon = 1e-30

	# change the unit from m2/yr to m2/day
	som_diffus_day = som_diffus / days_per_year
	som_adv_flux_day = som_adv_flux / days_per_year # float does not require grad
	cryoturb_diffusion_k_day = cryoturb_diffusion_k / days_per_year

	# print("som_diffus_day.requires_grad: ", som_diffus_day.requires_grad)

	# print("cryoturb_diffusion_k_day.requires_grad: ", cryoturb_diffusion_k_day.requires_grad)

	tri_ma = (torch.zeros([npool_vr, npool_vr])).to(device)

	som_adv_coef = torch.zeros(nlevdecomp+1).to(device) # SOM advective flux (m/day)
	som_diffus_coef = torch.zeros(nlevdecomp+1).to(device) # SOM diffusivity due to bio/cryo-turbation (m2/day)
	diffus = torch.zeros(nlevdecomp+1).to(device) # diffusivity (m2/day)  (includes spinup correction, if any)
	adv_flux = torch.zeros(nlevdecomp+1).to(device) # advective flux (m/day)  (includes spinup correction, if any)

	a_tri_e = torch.zeros(nlevdecomp).to(device) # "a" vector for tridiagonal matrix
	b_tri_e = torch.zeros(nlevdecomp).to(device) # "b" vector for tridiagonal matrix
	c_tri_e = torch.zeros(nlevdecomp).to(device) # "c" vector for tridiagonal matrix
	r_tri_e = torch.zeros(nlevdecomp).to(device) #"r" vector for tridiagonal solution

	a_tri_dz = torch.zeros(nlevdecomp).to(device) # "a" vector for tridiagonal matrix with considering the depth
	b_tri_dz = torch.zeros(nlevdecomp).to(device) # "b" vector for tridiagonal matrix with considering the depth
	c_tri_dz = torch.zeros(nlevdecomp).to(device) # "c" vector for tridiagonal matrix with considering the depth

	# d_p1_zp1 = torch.zeros(nlevdecomp+1).to(device) # diffusivity/delta_z for next j
	# # (set to zero for no diffusion)
	# d_m1_zm1 = torch.zeros(nlevdecomp+1).to(device) # diffusivity/delta_z for previous j
	# # (set to zero for no diffusion)
	f_p1 = torch.zeros(nlevdecomp+1).to(device) # water flux for next j
	f_m1 = torch.zeros(nlevdecomp+1).to(device) # water flux for previous j
	pe_p1 = torch.zeros(nlevdecomp+1).to(device) # Peclet # for next j
	pe_m1 = torch.zeros(nlevdecomp+1).to(device) # Peclet # for previous j
	
	w_p1 = torch.zeros(nlevdecomp+1).to(device)
	w_m1 = torch.zeros(nlevdecomp+1).to(device)

	#------ first get diffusivity / advection terms -------
	# Convert conditions to tensor operations
	active_layer_depth = torch.tensor(max(altmax.item(), altmax_lastyear.item())).to(device)
	# is_active_layer = zisoi[:nbedrock+1] < active_layer_depth
	# is_below_active_layer_and_cryoturb = (zisoi[:nbedrock+1] >= active_layer_depth) & (zisoi[:nbedrock+1] <= torch.min(torch.tensor(max_depth_cryoturb), zisoi[nbedrock+1]))
	is_active_layer = zisoi[:nlevdecomp+1] < active_layer_depth
	is_below_active_layer_and_cryoturb = (zisoi[:nlevdecomp+1] >= active_layer_depth) & (zisoi[:nlevdecomp+1] <= torch.min(torch.tensor(max_depth_cryoturb), zisoi[nlevdecomp+1]))
	is_bedrock_layer = torch.arange(nlevdecomp+1).to(device) > nbedrock

	# Initialize coefficients with zeros
	som_diffus_coef.fill_(0.)
	som_adv_coef.fill_(0.)

	if active_layer_depth <= max_altdepth_cryoturbation and active_layer_depth > 0.:
		som_diffus_coef[is_active_layer] = cryoturb_diffusion_k_day
		linear_decrease_factor = (1. - (zisoi[:nlevdecomp+1][is_below_active_layer_and_cryoturb] - active_layer_depth) / (torch.min(torch.tensor(max_depth_cryoturb), zisoi[nlevdecomp+1]) - active_layer_depth))
		som_diffus_coef[is_below_active_layer_and_cryoturb] = torch.maximum(cryoturb_diffusion_k_day * linear_decrease_factor, torch.tensor(0.))
	elif active_layer_depth > 0.:
		# Constant advection and diffusion up to bedrock
		som_adv_coef[:nbedrock+1] = som_adv_flux_day
		som_diffus_coef[:nbedrock+1] = som_diffus_day
	# No else clause needed for completely frozen soils since the initialization already sets coefficients to 0

	# Apply mask for bedrock layers (no advection or diffusion)
	som_adv_coef[is_bedrock_layer] = 0.
	som_diffus_coef[is_bedrock_layer] = 0.


	# Initial setup - vectorized
	adv_flux = torch.where(torch.abs(som_adv_coef) < epsilon, torch.full_like(som_adv_coef, epsilon), som_adv_coef)
	diffus = torch.where(torch.abs(som_diffus_coef) < epsilon, torch.full_like(som_diffus_coef, epsilon), som_diffus_coef)
	# print("diffus", diffus)
	# print("adv_flux requires grad", adv_flux.requires_grad)
	# print("diffus requires grad", diffus.requires_grad)

	# Initialize tensors
	f_m1 = adv_flux.clone()
	f_p1 = torch.cat((adv_flux[1:], torch.tensor([0.]).to(device)))  # Shift adv_flux down and pad with 0
	w_m1 = torch.zeros_like(adv_flux)
	w_p1 = torch.zeros_like(adv_flux)
	d_m1_zm1 = torch.zeros_like(adv_flux)
	d_p1_zp1 = torch.zeros_like(adv_flux)

	# Calculations that apply for all layers except the special cases at the top and bottom
	# w_m1[1:] = (zisoi[:nlevdecomp+1][:-1] - zsoi[:nlevdecomp+1][:-1]) / dz_node[:nlevdecomp+1][1:]  # Skip the first layer for w_m1
	# Try to avoid inplace operations
	w_m1 = torch.cat([w_m1[:1], (zisoi[:nlevdecomp+1][:-1] - zsoi[:nlevdecomp+1][:-1]) / dz_node[:nlevdecomp+1][1:]])
	# At the bottom, assume no gradient in dz (i.e., they're the same)
	# w_p1[:-2] = (zsoi[:nlevdecomp+1][1:-1] - zisoi[:nlevdecomp+1][:-2]) / dz_node[:nlevdecomp+1][1:-1]  # Skip the last two layers for w_p1
	# Try to avoid inplace operations
	w_p1 = torch.cat([(zsoi[:nlevdecomp+1][1:-1] - zisoi[:nlevdecomp+1][:-2]) / dz_node[:nlevdecomp+1][1:-1], w_p1[-2:]])
	

	# Harmonic mean for internal layers, with adjustments for boundary conditions
	inner_layers = (diffus[1:] > 0) & (diffus[:-1] > 0)  # Boolean mask for layers where both adjacent diffusivities are > 0
	# d_m1_zm1[1:] = torch.where(inner_layers, 1. / ((1. - w_m1[1:]) / diffus[1:] + w_m1[1:] / diffus[:-1]), 0.)
	# d_p1_zp1[:-1] = torch.where(inner_layers, 1. / ((1. - w_p1[:-1]) / diffus[:-1] + w_p1[:-1] / diffus[1:]), (1. - w_m1[:-1]) * diffus[:-1] + w_p1[:-1] * diffus[1:])
	d_m1_zm1 = torch.cat([d_m1_zm1[:1], torch.where(inner_layers, 1. / ((1. - w_m1[1:]) / diffus[1:] + w_m1[1:] / diffus[:-1]), torch.zeros_like(d_m1_zm1[1:]))])
	d_p1_zp1_temp = torch.where(inner_layers, 1. / ((1. - w_p1[:-1]) / diffus[:-1] + w_p1[:-1] / diffus[1:]), (1. - w_m1[:-1]) * diffus[:-1] + w_p1[:-1] * diffus[1:])
	d_p1_zp1 = torch.cat([d_p1_zp1_temp, d_p1_zp1[-1:]])

	# Adjust for dz_node scaling
	d_m1_zm1[1:] /= dz_node[:nlevdecomp+1][1:]
	d_p1_zp1[:-1] /= dz_node[:nlevdecomp+1][1:]

	# print("d_p1_zp1", d_p1_zp1)


	# Layer lower than nbedrock-1: d_p1_zp1 = d_m1_zm1
	

	# Bottom layer - assume no gradient in dz
	# w_m1[-1] = (zisoi[:nlevdecomp+1][-2] - zsoi[:nlevdecomp+1][-2]) / dz_node[:nlevdecomp+1][-1]
	# d_m1_zm1[-1] = 1. / ((1. - w_m1[-1]) / diffus[-1] + w_m1[-1] / diffus[-2]) if diffus[-1] > 0 and diffus[-2] > 0 else 0.
	# d_m1_zm1[-1] /= dz_node[:nlevdecomp+1][-1]
	w_m1_bottom = (zisoi[:nlevdecomp+1][-2] - zsoi[:nlevdecomp+1][-2]) / dz_node[:nlevdecomp+1][-1]
	d_m1_zm1_bottom = 1. / ((1. - w_m1_bottom) / diffus[-1] + w_m1_bottom / diffus[-2]) if diffus[-1] > 0 and diffus[-2] > 0 else 0.
	d_m1_zm1_bottom /= dz_node[:nlevdecomp+1][-1]

	d_m1_zm1= torch.cat([d_m1_zm1[:-1], d_m1_zm1_bottom.unsqueeze(0)])
	d_p1_zp1 = torch.cat([d_p1_zp1[:nbedrock], d_m1_zm1[nbedrock:]])


	# d_p1_zp1[nbedrock-1:] = d_m1_zm1[nbedrock-1:]
	# # No advective flux for the layer between nbedrock and nlevdecomp
	# f_p1[nbedrock-1:] = 0

	# No advective flux for the layer between nbedrock and nlevdecomp without in-place modification
	f_p1 = torch.cat([f_p1[:nbedrock], torch.zeros_like(f_p1[nbedrock:])])

	# assign 0 if nlevdecomp > nbedrock
	# for example, if nbedrock = 7, only select top 6 elements
	w_p1 = torch.where(torch.arange(nlevdecomp+1).to(device) > nbedrock-1, torch.zeros_like(w_p1), w_p1)

	# Boundary conditions
	# Top layer
	w_p1[0] = (zsoi[:nlevdecomp+1][1] - zisoi[:nlevdecomp+1][0]) / dz_node[:nlevdecomp+1][1]
	d_p1_zp1[0] = 1. / ((1. - w_p1[0]) / diffus[0] + w_p1[0] / diffus[1]) if diffus[1] > 0 and diffus[0] > 0 else 0.
	d_p1_zp1[0] /= dz_node[:nlevdecomp+1][1]

	# Peclet numbers
	pe_m1 = torch.where(d_m1_zm1 == 0, torch.zeros_like(f_m1), f_m1 / (d_m1_zm1 + epsilon))
	pe_p1 = torch.where(d_p1_zp1 == 0, torch.zeros_like(f_p1), f_p1 / (d_p1_zp1 + epsilon))

	# Pre-compute the 'aaa' values for Patankar functions
	aaa_m = torch.maximum(torch.zeros_like(pe_m1), (1. - 0.1 * pe_m1.abs())**5)
	aaa_p = torch.maximum(torch.zeros_like(pe_p1), (1. - 0.1 * pe_p1.abs())**5)

	# Vectorized computation of tridiagonal coefficients
	a_tri_e = -(d_m1_zm1 * aaa_m + torch.maximum(f_m1, torch.zeros_like(f_m1)))
	c_tri_e = -(d_p1_zp1 * aaa_p + torch.maximum(-f_p1, torch.zeros_like(f_p1)))
	b_tri_e = -a_tri_e - c_tri_e

	a_tri_dz = a_tri_e / dz[:nlevdecomp+1]
	b_tri_dz = b_tri_e / dz[:nlevdecomp+1]
	c_tri_dz = c_tri_e / dz[:nlevdecomp+1]
	a_tri_dz = a_tri_dz[:-1]
	b_tri_dz = b_tri_dz[:-1]
	c_tri_dz = c_tri_dz[:-1]

	# Try to get rid of the for loop
	# Expand a_tri_dz, b_tri_dz, c_tri_dz
	expanded_a = torch.cat([torch.zeros(20, device=device), a_tri_dz.repeat(npool-1)])
	expanded_b = torch.cat([torch.zeros(20, device=device), b_tri_dz.repeat(npool-1)])
	expanded_c = torch.cat([torch.zeros(20, device=device), c_tri_dz.repeat(npool-1)])

	# Fill diagonal with expanded_b
	tri_ma.fill_diagonal_(0)  # Ensure diagonal is clear before setting
	tri_ma[torch.arange(140), torch.arange(140)] = expanded_b

	# Adjust for upper boundary conditions across blocks
	upper_boundary_indices = torch.arange(20, 140, 20)
	tri_ma[upper_boundary_indices, upper_boundary_indices] = -expanded_c[upper_boundary_indices]

	# Adjust for bottom boundary conditions across blocks
	bottom_boundary_indices = torch.arange(39, 140, 20)
	tri_ma[bottom_boundary_indices, bottom_boundary_indices] = -expanded_a[bottom_boundary_indices]

	# Fill off-diagonals for c_tri_dz
	off_diag_indices_right = torch.arange(19, 139)  # Right off-diagonal
	tri_ma[torch.arange(19, 139), torch.arange(20, 140)] = expanded_c[off_diag_indices_right]
	

	# Fill off-diagonals for a_tri_dz
	off_diag_indices_left = torch.arange(20, 140)  # Left off-diagonal
	tri_ma[torch.arange(20, 140), torch.arange(19, 139)] = expanded_a[off_diag_indices_left]

	# Adjust for upper boundary conditions across blocks
	tri_ma[upper_boundary_indices, upper_boundary_indices-1] = 0
	
	# Adjust for bottom boundary conditions across blocks
	bottom_boundary_indices = torch.arange(39, 120, 20)
	tri_ma[bottom_boundary_indices, bottom_boundary_indices+1] = 0
	

	return tri_ma

# end def tri_matrix

def catanf(t1):
	catanf_results = 11.75 +(29.7 / np.pi) * torch.arctan( torch.tensor(np.pi) * 0.031  * ( t1 - 15.4 ))
	return catanf_results



