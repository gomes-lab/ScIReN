import torch
import visualization_utils


def binns_loss(y_pred, y_true, pred_para, plot_path=""):
	"""
	Computes the main losses used in BINN.
	 
	y_pred: predicted SOC, shape [batch, observations_per_site]
	y_true: true_SOC, shape [batch, observations_per_site]
	Many entries can be nan (nan positions should be the same for y_pred and y_true).
	pred_para: predicted parameters, shape [batch, num_params]

	Returns
	1) L1 loss
	2) Smooth L1 loss
	3) L2 loss
	4) Parameter regularization loss (penalizes extreme param values)
	5) Modeling inefficiency (1 - NSE, or 1 - R^2)

	All of these (except parameter regularization loss) compare how close
	predictions (y_pred) are with observations (y_true).
	"""
	# Predicted (simulated) SOC
	soc_simu = y_pred[:, 0, :]
	POM_simu = y_pred[:, 1, :]
	MAOM_simu = y_pred[:, 2, :]

	# Observed SOC
	soc_true = y_true[:, 0, :]
	POM_true = y_true[:, 1, :]
	MAOM_true = y_true[:, 2, :]

	# Predicted parameters
	pred_para = pred_para

	# flatten simulated and true SOC
	soc_simu_vector = torch.reshape(soc_simu, [1, -1])
	POM_simu_vector = torch.reshape(POM_simu, [1, -1])
	MAOM_simu_vector = torch.reshape(MAOM_simu, [1, -1])
	soc_true_vector = torch.reshape(soc_true, [1, -1])
	POM_true_vector = torch.reshape(POM_true, [1, -1])
	MAOM_true_vector = torch.reshape(MAOM_true, [1, -1])

	# exclude nan
	valid_loc = torch.where(torch.isnan(soc_simu_vector+soc_true_vector) == False)
	soc_simu_vector = soc_simu_vector[valid_loc]
	soc_true_vector = soc_true_vector[valid_loc]

	valid_loc = torch.where(torch.isnan(POM_simu_vector+POM_true_vector) == False)
	POM_simu_vector = POM_simu_vector[valid_loc]
	POM_true_vector = POM_true_vector[valid_loc]

	valid_loc = torch.where(torch.isnan(MAOM_simu_vector+MAOM_true_vector) == False)
	MAOM_simu_vector = MAOM_simu_vector[valid_loc]
	MAOM_true_vector = MAOM_true_vector[valid_loc]

	# modeling inefficiency
	modeling_inefficiency_soc = torch.sum((soc_simu_vector - soc_true_vector)**2)/torch.sum((soc_true_vector - torch.mean(soc_true_vector))**2)
	if valid_loc[0].shape[0] == 1:
		modeling_inefficiency_POM = torch.abs(POM_simu_vector - POM_true_vector) / (POM_true_vector)
	else:
		modeling_inefficiency_POM = torch.sum((POM_simu_vector - POM_true_vector)**2)/torch.sum((POM_true_vector - torch.mean(POM_true_vector))**2)
	if valid_loc[0].shape[0] == 1:
		modeling_inefficiency_MAOM = torch.abs(MAOM_simu_vector - MAOM_true_vector) / (MAOM_true_vector)
	else:
		modeling_inefficiency_MAOM = torch.sum((MAOM_simu_vector - MAOM_true_vector)**2)/torch.sum((MAOM_true_vector - torch.mean(MAOM_true_vector))**2)


	# If desired, plot true vs predicted here
	if plot_path != "":
		visualization_utils.plot_true_vs_predicted(plot_path, soc_simu_vector, soc_true_vector)

	# Regularization for predicted parameters using cosh
	# Encourage parameters to be around 0.5
	target_value = 0.5
	scale_factor = 10
	param_reg_loss = torch.mean(torch.cosh(scale_factor*(pred_para - target_value)) - 1)

	# Weighting factor for regularization term
	regularization_weight = 100

	# z-score normalization
	def z(x, μσ): return (x - μσ[0]) / μσ[1]

	# Calculate the supervised losses
	l1_loss = torch.nn.functional.smooth_l1_loss(soc_simu_vector, soc_true_vector, reduction='mean')
	# l1_loss = torch.nn.functional.l1_loss(soc_simu_vector, soc_true_vector, reduction='mean')
	l1_loss_POM = torch.nn.functional.smooth_l1_loss(POM_simu_vector, POM_true_vector, reduction='mean')
	# l1_loss_POM = torch.nn.functional.mse_loss(POM_simu_vector, POM_true_vector, reduction='mean')
	l1_loss_MAOM = torch.nn.functional.smooth_l1_loss(MAOM_simu_vector, MAOM_true_vector, reduction='mean')

	# # Define the scales for z-score normalization
	# scales = {
	# 	'soc': (torch.mean(soc_true_vector), torch.std(soc_true_vector)),
	# 	'POM': (torch.mean(POM_true_vector), torch.std(POM_true_vector)),
	# 	'MAOM': (torch.mean(MAOM_true_vector), torch.std(MAOM_true_vector))
	# }

	# # Calculate the normalized losses
	# l1_loss = torch.nn.functional.smooth_l1_loss(z(soc_simu_vector, scales['soc']), z(soc_true_vector, scales['soc']), reduction='mean')
	# l1_loss_POM = torch.nn.functional.smooth_l1_loss(z(POM_simu_vector, scales['POM']), z(POM_true_vector, scales['POM']), reduction='mean')
	# l1_loss_MAOM = torch.nn.functional.smooth_l1_loss(z(MAOM_simu_vector, scales['MAOM']), z(MAOM_true_vector, scales['MAOM']), reduction='mean')
	
	# # Calculate the loss of log values
	# tricky = 1e-6
	# l1_loss = torch.nn.functional.smooth_l1_loss(torch.log(soc_simu_vector + tricky), torch.log(soc_true_vector + tricky), reduction='sum')
	# l1_loss_POM = torch.nn.functional.smooth_l1_loss(torch.log(POM_simu_vector + tricky), torch.log(POM_true_vector + tricky), reduction='sum')
	# l1_loss_MAOM = torch.nn.functional.smooth_l1_loss(torch.log(MAOM_simu_vector + tricky), torch.log(MAOM_true_vector + tricky), reduction='sum')
	if torch.isnan(l1_loss_POM):
		l1_loss_POM = torch.tensor(0.0)
		modeling_inefficiency_POM = torch.tensor(0.0)
	if torch.isnan(l1_loss_MAOM):
		l1_loss_MAOM = torch.tensor(0.0)
		modeling_inefficiency_MAOM = torch.tensor(0.0)

	POM_weight = 1
	MAOM_weight = 1

	loss = l1_loss + regularization_weight * param_reg_loss + POM_weight * l1_loss_POM + MAOM_weight * l1_loss_MAOM

	return loss, modeling_inefficiency_soc, modeling_inefficiency_POM, modeling_inefficiency_MAOM, l1_loss, l1_loss_POM, l1_loss_MAOM, param_reg_loss
# end binns loss



def compute_param_violation_loss(unconstrained_params):
	"""
	If using hardsigmoid param_constraint, penalizes if unconstrained_params are outside the [-3, 3] range.
	"""
	return torch.sum(torch.clamp(torch.abs(unconstrained_params) - 3.0, min=0))


def compute_unconstrained_param_loss(unconstrained_params):
	"""
	If using hardsigmoid param_constraint, penalizes unconstrained values far away from zero.
	"""
	return (unconstrained_params ** 2).mean()


def compute_param_matching_loss(pred_para, true_para):
	"""
	Simple L2 loss on the parameters. This is cheating but can be used for debugging.
	"""
	return torch.nn.functional.mse_loss(pred_para, true_para)