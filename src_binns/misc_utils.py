import collections
import torch
import torch.nn
import numpy as np
from mlp import mlp_wrapper, nn_only
import warnings
import os
import matplotlib.pyplot as plt
import math


@torch.no_grad()
def update_bn_custom(loader, model, device=None):
	r"""Method from SWA library, but adapted to use our custom data,
	where each dataloader batch contains x,z,y,profile_id and x,z
	need to be passed through the model.

	Documentation from SWA:
	Updates BatchNorm running_mean, running_var buffers in the model.
	It performs one pass over data in `loader` to estimate the activation
	statistics for BatchNorm layers in the model.
	Args:
		loader (torch.utils.data.DataLoader): dataset loader to compute the
			activation statistics on. Each data batch should be either a
			tensor, or a list/tuple whose first element is a tensor
			containing data.
		model (torch.nn.Module): model for which we seek to update BatchNorm
			statistics.
		device (torch.device, optional): If set, data will be transferred to
			:attr:`device` before being passed into :attr:`model`.

	Example:
		>>> # xdoctest: +SKIP("Undefined variables")
		>>> loader, model = ...
		>>> torch.optim.swa_utils.update_bn(loader, model)

	.. note::
		The `update_bn` utility assumes that each data batch in :attr:`loader`
		is either a tensor or a list or tuple of tensors; in the latter case it
		is assumed that :meth:`model.forward()` should be called on the first
		element of the list or tuple corresponding to the data batch.
	"""
	momenta = {}
	for module in model.modules():
		if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
			module.reset_running_stats()
			momenta[module] = module.momentum

	if not momenta:
		return

	was_training = model.training
	model.train()
	for module in momenta.keys():
		module.momentum = None

	for input in loader:
		batch_x, batch_y, batch_z, batch_profile_id = input
		if device is not None:
			batch_x = batch_x.to(device)
			batch_z = batch_z.to(device)

		model(batch_x, batch_z, whether_predict=0)

	for bn_module in momenta.keys():
		bn_module.momentum = momenta[bn_module]
	model.train(was_training)



def zero_gradients(x):
	"""
	This was formerly an old function in "torch.autograd.gradcheck".
	It is now removed so we reproduce the code here, see
	https://discuss.pytorch.org/t/from-torch-autograd-gradcheck-import-zero-gradients/127462
	"""
	if isinstance(x, torch.Tensor):
		if x.grad is not None:
			x.grad.detach_()
			x.grad.zero_()
	elif isinstance(x, collections.abc.Iterable):
		for elem in x:
			zero_gradients(elem)


def select_depth(tensor_simu, tensor_frocing_steady_state, tensor_obs_layer_depth):
	simu = tensor_simu
	# para = (tensor_para - (-1)) /(1 - (-1)) # conversion from Hardttanh [-1, 1] to [0, 1]
	frocing_steady_state = tensor_frocing_steady_state
	obs_layer_depth = tensor_obs_layer_depth
	device = tensor_simu.device

	# depth of the node
	zsoi_local = torch.tensor([1.000000000000000E-002, 4.000000000000000E-002, 9.000000000000000E-002, \
		0.160000000000000, 0.260000000000000, 0.400000000000000, \
		0.580000000000000, 0.800000000000000, 1.06000000000000, \
		1.36000000000000, 1.70000000000000, 2.08000000000000, \
		2.50000000000000, 2.99000000000000, 3.58000000000000, \
		4.27000000000000, 5.06000000000000, 5.95000000000000, \
		6.94000000000000, 8.03000000000000, 9.79500000000000, \
		13.3277669529664, 19.4831291701244, 28.8707244343160, \
		41.9984368640029])
	zsoi = zsoi_local.to(device)
	n_soil_layer = 20

	# final ouputs of simulation
	profile_num = simu.shape[0]
	simu_ouput = (torch.ones((profile_num, 200))*np.nan).to(device)
	# calculate soc solution for each profile
	for iprofile in range(0, profile_num):
		profile_simu = tensor_simu[iprofile, :]
		profile_force_steady_state = frocing_steady_state[iprofile, :, :, :]
		profile_obs_layer_depth = obs_layer_depth[iprofile, :]
		valid_layer_loc = torch.where(torch.isnan(profile_obs_layer_depth) == False)[0]

		if torch.isnan(torch.sum(profile_simu)) == False and \
			torch.isnan(torch.sum(profile_force_steady_state[0:12, 0, 1:8])) == False and \
			torch.isnan(torch.sum(profile_force_steady_state[0:20, 0:12, 8:13])) == False:

			# print(profile_para)
			# model simulation
			profile_simu_soc = profile_simu

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
					simu_ouput[iprofile, valid_layer_loc[ilayer]] = profile_simu_soc[node_depth_lower_loc]
				else:
					simu_ouput[iprofile, valid_layer_loc[ilayer]] = \
					profile_simu_soc[node_depth_lower_loc] \
					+ (profile_simu_soc[node_depth_upper_loc] - profile_simu_soc[node_depth_lower_loc]) \
					/(zsoi[node_depth_upper_loc] - zsoi[node_depth_lower_loc]) \
					*(layer_depth - zsoi[node_depth_lower_loc])
			# end for
		# end if
	#end for iprofile
	return simu_ouput
# end nn_model



def inject_noise(model, noise_std):
	"""
	Adds a small amount of random noise to the parameters of the network.

	Source: https://github.com/shibhansh/loss-of-plasticity/blob/main/lop/incremental_cifar/incremental_cifar_experiment.py
	"""
	with torch.no_grad():
		for param in model.parameters():
			param.add_(torch.randn(param.size(), device=param.device) * noise_std)



def print_summary(tensor, message="", dim=None):
	if dim is None:
		print(message, "- Shape", tensor.shape, "Mean", tensor.mean(), "Std", tensor.std(), "Min", tensor.min(), "Max", tensor.max())
	else:
		print(message, "- Shape", tensor.shape, "Mean", tensor.mean(dim=dim), "Std", tensor.std(dim=dim), "Min", tensor.min(dim=dim).values, "Max", tensor.max(dim=dim).values)


import sys

class Logger(object):
	"""
	Logger that writes printed statements to both stdout (terminal) and log file.
	To use, set "sys.stdout = Logger(log_file)"
	Source: https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
	"""
	def __init__(self, log_file):
		self.terminal = sys.stdout
		self.log = open(log_file, "a")
   
	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)  

	def flush(self):
		# this flush method is needed for python 3 compatibility.
		# this handles the flush command by doing nothing.
		# you might want to specify some extra behavior here.
		pass


def get_activation(activation):
	if activation == 'relu':
		return torch.nn.ReLU()
	elif activation == 'leaky_relu':
		return torch.nn.LeakyReLU(negative_slope=0.3)
	elif activation == 'tanh':
		return torch.nn.Tanh()
	else:
		raise ValueError("Unsupported activation")
	return act


def get_param_constraint(param_constraint):
	if param_constraint == "sigmoid":
		return torch.nn.Sigmoid()
	elif param_constraint == "hardsigmoid":
		return torch.nn.Hardsigmoid()
	elif param_constraint == "none":
		return torch.nn.Identity()
	else:
		raise ValueError("Invalid param_constraint")


def get_model(args, var4nn, var_idx_to_emb, device, para_index, train_x, train_y):
	"""
	Given the commandline args, returns the correct model class and a dict of kwargs
	"""
	if args.model in ["new_mlp", "lipmlp", "senn", "nam", "nam_joint", "nam_joint2", "nag", "kan"]:
		model_class = mlp_wrapper
		model_kwargs = {"input_vars": len(var4nn),
						"var_idx_to_emb": var_idx_to_emb,
						"vertical_mixing": args.vertical_mixing,
						"vectorized": args.vectorized,
						"pos_enc": args.pos_enc,
						"base_model": args.model,
						"one_hot": (args.categorical == "one_hot"),
						"use_bn": args.use_bn,
						"dropout_prob": args.dropout_prob,
						"activation": args.activation,
						"param_constraint": args.param_constraint,  
						"device": device,
						"min_temp": args.min_temp,
						"max_temp": args.max_temp,
						"init": args.init,
						"width": args.width,
						"num_layers": args.num_layers,
						"residual": args.residual,
						"para_index": para_index,
						"kan_grid": args.kan_grid,
						"kan_grid_margin": args.kan_grid_margin,
						"kan_noise": args.kan_noise,
						"kan_base_fun": args.kan_base_fun,
						"kan_affine_trainable": args.kan_affine_trainable,
						"kan_absolute_deviation": args.kan_absolute_deviation}

	elif args.model == 'nn_only':
		model_class = nn_only

		# Calculate mean/std of Y
		if args.standardize_output:
			output_values = train_y.flatten()[~torch.isnan(train_y.flatten())]
			output_mean = torch.mean(output_values)
			output_std = torch.std(output_values)
		else:
			output_mean, output_std = None, None
		print("NN only, output_mean", output_mean, "output_std", output_std)

		model_kwargs = {"input_vars": len(var4nn),
						"var_idx_to_emb": var_idx_to_emb,
						"pos_enc": args.pos_enc,
						"output_dim": 140,
						"base_model": "new_mlp",
						"one_hot": (args.categorical == "one_hot"),
						"use_bn": args.use_bn,
						"dropout_prob": args.dropout_prob,
						"activation": args.activation,
						"output_mean": output_mean,
						"output_std": output_std,
						"init": args.init,
						"width": args.width,
						"num_layers": args.num_layers,
						"residual": args.residual}

	else:
		raise ValueError("Invalid args.model")

	# Standardize input if requested
	if args.standardize_input:
		model_kwargs["train_x"] = train_x.to(device)
	return model_class, model_kwargs



@torch.no_grad()
def get_wd_params(model: torch.nn.Module):
    """
    Do not apply weight decay on biases.
    Returns (1) list of parameters to apply weight decay (all non-bias params)
    and (2) list of parameters to NOT apply weight decay (biases).

    Source: https://discuss.pytorch.org/t/weight-decay-only-for-weights-of-nn-linear-and-nn-conv/114348
    """
    decay = list()
    no_decay = list()
    for name, param in model.named_parameters():
        # print('checking {}'.format(name))
        if hasattr(param,'requires_grad') and not param.requires_grad:
            continue
        if 'bias' in name:
            no_decay.append(param)
        else:
            decay.append(param)
        # if 'weight' in name and 'norm' not in name and 'bn' not in name:
        #     decay.append(param)
        # else:
        #     no_decay.append(param)
    return decay, no_decay


def get_optimizer_and_scheduler(model, args):
	# Optimizer
	no_decay, decay = get_wd_params(model)
	if args.optimizer == "AdamW":
		optimizer = torch.optim.AdamW([{'params': no_decay, 'weight_decay': 0},
										{'params': decay, 'weight_decay': args.weight_decay}],
										lr=args.lr)
		# optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	elif args.optimizer == "SGD":
		optimizer = torch.optim.SGD([{'params': no_decay, 'weight_decay': 0},
										{'params': decay, 'weight_decay': args.weight_decay}],
										lr=args.lr, momentum=args.momentum)
		# optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
	# elif args.optimizer == "LBFGS":
	#   # NOTE: currently disabling LBFGS as it requires creating a closure to wrap the forward pass, which is ugly.
	# 	# From the KAN repo https://github.com/KindXiaoming/pykan/blob/master/kan/MultKAN.py#L1498
	# 	optimizer = torch.optim.LBFGS(model.parameters(), lr=args.lr, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32)  #, tolerance_ys=1e-32)
	else:
		raise ValueError("Invalid args.optimizer")
	# print("GETTING OPTIMIZER", list(model.parameters()))

	# If desired, add a learning rate scheduler that decays the learning rate throughout training
	if args.scheduler == "reduce_on_plateau":
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.2)  #, mode="max")
	elif args.scheduler == "step":
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
	elif args.scheduler == "cosine":
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
	elif args.scheduler == "cosine_restarts":
		scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20)
	elif args.scheduler == "none":
		scheduler = None
	else:
		raise ValueError("Invalid args.scheduler")
	return optimizer, scheduler


def kl_divergence(true, pred):
	"""
	Assumes each COLUMN of true/pred is a prob dist, Tensor or numpy
	"""
	if torch.is_tensor(true):
		kl = (true * torch.log(1e-6 + true / pred)).sum(dim=0).mean()
	else:  # Assume Numpy
		kl = (true * np.log(1e-6 + true / pred)).sum(axis=0).mean()
	return kl



def compute_metrics(true, pred):
	from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
	non_nan = ~np.isnan(true) & ~np.isnan(pred)
	true, pred = true[non_nan], pred[non_nan]
	r2 = r2_score(true, pred)
	mse = mean_squared_error(true, pred)
	mae = mean_absolute_error(true, pred)
	corr = np.corrcoef(true, pred)[0, 1]
	return r2, mse, mae, corr


def plot_functional_relationships(args, best_guess_model, input_names, output_names, plot_dir, epoch_str, true_relationships=None):
	"""
	Plot functional relationships learned by the model
	"""
	# Test functional relationship retrieval
	if args.model != "nn_only":
		cached_nn_input = best_guess_model.new_input

		# Feature importance by Jacobian
		jacobian = best_guess_model.get_jacobian(cached_nn_input)  # [batch, n_params, n_inputs]
		avg_jacobian_magnitude = jacobian.abs().mean(dim=0).detach().cpu().numpy().T  # transpose to [n_inputs, n_params]
		jacobian_importances = avg_jacobian_magnitude / avg_jacobian_magnitude.sum(axis=0, keepdims=True)

		# Special predictor function to use NN (alibi library)
		from alibi.explainers import ALE, PartialDependenceVariance, plot_ale, plot_pd_variance
		@torch.no_grad()
		def predictor(X: np.ndarray) -> np.ndarray:
			assert best_guess_model.mlp.training == False, "Model must be in eval mode"
			X = torch.as_tensor(X, device=cached_nn_input.device)
			return best_guess_model.mlp(X).cpu().numpy()

		# Feature importance: log PDP Variance scores
		with warnings.catch_warnings():  # suppress warnings inside alibi code
			warnings.simplefilter("ignore")
			pd_variance = PartialDependenceVariance(predictor=predictor,
													feature_names=input_names,
													target_names=output_names)
			exp_importance = pd_variance.explain(cached_nn_input.detach().cpu().numpy(), method='importance')
			importance_scores = exp_importance.data['feature_importance'].T  # transpose to [n_inputs, n_params]
			pdv_importances = importance_scores / importance_scores.sum(axis=0, keepdims=True)

			# Plot PDP variance
			plot_pd_variance(exp=exp_importance)
			plt.savefig(os.path.join(plot_dir, f"{epoch_str}_pd_variance.png"))
			plt.close()

			# Accumulated Local Effects plot (similar to partial dependence plot
			# but better with correlated features)
			ale = ALE(predictor, feature_names=input_names, target_names=output_names)
			exp = ale.explain(cached_nn_input.detach().cpu().numpy())
			plot_ale(exp)
			plt.savefig(os.path.join(plot_dir, f"{epoch_str}_ale.png"))
			plt.close()

		method_str = "Blackbox-Hybrid" if args.model == "new_mlp" else f"KAN {args.num_layers}-layer"
		predicted_importances_all = {f"{method_str} (Jacobian)": jacobian_importances,
										f"{method_str}\n(Partial Dependence Variance)": pdv_importances}
		if args.model == "kan" and args.num_layers == 1:
			# If using one-layer KAN, read off functional relationships with mask
			kan_importances = best_guess_model.mlp.edge_scores[0].permute(1, 0).detach().cpu().numpy()
			predicted_importances_all["KAN"] = kan_importances / kan_importances.sum(axis=0, keepdims=True)

			# set importances of pruned edges to zero
			kan_pruned_importances = kan_importances.copy()
			kan_pruned_importances[best_guess_model.mlp.act_fun[0].mask == 0] = 0.
			predicted_importances_all["KAN_pruned"] = kan_pruned_importances / kan_pruned_importances.sum(axis=0, keepdims=True)

		# Save picture of functional relationships. Source: https://stackoverflow.com/questions/69986007/matplotlib-imshow-with-1-color-for-each-discrete-value
		n_plots = len(predicted_importances_all) if true_relationships is None else len(predicted_importances_all)+1
		fig, axeslist = plt.subplots(1, n_plots, figsize=(6*n_plots, 6))
		cmap = plt.get_cmap('Greens')
		for pred_idx, (pred_method, pred_rel) in enumerate(predicted_importances_all.items()):
			im = axeslist[pred_idx].imshow(pred_rel, cmap=cmap, vmin=0, vmax=1)  #, vmin=-0.5, vmax=5.5, cmap=cmap, interpolation="none")
			axeslist[pred_idx].set_xticks(np.arange(len(output_names)))
			axeslist[pred_idx].set_yticks(np.arange(len(input_names)))
			axeslist[pred_idx].set_xticklabels(output_names, rotation='vertical')
			axeslist[pred_idx].set_yticklabels(input_names)

			# If functional relationships are known, compare KAN's predicted relationships with ground-truth relationships
			if true_relationships is not None:
				relationship_kl = kl_divergence(true_relationships, pred_rel)
				relationship_l2 = math.sqrt(((true_relationships - pred_rel) ** 2).sum())
				functional_acc_str = f"\n(KL: {relationship_kl:.3f}, Euclidean dist: {relationship_l2:.3f})"
			else:
				functional_acc_str = ""
			axeslist[pred_idx].set_title(f"Predicted by {pred_method}{functional_acc_str}")

		if true_relationships is not None:
			im = axeslist[-1].imshow(true_relationships, cmap=cmap, vmin=0, vmax=1)  #, vmin=-0.5, vmax=5.5, cmap=cmap, interpolation="none")
			axeslist[-1].set_xticks(np.arange(len(output_names)))
			axeslist[-1].set_yticks(np.arange(len(input_names)))
			axeslist[-1].set_xticklabels(output_names, rotation='vertical')
			axeslist[-1].set_yticklabels(input_names)
			axeslist[-1].set_title("Ground-truth")
		fig.colorbar(im, orientation="vertical", ax=axeslist[-1])
		plt.tight_layout()
		plt.savefig(os.path.join(plot_dir, f"{epoch_str}_functional_relationships.png"))
		plt.close()

	# Return predicted relationships for "default" method (+accuracy metrics if true relationships known)
	DEFAULT_REL = "KAN" if "KAN" in predicted_importances_all else f"{method_str}\n(Partial Dependence Variance)"
	if args.model != "nn_only" and true_relationships is not None:
		default_kl = kl_divergence(true_relationships, predicted_importances_all[DEFAULT_REL])
		default_l2 = math.sqrt(((true_relationships - predicted_importances_all[DEFAULT_REL]) ** 2).sum())
	else:
		default_kl, default_l2 = None, None
	return predicted_importances_all[DEFAULT_REL], default_kl, default_l2