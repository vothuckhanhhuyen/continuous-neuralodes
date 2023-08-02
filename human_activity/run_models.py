###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt

import time
import datetime
import argparse
import numpy as np
import pandas as pd
from random import SystemRandom
from sklearn import model_selection

import torch
import torch.nn as nn
from torch.nn.functional import relu
import torch.optim as optim

import lib.utils as utils
from lib.plotting import *

from lib.rnn_baselines import *
from lib.ode_rnn import *
from lib.create_latent_ode_model import create_LatentODE_model
from lib.parse_datasets import parse_datasets
from lib.ode_func import ODEFunc, ODEFunc_w_Poisson
from lib.diffeq_solver import DiffeqSolver
from mujoco_physics import HopperPhysics

from lib.utils import compute_loss_all_batches

import csv

# Generative model for noisy data based on ODE
parser = argparse.ArgumentParser('Latent ODE')
parser.add_argument('-n',  type=int, default=100, help="Size of the dataset")
parser.add_argument('--niters', type=int, default=300)
parser.add_argument('--lr',  type=float, default=1e-2, help="Starting learning rate.")
parser.add_argument('-b', '--batch-size', type=int, default=50)
parser.add_argument('--viz', action='store_true', help="Show plots while training")

parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('-r', '--random-seed', type=int, default=1991, help="Random_seed")

parser.add_argument('--dataset', type=str, default='periodic', help="Dataset to load. Available: physionet, activity, hopper, periodic")
parser.add_argument('-s', '--sample-tp', type=float, default=None, help="Number of time points to sub-sample."
	"If > 1, subsample exact number of points. If the number is in [0,1], take a percentage of available points per time series. If None, do not subsample")

parser.add_argument('-c', '--cut-tp', type=int, default=None, help="Cut out the section of the timeline of the specified length (in number of points)."
	"Used for periodic function demo.")

parser.add_argument('--quantization', type=float, default=0.1, help="Quantization on the physionet dataset."
	"Value 1 means quantization by 1 hour, value 0.1 means quantization by 0.1 hour = 6 min")

parser.add_argument('--latent-ode', action='store_true', help="Run Latent ODE seq2seq model")
parser.add_argument('--z0-encoder', type=str, default='odernn', help="Type of encoder for Latent ODE model: odernn or rnn")

parser.add_argument('--classic-rnn', action='store_true', help="Run RNN baseline: classic RNN that sees true points at every point. Used for interpolation only.")
parser.add_argument('--rnn-cell', default="gru", help="RNN Cell type. Available: gru (default), expdecay")
parser.add_argument('--input-decay', action='store_true', help="For RNN: use the input that is the weighted average of impirical mean and previous value (like in GRU-D)")

parser.add_argument('--ode-rnn', action='store_true', help="Run ODE-RNN baseline: RNN-style that sees true points at every point. Used for interpolation only.")

parser.add_argument('--rnn-vae', action='store_true', help="Run RNN baseline: seq2seq model with sampling of the h0 and ELBO loss.")

parser.add_argument('-l', '--latents', type=int, default=6, help="Size of the latent state")
parser.add_argument('--rec-dims', type=int, default=20, help="Dimensionality of the recognition model (ODE or RNN).")

parser.add_argument('--rec-layers', type=int, default=1, help="Number of layers in ODE func in recognition ODE")
parser.add_argument('--gen-layers', type=int, default=1, help="Number of layers in ODE func in generative ODE")

parser.add_argument('-u', '--units', type=int, default=100, help="Number of units per layer in ODE func")
parser.add_argument('-g', '--gru-units', type=int, default=100, help="Number of units per layer in each of GRU update networks")

parser.add_argument('--poisson', action='store_true', help="Model poisson-process likelihood for the density of events in addition to reconstruction.")
parser.add_argument('--classif', action='store_true', help="Include binary classification loss -- used for Physionet dataset for hospiral mortality")

parser.add_argument('--linear-classif', action='store_true', help="If using a classifier, use a linear classifier instead of 1-layer NN")
parser.add_argument('--extrap', action='store_true', help="Set extrapolation mode. If this flag is not set, run interpolation mode.")

parser.add_argument('-t', '--timepoints', type=int, default=100, help="Total number of time-points")
parser.add_argument('--max-t',  type=float, default=5., help="We subsample points in the interval [0, args.max_tp]")
parser.add_argument('--noise-weight', type=float, default=0.01, help="Noise amplitude for generated traejctories")

parser.add_argument('--gpu', type=int, default=0, help='The GPU device number')

parser.add_argument('--node', type=str, default='NODE', \
                    choices=['NODE', 'HBNODE', 'GHBNODE', 'NesterovNODE', "G5HDANNODE", "G7HDANNODE", "RMSpropNODEver2", "GRMSpropNODEver2", "GRMSpropNODEver2_altered", "RMSpropNODEver2_3", "GRMSpropNODEver2_3",
                             "RMSpropNODEver2_4", "GRMSpropNODEver2_4", "GRMSpropNODEver2_4_altered", "RMSpropNODEver2_5", "GRMSpropNODEver2_5", "GRMSpropNODEver2_5_altered"])
parser.add_argument('--solver', type=str, choices=['euler', 'dopri5'], default='euler')
parser.add_argument('--atol', type=float, default=1e-5) ## default: 1e-1, 1e-1
parser.add_argument('--rtol', type=float, default=1e-5)

parser.add_argument('--start_time', type=int, default=0)
parser.add_argument('--xi', type=float, default=0.1)
parser.add_argument('--alpha', type=float, default=1e2)
parser.add_argument('--corrf', action='store_true')
parser.add_argument('--alphaf', action='store_false')
parser.add_argument('--actv', type=str, default='tanh')
parser.add_argument('--corr_m', type=float, default=0.1)
parser.add_argument('--corr_mf', action='store_true')

parser.add_argument('--start_epoch', type=int, default=0)

args = parser.parse_args()

device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
file_name = os.path.basename(__file__)[:-3]
utils.makedirs(args.save)
utils.makedirs(os.path.join(args.save, args.node))

#####################################################################################################
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
	torch.manual_seed(args.random_seed)
	np.random.seed(args.random_seed)

	if args.rnn_vae:
		method = "rnn-vae"
	elif args.classic_rnn:
		method = "classic-rnn"
	elif args.ode_rnn:
		method = "ode-rnn"
	elif args.latent_ode:
		method = "latent-ode"
	else:
		raise Exception("Model not specified")

	experimentID = args.load
	if experimentID is None:
		# Make a new experiment ID
		experimentID = int(SystemRandom().random()*100000)
	if args.corrf == False:
		ckpt_path = os.path.join(args.save, args.node, f"{method}_{args.node}_{args.atol}_{args.rtol}_{args.alpha}_{args.alphaf}_{args.xi}_{args.corr_m}_{args.actv}_experiment_" + str(experimentID) + '.ckpt')
	else:
		ckpt_path = os.path.join(args.save, args.node, f"{method}_{args.node}_{args.atol}_{args.rtol}_{args.alpha}_{args.alphaf}_experiment_" + str(experimentID) + '.ckpt')
	start = time.time()
	print("Sampling dataset of {} training examples".format(args.n))
	
	input_command = sys.argv
	ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
	if len(ind) == 1:
		ind = ind[0]
		input_command = input_command[:ind] + input_command[(ind+2):]
	input_command = " ".join(input_command)

	# utils.makedirs("results/")

	##################################################################
	data_obj = parse_datasets(args, device)
	input_dim = data_obj["input_dim"]

	classif_per_tp = False
	if ("classif_per_tp" in data_obj):
		# do classification per time point rather than on a time series as a whole
		classif_per_tp = data_obj["classif_per_tp"]

	if args.classif and (args.dataset == "hopper" or args.dataset == "periodic"):
		raise Exception("Classification task is not available for MuJoCo and 1d datasets")

	n_labels = 1
	if args.classif:
		if ("n_labels" in data_obj):
			n_labels = data_obj["n_labels"]
		else:
			raise Exception("Please provide number of labels for classification task")

	##################################################################
	# Create the model
	obsrv_std = 0.01
	if args.dataset == "hopper":
		obsrv_std = 1e-3 

	obsrv_std = torch.Tensor([obsrv_std]).to(device)

	z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))

	if args.rnn_vae:
		if args.poisson:
			print("Poisson process likelihood not implemented for RNN-VAE: ignoring --poisson")

		# Create RNN-VAE model
		model = RNN_VAE(input_dim, args.latents, 
			device = device, 
			rec_dims = args.rec_dims, 
			concat_mask = True, 
			obsrv_std = obsrv_std,
			z0_prior = z0_prior,
			use_binary_classif = args.classif,
			classif_per_tp = classif_per_tp,
			linear_classifier = args.linear_classif,
			n_units = args.units,
			input_space_decay = args.input_decay,
			cell = args.rnn_cell,
			n_labels = n_labels,
			train_classif_w_reconstr = (args.dataset == "physionet")
			).to(device)


	elif args.classic_rnn:
		if args.poisson:
			print("Poisson process likelihood not implemented for RNN: ignoring --poisson")

		if args.extrap:
			raise Exception("Extrapolation for standard RNN not implemented")
		# Create RNN model
		model = Classic_RNN(input_dim, args.latents, device, 
			concat_mask = True, obsrv_std = obsrv_std,
			n_units = args.units,
			use_binary_classif = args.classif,
			classif_per_tp = classif_per_tp,
			linear_classifier = args.linear_classif,
			input_space_decay = args.input_decay,
			cell = args.rnn_cell,
			n_labels = n_labels,
			train_classif_w_reconstr = (args.dataset == "physionet")
			).to(device)
	elif args.ode_rnn:
		# Create ODE-GRU model
		n_ode_gru_dims = args.latents
				
		if args.poisson:
			print("Poisson process likelihood not implemented for ODE-RNN: ignoring --poisson")

		if args.extrap:
			raise Exception("Extrapolation for ODE-RNN not implemented")

		if args.node in ("RMSpropNODEver2", "GRMSpropNODEver2", "GRMSpropNODEver2_altered"):
			ode_func_net = utils.create_net(2 * n_ode_gru_dims, n_ode_gru_dims, 
				n_layers = args.rec_layers, n_units = args.units, nonlinear = nn.Tanh)
		elif args.node in ("RMSpropNODEver2_4", "GRMSpropNODEver2_4", "GRMSpropNODEver2_4_altered"):
			ode_func_net1 = utils.create_net(n_ode_gru_dims, n_ode_gru_dims, 
				n_layers = args.rec_layers, n_units = args.units, nonlinear = nn.Tanh)
			ode_func_net2 = utils.create_net(2 * n_ode_gru_dims, n_ode_gru_dims, 
				n_layers = args.rec_layers, n_units = args.units, nonlinear = nn.Tanh)
			ode_func_net = (ode_func_net1, ode_func_net2)
		else:
			ode_func_net = utils.create_net(n_ode_gru_dims, n_ode_gru_dims, 
				n_layers = args.rec_layers, n_units = args.units, nonlinear = nn.Tanh)
		# ode_func_net = utils.create_net(n_ode_gru_dims, n_ode_gru_dims, 
		# 		n_layers = args.rec_layers, n_units = args.units, nonlinear = nn.Tanh)

		rec_ode_func = ODEFunc(
			input_dim = input_dim, 
			latent_dim = n_ode_gru_dims,
			ode_func_net = ode_func_net,
			node=args.node,
			actv=nn.Sigmoid(),
			alpha=args.alpha,
   			corr=args.xi,
      		corrf=args.corrf,
        	corr_m=args.corr_m,
      		corr_mf=args.corr_mf,	
			device = device).to(device)

		# z0_diffeq_solver = DiffeqSolver(input_dim, rec_ode_func, "euler", args.latents, 
		# 	odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)
		z0_diffeq_solver = DiffeqSolver(input_dim, rec_ode_func, args.solver, args.latents, 
			odeint_rtol = args.rtol, odeint_atol = args.atol, device = device, node=args.node)
	
		model = ODE_RNN(input_dim, 
            n_ode_gru_dims, 
            device = device, 
			z0_diffeq_solver = z0_diffeq_solver, 
   			n_gru_units = args.gru_units,
			concat_mask = True, 
   			obsrv_std = obsrv_std,
			use_binary_classif = args.classif,
			classif_per_tp = classif_per_tp,
			n_labels = n_labels,
			train_classif_w_reconstr = (args.dataset == "physionet"),
			node=args.node,
			start_time=args.start_time,
			activation_h=nn.Tanh(),
			).to(device)
	elif args.latent_ode:
		model = create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device, 
			classif_per_tp = classif_per_tp,
			n_labels = n_labels, actv=nn.Tanh(), activation_h=nn.Tanh())
	else:
		raise Exception("Model not specified")

	##################################################################

	if args.viz:
		viz = Visualizations(device)

	##################################################################
	
	#Load checkpoint and evaluate the model
	if args.load is not None:
		model = utils.get_ckpt_model(ckpt_path, model, device)
		# exit()

	##################################################################
	# Training

	if args.corrf == False:
		log_path = os.path.join(args.save, args.node, f"{method}_{args.node}_{args.rtol}_{args.alpha}_{args.alphaf}_{args.xi}_{args.corr_m}_{args.actv}_experiment_" + str(experimentID) + '.log')
	else:
		log_path = os.path.join(args.save, args.node, f"{method}_{args.node}_{args.rtol}_{args.alpha}_{args.alphaf}_experiment_" + str(experimentID) + '.log')
 	# if not os.path.exists("logs/"):
	# 	utils.makedirs("logs/")
	logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
	logger.info(input_command)

	if args.corrf == False:
		csv_path = os.path.join(args.save, args.node, f"{method}_{args.node}_{args.atol}_{args.rtol}_{args.alpha}_{args.alphaf}_{args.xi}_{args.corr_m}_{args.actv}_experiment_" + str(experimentID) + '.csv')
	else:
		csv_path = os.path.join(args.save, args.node, f"{method}_{args.node}_{args.atol}_{args.rtol}_{args.alpha}_{args.alphaf}_experiment_" + str(experimentID) + '.csv')
	
	optimizer = optim.Adamax(model.parameters(), lr=args.lr)

	num_batches = data_obj["n_train_batches"]
 
	print("==> Train model {}, params {}".format(type(model), count_parameters(model)))
	device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
	print("==> Use accelerator: ", device)

	for itr in range(num_batches * args.start_epoch + 1, num_batches * (args.niters + 1)):
		train_start_time = time.time()
		utils.update_learning_rate(optimizer, decay_rate = 0.999, lowest = args.lr / 10)

		wait_until_kl_inc = 10
		if itr // num_batches < wait_until_kl_inc:
			kl_coef = 0.
		else:
			kl_coef = (1-0.99** (itr // num_batches - wait_until_kl_inc))

		# model.z0_diffeq_solver.ode_func.nfe = 0
		optimizer.zero_grad()
		batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
		train_res = model.compute_all_losses(batch_dict, n_traj_samples = 3, kl_coef = kl_coef)
		train_nfe = model.z0_diffeq_solver.ode_func.nfe
		model.z0_diffeq_solver.ode_func.nfe = 0

		# model.z0_diffeq_solver.ode_func.nfe = 0
		train_res["loss"].backward()
		optimizer.step()
		train_nbe = model.z0_diffeq_solver.ode_func.nfe
		model.z0_diffeq_solver.ode_func.nfe = 0
		train_end_time = time.time()
		train_time = train_start_time - train_end_time

		n_iters_to_viz = 1
		if itr % (n_iters_to_viz * num_batches) == 0:
			with torch.no_grad():
				# model.z0_diffeq_solver.ode_func.nfe = 0
				test_start_time = time.time()
				test_res = compute_loss_all_batches(model, 
					data_obj["test_dataloader"], args,
					n_batches = data_obj["n_test_batches"],
					experimentID = experimentID,
					device = device,
					n_traj_samples = 3, kl_coef = kl_coef)
				test_nfe = model.z0_diffeq_solver.ode_func.nfe
				model.z0_diffeq_solver.ode_func.nfe = 0
				test_end_time = time.time()
				test_time = test_start_time - test_end_time
    
				logger.info("Train NFE: {}".format(train_nfe))
				logger.info("Train NBE: {}".format(train_nbe))
				logger.info("Test NFE: {}".format(test_nfe))
				logger.info("Train time: {}".format(train_time))
				logger.info("Test time: {}".format(test_time))	

				message = 'Epoch {:04d}/{:04d} [Test seq (cond on sampled tp)] | Loss {:.6f} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
					itr//num_batches, 
					args.niters,
					test_res["loss"].detach(), test_res["likelihood"].detach(), 
					test_res["kl_first_p"], test_res["std_first_p"])
		 	
				logger.info("Experiment " + str(experimentID))
				logger.info(message)
				logger.info("KL coef: {}".format(kl_coef))
				logger.info("Train loss (one batch): {}".format(train_res["loss"].detach()))
				logger.info("Train CE loss (one batch): {}".format(train_res["ce_loss"].detach()))

				rec_names = ["epoch", 
							"test_loss", 
							"test_likelihood", 
							"test_kl_first_p", 
							"test_std_first_p", 
							"train_kl_coef",
							"train_loss",
							"train_ce_loss",
       						"train_nfe",
							"train_nbe",
       						"test_nfe",
             				"train_time",
                 			"test_time"]
				printouts = [itr//num_batches, 
							test_res["loss"].detach().item(), 
							test_res["likelihood"].detach().item(), 
							test_res["kl_first_p"], 
							test_res["std_first_p"],
							kl_coef,
							train_res["loss"].item(),
							train_res["ce_loss"].item(),
       						train_nfe,
             				train_nbe,
                 			test_nfe,
                    		train_time,
                      		test_time]
				
				if "auc" in test_res:
					logger.info("Classification AUC (TEST): {:.4f}".format(test_res["auc"]))
					rec_names.append("test_auc")
					printouts.append(test_res["auc"].item())

				if "mse" in test_res:
					logger.info("Test MSE: {:.4f}".format(test_res["mse"]))
					rec_names.append("test_mse")
					printouts.append(test_res["mse"].item())

				if "accuracy" in train_res:
					logger.info("Classification accuracy (TRAIN): {:.4f}".format(train_res["accuracy"]))
					rec_names.append("train_acc")
					printouts.append(train_res["accuracy"].item())

				if "accuracy" in test_res:
					logger.info("Classification accuracy (TEST): {:.4f}".format(test_res["accuracy"]))
					rec_names.append("test_acc")
					printouts.append(test_res["accuracy"])

				if "pois_likelihood" in test_res:
					logger.info("Poisson likelihood: {}".format(test_res["pois_likelihood"]))
					rec_names.append("test_pois_likelihood")
					printouts.append(test_res["pois_likelihood"].item())

				if "ce_loss" in test_res:
					logger.info("CE loss: {}".format(test_res["ce_loss"]))
					rec_names.append("test_ce_loss")
					printouts.append(test_res["ce_loss"].item())
     
				max_cuda_memory_allocated = torch.cuda.max_memory_allocated(device=device)
				print("Max memory allocated:", max_cuda_memory_allocated)
				logger.info("Max memory allocated: {}".format(max_cuda_memory_allocated)) 
    
				csvfile = open(csv_path, 'a')
				writer = csv.writer(csvfile)
				writer.writerow(printouts)
				csvfile.close()

			torch.save({
				'args': args,
				'state_dict': model.state_dict(),
			}, ckpt_path)

			# Plotting
			if args.viz:
				with torch.no_grad():
					test_dict = utils.get_next_batch(data_obj["test_dataloader"])

					print("plotting....")
					if isinstance(model, LatentODE) and (args.dataset == "periodic"): #and not args.classic_rnn and not args.ode_rnn:
						plot_id = itr // num_batches // n_iters_to_viz
						viz.draw_all_plots_one_dim(test_dict, model, 
							plot_name = file_name + "_" + str(experimentID) + "_{:03d}".format(plot_id) + ".png",
						 	experimentID = experimentID, save=True)
						plt.pause(0.01)
	torch.save({
		'args': args,
		'state_dict': model.state_dict(),
	}, ckpt_path)

