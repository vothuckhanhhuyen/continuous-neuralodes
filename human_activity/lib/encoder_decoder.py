###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

# from ast import type_ignore
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu
import lib.utils as utils
from torch.distributions import Categorical, Normal
import lib.utils as utils
from torch.nn.modules.rnn import LSTM, GRU
from lib.utils import get_device


# GRU description: 
# http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/
class GRU_unit(nn.Module):
	def __init__(self, latent_dim, input_dim, 
		update_gate = None,
		reset_gate = None,
		new_state_net = None,
		n_units = 100,
		device = torch.device("cpu"),
		node = "NODE"):
		super(GRU_unit, self).__init__()

		self.node = node

		if update_gate is None:
			if self.node in ('HBNODE', 'GHBNODE', 'NesterovNODE', 'GNesterovNODE', "RMSpropNODE", "GRMSpropNODE"):	
				self.update_gate = nn.Sequential(
					nn.Linear(latent_dim * 4 + input_dim, n_units),
					nn.Tanh(),
					nn.Linear(n_units, latent_dim * 2),
					nn.Sigmoid())
				utils.init_network_weights(self.update_gate)
			elif self.node in ("TOAES_NODE", "GTOAES_NODE"):
				self.update_gate = nn.Sequential(
                    nn.Linear(latent_dim * 6 + input_dim, n_units),
                    nn.Tanh(),
                    nn.Linear(n_units, latent_dim * 3),
                    nn.Sigmoid())
				utils.init_network_weights(self.update_gate)
			else:
				self.update_gate = nn.Sequential(
					nn.Linear(latent_dim * 2 + input_dim, n_units),
					nn.Tanh(),
					nn.Linear(n_units, latent_dim),
					nn.Sigmoid())
				utils.init_network_weights(self.update_gate)
		else: 
			self.update_gate  = update_gate

		if reset_gate is None:
			if self.node in ('HBNODE', 'GHBNODE', 'NesterovNODE', 'GNesterovNODE', "RMSpropNODE", "GRMSpropNODE"):	
				self.reset_gate = nn.Sequential(
					nn.Linear(latent_dim * 4 + input_dim, n_units),
					nn.Tanh(),
					nn.Linear(n_units, latent_dim * 2),
					nn.Sigmoid())
				utils.init_network_weights(self.reset_gate)
			elif self.node in ("TOAES_NODE", "GTOAES_NODE"):
				self.reset_gate = nn.Sequential(
                    nn.Linear(latent_dim * 6 + input_dim, n_units),
                    nn.Tanh(),
                    nn.Linear(n_units, latent_dim * 3),
                    nn.Sigmoid())
				utils.init_network_weights(self.reset_gate)
			else:
				self.reset_gate = nn.Sequential(
					nn.Linear(latent_dim * 2 + input_dim, n_units),
					nn.Tanh(),
					nn.Linear(n_units, latent_dim),
					nn.Sigmoid())
				utils.init_network_weights(self.reset_gate)
		else: 
			self.reset_gate  = reset_gate

		if new_state_net is None:
			if self.node in ('HBNODE', 'GHBNODE', 'NesterovNODE', 'GNesterovNODE', "RMSpropNODE", "GRMSpropNODE"):	
				self.new_state_net = nn.Sequential(
					nn.Linear(latent_dim * 4 + input_dim, n_units),
					nn.Tanh(),
					nn.Linear(n_units, latent_dim * 4))
				utils.init_network_weights(self.new_state_net)
			elif self.node in ("TOAES_NODE", "GTOAES_NODE"):
				self.new_state_net = nn.Sequential(
                    nn.Linear(latent_dim * 6 + input_dim, n_units),
                    nn.Tanh(),
                    nn.Linear(n_units, latent_dim * 6))
				utils.init_network_weights(self.new_state_net)
			else:
				self.new_state_net = nn.Sequential(
					nn.Linear(latent_dim * 2 + input_dim, n_units),
					nn.Tanh(),
					nn.Linear(n_units, latent_dim * 2))
				utils.init_network_weights(self.new_state_net)
		else: 
			self.new_state_net  = new_state_net

	def forward(self, y_mean, y_std, x, masked_update = True):
		if self.node in ('HBNODE', 'GHBNODE', 'NesterovNODE', 'GNesterovNODE', "RMSpropNODE", "GRMSpropNODE"):	
			y_mean = torch.cat([y_mean[:, :, 0, :], y_mean[:, :, 1, :]], -1)
			y_std = torch.cat([y_std[:, :, 0, :], y_std[:, :, 1, :]], -1)
		if self.node in ("TOAES_NODE", "GTOAES_NODE"):
			y_mean, y_std = rearrange(y_mean, 'b c x y -> b c (x y)'), rearrange(y_std, 'b c x y -> b c (x y)')
		y_concat = torch.cat([y_mean, y_std, x], -1)

		update_gate = self.update_gate(y_concat)
		reset_gate = self.reset_gate(y_concat)
		concat = torch.cat([y_mean * reset_gate, y_std * reset_gate, x], -1)
		
		if self.node in ("TOAES_NODE", "GTOAES_NODE"):
            # moi new_state phai la 45
			new_state, new_state_std = utils.split_last_dim(self.new_state_net(concat))
		else:   
			new_state, new_state_std = utils.split_last_dim(self.new_state_net(concat))
		new_state_std = new_state_std.abs()

		new_y = (1-update_gate) * new_state + update_gate * y_mean
		new_y_std = (1-update_gate) * new_state_std + update_gate * y_std

		assert(not torch.isnan(new_y).any())

		if masked_update:
			# IMPORTANT: assumes that x contains both data and mask
			# update only the hidden states for hidden state only if at least one feature is present for the current time point
			n_data_dims = x.size(-1)//2
			mask = x[:, :, n_data_dims:]
			utils.check_mask(x[:, :, :n_data_dims], mask)
			
			mask = (torch.sum(mask, -1, keepdim = True) > 0).float()

			assert(not torch.isnan(mask).any())

			new_y = mask * new_y + (1-mask) * y_mean
			new_y_std = mask * new_y_std + (1-mask) * y_std

			if torch.isnan(new_y).any():
				print("new_y is nan!")
				print(mask)
				print(y_mean)
				# print(prev_new_y)
				exit()
    
		if self.node in ('HBNODE', 'GHBNODE', 'NesterovNODE', 'GNesterovNODE', "RMSpropNODE", "GRMSpropNODE"):
			new_y_h, new_y_dh = utils.split_last_dim(new_y)
			new_y_std_h, new_y_std_dh = utils.split_last_dim(new_y_std)
			new_y = torch.stack((new_y_h, new_y_dh), dim=2)
			new_y_std = torch.stack((new_y_std_h, new_y_std_dh), dim=2)
		if self.node in ("TOAES_NODE", "GTOAES_NODE"):
			new_y_h, new_y_dh, new_y_ddh = utils.split_last_dim_3(new_y)
			new_y_std_h, new_y_std_dh, new_y_std_ddh = utils.split_last_dim_3(new_y_std)
			new_y = torch.stack((new_y_h, new_y_dh, new_y_ddh), dim=2)
			new_y_std = torch.stack((new_y_std_h, new_y_std_dh, new_y_std_ddh), dim=2)
		new_y_std = new_y_std.abs()
		return new_y, new_y_std



class Encoder_z0_RNN(nn.Module):
	def __init__(self, latent_dim, input_dim, lstm_output_size = 20, 
		use_delta_t = True, device = torch.device("cpu")):
		
		super(Encoder_z0_RNN, self).__init__()
	
		self.gru_rnn_output_size = lstm_output_size
		self.latent_dim = latent_dim
		self.input_dim = input_dim
		self.device = device
		self.use_delta_t = use_delta_t

		self.hiddens_to_z0 = nn.Sequential(
		   nn.Linear(self.gru_rnn_output_size, 50),
		   nn.Tanh(),
		   nn.Linear(50, latent_dim * 2),)

		utils.init_network_weights(self.hiddens_to_z0)

		input_dim = self.input_dim

		if use_delta_t:
			self.input_dim += 1
		self.gru_rnn = GRU(self.input_dim, self.gru_rnn_output_size).to(device)

	def forward(self, data, time_steps, run_backwards = True):
		# IMPORTANT: assumes that 'data' already has mask concatenated to it 

		# data shape: [n_traj, n_tp, n_dims]
		# shape required for rnn: (seq_len, batch, input_size)
		# t0: not used here
		n_traj = data.size(0)

		assert(not torch.isnan(data).any())
		assert(not torch.isnan(time_steps).any())

		data = data.permute(1,0,2) 

		if run_backwards:
			# Look at data in the reverse order: from later points to the first
			data = utils.reverse(data)

		if self.use_delta_t:
			delta_t = time_steps[1:] - time_steps[:-1]
			if run_backwards:
				# we are going backwards in time with
				delta_t = utils.reverse(delta_t)
			# append zero delta t in the end
			delta_t = torch.cat((delta_t, torch.zeros(1).to(self.device)))
			delta_t = delta_t.unsqueeze(1).repeat((1,n_traj)).unsqueeze(-1)
			data = torch.cat((delta_t, data),-1)

		outputs, _ = self.gru_rnn(data)

		# LSTM output shape: (seq_len, batch, num_directions * hidden_size)
		last_output = outputs[-1]

		self.extra_info ={"rnn_outputs": outputs, "time_points": time_steps}

		mean, std = utils.split_last_dim(self.hiddens_to_z0(last_output))
		std = std.abs()

		assert(not torch.isnan(mean).any())
		assert(not torch.isnan(std).any())

		return mean.unsqueeze(0), std.unsqueeze(0)



class Encoder_z0_ODE_RNN(nn.Module):
	# Derive z0 by running ode backwards.
	# For every y_i we have two versions: encoded from data and derived from ODE by running it backwards from t_i+1 to t_i
	# Compute a weighted sum of y_i from data and y_i from ode. Use weighted y_i as an initial value for ODE runing from t_i to t_i-1
	# Continue until we get to z0
	def __init__(self, latent_dim, input_dim, z0_diffeq_solver = None, 
		z0_dim = None, GRU_update = None, 
		n_gru_units = 100, 
		device = torch.device("cpu"),
		node="NODE",
		nesterov_factor=3,
		activation_h=nn.Tanh()):

		super(Encoder_z0_ODE_RNN, self).__init__()

		self.node = node

		if z0_dim is None:
			self.z0_dim = latent_dim
		else:
			self.z0_dim = z0_dim

		if GRU_update is None:
			self.GRU_update = GRU_unit(latent_dim, input_dim, 
				n_units = n_gru_units, 
				device=device, node=self.node).to(device)
		else:
			self.GRU_update = GRU_update

		self.z0_diffeq_solver = z0_diffeq_solver
		self.latent_dim = latent_dim
		self.input_dim = input_dim
		self.device = device
		self.extra_info = None

		self.transform_z0 = nn.Sequential(
			nn.Linear(latent_dim * 2, 100),
			nn.Tanh(),
			nn.Linear(100, self.z0_dim * 2),)
		utils.init_network_weights(self.transform_z0)

		self.nesterov_factor = nesterov_factor
		self.activation_h = nn.Identity() if activation_h is None else activation_h


	def forward(self, data, time_steps, run_backwards = True, save_info = False, start_time=0):
		# data, time_steps -- observations and their time stamps
		# IMPORTANT: assumes that 'data' already has mask concatenated to it 
		assert(not torch.isnan(data).any())
		assert(not torch.isnan(time_steps).any())
  
		# time_steps = torch.add(time_steps, start_time)

		n_traj, n_tp, n_dims = data.size()
		if len(time_steps) == 1:
			prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)
			prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)

			xi = data[:,0,:].unsqueeze(0)

			last_yi, last_yi_std = self.GRU_update(prev_y, prev_std, xi)
			extra_info = None
		else:
			last_yi, last_yi_std, _, extra_info = self.run_odernn(
				data, time_steps, run_backwards = run_backwards,
				save_info = save_info, start_time=start_time)

		if self.node in ('HBNODE', 'GHBNODE', 'NesterovNODE', 'GNesterovNODE', "RMSpropNODE", "GRMSpropNODE"):	
			means_z0 = last_yi.reshape(1, n_traj, 2, self.latent_dim)
			std_z0 = last_yi_std.reshape(1, n_traj, 2, self.latent_dim)
		else:
			means_z0 = last_yi.reshape(1, n_traj, self.latent_dim)
			std_z0 = last_yi_std.reshape(1, n_traj, self.latent_dim)

		means_z0, std_z0 = utils.split_last_dim(self.transform_z0(torch.cat((means_z0, std_z0), -1)))
		std_z0 = std_z0.abs()
		if save_info:
			self.extra_info = extra_info

		return means_z0, std_z0


	def run_odernn(self, data, time_steps, run_backwards = True, save_info = False, start_time=0):
		# IMPORTANT: assumes that 'data' already has mask concatenated to it 

		# time_steps = np.add(time_steps, start_time)
		time_steps = torch.add(time_steps, start_time)

		n_traj, n_tp, n_dims = data.size()
		extra_info = []

		t0 = time_steps[-1]
		if run_backwards:
			t0 = time_steps[0]

		device = get_device(data)

		if self.node in ('HBNODE', 'GHBNODE', 'NesterovNODE', 'GNesterovNODE', "RMSpropNODE", "GRMSpropNODE"):	
			prev_y = torch.zeros((1, n_traj, 2, self.latent_dim)).to(device)
			prev_std = torch.zeros((1, n_traj, 2, self.latent_dim)).to(device)
		elif self.node in ("TOAES_NODE", "GTOAES_NODE"):
			prev_y = torch.zeros((1, n_traj, 3, self.latent_dim)).to(device)
			prev_std = torch.zeros((1, n_traj, 3, self.latent_dim)).to(device)
		else:
			prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(device)
			prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(device)

		if self.node == 'GNesterovNODE': 
			nesterov_algebraic = True
			diff_from_alg = False
		else:
			nesterov_algebraic = False
			diff_from_alg = False
   
		if self.node in ("RMSpropNODE", "GRMSpropNODE"):
			abs_second = True
		else:
			abs_second = False

		prev_t, t_i = time_steps[-1] + 0.01,  time_steps[-1]

		interval_length = time_steps[-1] - time_steps[0]
		minimum_step = interval_length / 50

		assert(not torch.isnan(data).any())
		assert(not torch.isnan(time_steps).any())

		latent_ys = []
		# Run ODE backwards and combine the y(t) estimates using gating
		time_points_iter = range(0, len(time_steps))
		if run_backwards:
			time_points_iter = reversed(time_points_iter)

		if diff_from_alg:
			prev_y_x, prev_y_m = self.calc_differential_from_algebraic(prev_y, prev_t)
			prev_y = torch.cat((prev_y_x, prev_y_m), dim=2)
		if abs_second:
			prev_y_x, prev_y_m = torch.split(prev_y, 1, dim=2)
			prev_y_m = torch.abs(prev_y_m)
			prev_y = torch.cat((prev_y_x, prev_y_m), dim=2)

		for i in time_points_iter:
			if (prev_t - t_i) < minimum_step:
				time_points = torch.stack((prev_t, t_i))
				inc = self.z0_diffeq_solver.ode_func(prev_t, prev_y) * (t_i - prev_t)
				assert(not torch.isnan(inc).any())
				ode_sol = prev_y + inc
				ode_sol = torch.stack((prev_y, ode_sol), 2).to(device)
				assert(not torch.isnan(ode_sol).any())
			else:
				n_intermediate_tp = max(2, ((prev_t - t_i) / minimum_step).int())

				time_points = utils.linspace_vector(prev_t, t_i, n_intermediate_tp)
				ode_sol = self.z0_diffeq_solver(prev_y, time_points)
				assert(not torch.isnan(ode_sol).any())

			if torch.mean(ode_sol[:, :, 0, :]  - prev_y) >= 0.001:
				print("Error: first point of the ODE is not equal to initial value")
				print(torch.mean(ode_sol[:, :, 0, :]  - prev_y))
				exit()
			#assert(torch.mean(ode_sol[:,:,0,:]  - prev_y) < 0.001)

			yi_ode = ode_sol[:, :, -1, :]

			if nesterov_algebraic:
				yi_ode = self.calc_algebraic_factor(yi_ode, time_points)

			xi = data[:, i, :].unsqueeze(0)
			
			yi, yi_std = self.GRU_update(yi_ode, prev_std, xi) #

			prev_y, prev_std = yi, yi_std			
			prev_t, t_i = time_steps[i], time_steps[i - 1]

			if self.node in ('HBNODE', 'GHBNODE', 'NesterovNODE', 'GNesterovNODE', "RMSpropNODE", "GRMSpropNODE", "TOAES_NODE", "GTOAES_NODE"):	
				latent_ys.append(yi[:, :, 0, :])
			else:
				latent_ys.append(yi)

			if save_info:
				d = {"yi_ode": yi_ode.detach(), #"yi_from_data": yi_from_data,
					"yi": yi.detach(), "yi_std": yi_std.detach(), 
					"time_points": time_points.detach(), "ode_sol": ode_sol.detach()}
				extra_info.append(d)

		latent_ys = torch.stack(latent_ys, 1)

		assert(not torch.isnan(yi).any())
		assert(not torch.isnan(yi_std).any())

		return yi, yi_std, latent_ys, extra_info

	def calc_algebraic_factor(self, z, time_points):
		# print(time_points.size())
		# split the input into the starting time step and the other time steps
		z_0 = z[:1]
		z_T = z[1:] 
		# get the corresponding value of t for the other time steps
		if len(time_points.size()) == 2:
			T = time_points[:, -1]
		else:
			T = time_points[1:]
		x, m = torch.split(z_T, 1, dim=2)
		# T^(-3/2) * e^(T/2)
		k = torch.pow(T, -self.nesterov_factor/2) * torch.exp(T / 2)
		if z.is_cuda and not T.is_cuda:
			k = k.to(device=self.device)
			T = T.to(device=self.device)
		k = self.activation_h(k)
		# h(T) = [x(T) m(T)] * Transpose([T^(-3/2)*e^(T/2) I])
		h = bmul(k, x)
		dh = bmul(k, m - bmul(self.nesterov_factor/2 * torch.pow(T, 1/2) * torch.exp(-T/2) - 1/2 * 1/k, h))
		z_t = torch.cat((h, dh), dim=2)
		out = torch.cat((z_0, z_t), dim=0)
		return out

	def calc_differential_from_algebraic(self, z, t):
		h, dh = torch.split(z, 1, dim=2)
		k_reciprocal = 1 / (torch.pow(t, -self.nesterov_factor/2) * torch.exp(t/2))
		if z.is_cuda:
			k_reciprocal = k_reciprocal.to(z.get_device())
		m = (self.nesterov_factor/2 * (1/t) * k_reciprocal - 1/2 * k_reciprocal) * h \
				+ k_reciprocal * dh
		x = h * k_reciprocal
		return x, m


class Decoder(nn.Module):
	def __init__(self, latent_dim, input_dim):
		super(Decoder, self).__init__()
		# decode data from latent space where we are solving an ODE back to the data space

		decoder = nn.Sequential(
		   nn.Linear(latent_dim, input_dim),)

		utils.init_network_weights(decoder)	
		self.decoder = decoder

	def forward(self, data):
		return self.decoder(data)

def bmul(vec, mat, axis=0):
    mat = mat.transpose(axis, -1)
    return (mat * vec.expand_as(mat)).transpose(axis, -1)


