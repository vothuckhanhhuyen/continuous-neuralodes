###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.spectral_norm import spectral_norm
from einops import rearrange

import lib.utils as utils

#####################################################################################################

class ODEFunc(nn.Module):
	def __init__(self, input_dim, latent_dim, ode_func_net, node="NODE", alpha=1e2, alphaf=True, actv=nn.Tanh(), corr=-100, corrf=True, corr_m=-100, corr_mf=True, nesterov_factor=3, device = torch.device("cpu")):
		"""
		input_dim: dimensionality of the input
		latent_dim: dimensionality used for ODE. Analog of a continous latent state
		"""
		super(ODEFunc, self).__init__()

		self.input_dim = input_dim
		self.device = device

		utils.init_network_weights(ode_func_net)
		self.gradient_net = ode_func_net

		self.node = node
		self.alpha = Parameter([alpha], frozen=alphaf)
		self.actv = actv
		self.nesterov_factor = nesterov_factor

		self.sp = nn.Softplus()
		self.corr = Parameter([corr], frozen=corrf)
		self.corr_m = Parameter([corr_m], frozen=corr_mf)
  
		self.nfe = 0

	def forward(self, t_local, y):
		"""
		Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

		t_local: current time point
		y: value at the current time point
		"""
		self.nfe += 1
		if self.node == 'HBNODE':
			theta, m = torch.split(y, 1, dim=2)
			f = self.get_ode_gradient_nn(t_local, theta)
			dtheta = m
			dm = - m + f
			out = torch.cat((dtheta, dm), dim=2)
		elif self.node == 'GHBNODE':
			theta, m = torch.split(y, 1, dim=2)
			f = self.get_ode_gradient_nn(t_local, theta)
			dtheta = self.actv(m)
			dm = - 0.99 * m + f - theta
			out = torch.cat((dtheta, dm), dim=2)
		elif self.node == 'NesterovNODE':
			theta, m = torch.split(y, 1, dim=2)
			f = self.get_ode_gradient_nn(t_local, theta)
			dtheta = self.actv(m)
			dm = -3/t_local * m + f
			out = torch.cat((dtheta, dm), dim=2)
		elif self.node == 'GNesterovNODE':
			h, dh = torch.split(y, 1, dim=2)
			f = self.get_ode_gradient_nn(t_local, h)
			t = torch.Tensor([t_local])
			if h.is_cuda:
				t = t.to(self.device)
			k_reciprocal = 1 / (self.actv(torch.pow(t, -self.nesterov_factor/2) * torch.exp(t/2)))
			m = (self.nesterov_factor/2 * (1/t) * k_reciprocal - 1/2 * k_reciprocal) * h \
					+ k_reciprocal * dh
			dtheta = self.actv(m)
			dm = f - m - self.sp(self.corr()) * h
			out = torch.cat((dtheta, dm), dim=2)
		elif self.node == 'RMSpropNODE':
			theta, m = torch.split(y, 1, dim=2)
			t_tensor = torch.Tensor([t_local])
			if theta.is_cuda:
				t_tensor = t_tensor.to(self.device)
			m = nn.Tanh()(torch.abs(m)) # *
			gtheta = self.get_ode_gradient_nn(t_local, theta) * self.get_ode_gradient_nn(t_local, m)
			dtheta = -gtheta
			dm = 1 / self.alpha() * (gtheta ** 2 - 1) * m
			out = torch.cat((dtheta, dm), dim=2)
		elif self.node == 'GRMSpropNODE':
			theta, m = torch.split(y, 1, dim=2)
			t_tensor = torch.Tensor([t_local])
			if theta.is_cuda:
				t_tensor = t_tensor.to(self.device)
			m = nn.Tanh()(torch.abs(m)) # *
			gtheta = self.get_ode_gradient_nn(t_local, theta) * self.get_ode_gradient_nn(t_local, m)			
			dtheta = -self.actv(gtheta)
			dm = 1 / self.alpha() * (self.actv(gtheta ** 2) - 1) * m - self.sp(self.corr()) * theta
			out = torch.cat((dtheta, dm), dim=2)
		elif self.node == "TOAES_NODE":
			h, p, q = torch.split(y, 1, dim=2)
			if self.learnable_mu:
				self.mu = self.mu_net(y)
				f = self.get_ode_gradient_nn(t_local, h + 1/torch.sqrt(self.mu) * p)
				dh = p
				dp = q
				dq = -3 * torch.sqrt(self.mu) * q - 2 * self.mu * p - torch.sqrt(self.mu) * f
				out = torch.cat((dh, dp, dq), dim=2)
			
			else:
				f = self.get_ode_gradient_nn(t_local, h + 1/np.sqrt(self.mu) * p)
				dh = p
				dp = q
				dq = -3 * np.sqrt(self.mu) * q - 2 * self.mu * p - np.sqrt(self.mu) * f
				out = torch.cat((dh, dp, dq), dim=2)
		elif self.node == "GTOAES_NODE":
			h, p, q = torch.split(y, 1, dim=2)
			if self.learnable_mu:
				self.mu = self.mu_net(y)
				# self.mu = self.mu_param
				f = self.get_ode_gradient_nn(t_local, h + 1/torch.sqrt(self.mu) * p)
				dh = self.actv(p)
				dp = self.actv(q)
				dq = -3 * torch.sqrt(self.mu) * q - 2 * self.mu * p - torch.sqrt(self.mu) * self.actv(f) - self.sp(self.corr()) * h
				out = torch.cat((dh, dp, dq), dim=2)
			
			else:
				f = self.get_ode_gradient_nn(t_local, h + 1/np.sqrt(self.mu) * p)
				dh = self.actv(p)
				dp = self.actv(q)
				dq = -3 * np.sqrt(self.mu) * q - 2 * self.mu * p - np.sqrt(self.mu) * self.actv(f) - self.sp(self.corr()) * h
				out = torch.cat((dh, dp, dq), dim=2)
		else:
			out = self.get_ode_gradient_nn(t_local, y)
		return out


	def get_ode_gradient_nn(self, t_local, y, first=True):
		return self.gradient_net(y)

	def sample_next_point_from_prior(self, t_local, y):
		"""
		t_local: current time point
		y: value at the current time point
		"""
		return self.get_ode_gradient_nn(t_local, y)

#####################################################################################################

class ODEFunc_w_Poisson(ODEFunc):
	
	def __init__(self, input_dim, latent_dim, ode_func_net,
		lambda_net, device = torch.device("cpu")):
		"""
		input_dim: dimensionality of the input
		latent_dim: dimensionality used for ODE. Analog of a continous latent state
		"""
		super(ODEFunc_w_Poisson, self).__init__(input_dim, latent_dim, ode_func_net, device)

		self.latent_ode = ODEFunc(input_dim = input_dim, 
			latent_dim = latent_dim, 
			ode_func_net = ode_func_net,
			device = device)

		self.latent_dim = latent_dim
		self.lambda_net = lambda_net
		# The computation of poisson likelihood can become numerically unstable. 
		#The integral lambda(t) dt can take large values. In fact, it is equal to the expected number of events on the interval [0,T]
		#Exponent of lambda can also take large values
		# So we divide lambda by the constant and then multiply the integral of lambda by the constant
		self.const_for_lambda = torch.Tensor([100.]).to(device)

	def extract_poisson_rate(self, augmented, final_result = True):
		y, log_lambdas, int_lambda = None, None, None

		assert(augmented.size(-1) == self.latent_dim + self.input_dim)		
		latent_lam_dim = self.latent_dim // 2

		if len(augmented.size()) == 3:
			int_lambda  = augmented[:,:,-self.input_dim:] 
			y_latent_lam = augmented[:,:,:-self.input_dim]

			log_lambdas  = self.lambda_net(y_latent_lam[:,:,-latent_lam_dim:])
			y = y_latent_lam[:,:,:-latent_lam_dim]

		elif len(augmented.size()) == 4:
			int_lambda  = augmented[:,:,:,-self.input_dim:]
			y_latent_lam = augmented[:,:,:,:-self.input_dim]

			log_lambdas  = self.lambda_net(y_latent_lam[:,:,:,-latent_lam_dim:])
			y = y_latent_lam[:,:,:,:-latent_lam_dim]

		# Multiply the intergral over lambda by a constant 
		# only when we have finished the integral computation (i.e. this is not a call in get_ode_gradient_nn)
		if final_result:
			int_lambda = int_lambda * self.const_for_lambda
			
		# Latents for performing reconstruction (y) have the same size as latent poisson rate (log_lambdas)
		assert(y.size(-1) == latent_lam_dim)

		return y, log_lambdas, int_lambda, y_latent_lam


	def get_ode_gradient_nn(self, t_local, augmented):
		y, log_lam, int_lambda, y_latent_lam = self.extract_poisson_rate(augmented, final_result = False)
		dydt_dldt = self.latent_ode(t_local, y_latent_lam)

		log_lam = log_lam - torch.log(self.const_for_lambda)
		return torch.cat((dydt_dldt, torch.exp(log_lam)),-1)

class Parameter(nn.Module):
    def __init__(self, val, frozen=False):
        super().__init__()
        val = torch.Tensor(val)
        self.val = val
        self.param = nn.Parameter(val)
        self.frozen = frozen

    def forward(self):
        if self.frozen:
            self.val = self.val.to(self.param.device)
            return self.val
        else:
            return self.param

    def freeze(self):
        self.val = self.param.detach().clone()
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    def __repr__(self):
        return "val: {}, param: {}".format(self.val.cpu(), self.param.detach().cpu())



