import torch
from torch import nn

from torchdiffeq import odeint_adjoint as odeint


class initial_velocity(nn.Module):
    def __init__(self, dim, nhidden):
        super(initial_velocity, self).__init__()
        self.tanh = nn.Hardtanh(min_val=-5.0, max_val=5.0, inplace=False)
        self.fc1 = nn.Linear(dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, dim)

    def forward(self, x0):
        out = self.fc1(x0)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        return torch.cat((x0, out))


class ODEBlock(nn.Module):

    def __init__(self, odefunc, t0_, tN_, tol, half=False, algebraic_from_differential=False, differential_from_algebraic=False, actv_k=None, use_momentum=False, actv_output=None):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_times = torch.tensor([t0_, tN_]).float()
        self.tol = tol
        self.algebraic_from_differential = algebraic_from_differential
        self.differential_from_algebraic = differential_from_algebraic
        self.nesterov_factor = 3
        self.half = half
        self.actv_k = nn.Identity() if actv_k is None else actv_k
        self.actv_output = nn.Identity() if actv_output is None else actv_output
        self.verbose = False
        self.use_momentum = use_momentum

    def forward(self, x0):
        solver = 'dopri5'
        if self.differential_from_algebraic:
            x0 = self.calc_differential_from_algebraic(x0, self.integration_times[0])
        out = odeint(self.odefunc, x0, self.integration_times,
                     rtol=self.tol, atol=self.tol, method=solver)
        if self.verbose:
            print("out ODEBlock:", out)
        if self.algebraic_from_differential:
            out = self.calc_algebraic_from_differential(out)
            if self.verbose:
                print("out ODEBlock after algebraic:", out)
        out = out[1]
        # used in 1st order system to take only the position but not the momentum
        if self.half:
            mid = int(len(x0)/2)
            h = out[:mid]
            # if you decide to use the momentum
            if self.use_momentum:
                dh = out[mid:]
                out = torch.cat((h, dh), dim=1)
            else:
                out = h
        return out

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

    def calc_algebraic_from_differential(self, z):
        if self.verbose:
            print("calculating algebraic_factor!")
        # split the input into the starting time step and the other time steps
        z_0 = z[:1]
        z_T = z[1:]
        T = self.integration_times[-1]
        assert T.requires_grad == False
        assert z_T.shape[1] % 2 == 0
        mid = z_T.shape[1] // 2
        x, m = torch.split(z_T, mid, dim=1)
        # T^(-3/2) * e^(T/2)
        k = torch.pow(T, -3/2) * torch.exp(T / 2)
        k = self.actv_k(k)
        if self.verbose:
            print("k:", k)
            print("T:", T)
        # h(T) = [x(T) m(T)] * Transpose([T^(-3/2)*e^(T/2) I])
        h = self.actv_output(x * k)
        dh = self.actv_output(k * (m - (3/2 * torch.pow(T, 1/2) * torch.exp(-T/2) - 1/2 * 1/k) * h))
        if self.verbose:
            print("h:", h)
            print("dh:", dh)
        z_t = torch.cat((h, dh), dim=1)
        out = torch.cat((z_0, z_t), dim=0)
        return out

    def calc_differential_from_algebraic(self, z, t):
        h, dh = torch.split(z, 1, dim=1)
        k_reciprocal = 1 / (torch.pow(t, -self.nesterov_factor/2) * torch.exp(t/2))
        if z.is_cuda:
            k_reciprocal = k_reciprocal.to(z.get_device())
        m = (self.nesterov_factor/2 * (1/t) * k_reciprocal - 1/2 * k_reciprocal) * h \
                + k_reciprocal * dh
        x = h * k_reciprocal
        return torch.cat([x, m], dim=1)


class Decoder(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(Decoder, self).__init__()
        self.tanh = nn.Hardtanh(min_val=-1.0, max_val=1.0, inplace=False)
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, z):
        out = self.fc(z)
        out = self.tanh(out)
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
