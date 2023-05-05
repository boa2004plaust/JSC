import torch
from torch import nn
import torch.nn.functional as F


class nn_bn_relu(nn.Module):
    def __init__(self, nin, nout):
        super(nn_bn_relu, self).__init__()
        self.linear = nn.Linear(nin, nout)
        self.relu = nn.ReLU(False)

    def forward(self, x, relu=True):
        if relu:
            return self.relu(self.linear(x))
        return self.linear(x)
        # return self.bn(self.linear(x))
        

class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128, n=1):
        super(Embed, self).__init__()

        self.n = n
        if n == 1:
            self.linear = nn.Linear(dim_in, dim_out)
        else:
            # SimCLR claims 2 layer MLP projections work better
            r = 2
            self.l1 = nn_bn_relu(dim_in, dim_in // r)
            self.l2 = nn_bn_relu(dim_in // r, dim_out)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        if self.n == 1:
            x = self.linear(x)
        else:
            x = self.l1(x, relu=True)
            x = self.l2(x, relu=False)

        return x


class ITLoss(nn.Module):
    """Information-Theoretic Loss function
    Args:
        opt.s_dim: the dimension of student's feature
        opt.t_dim: the dimension of teacher's feature
    """
    def __init__(self, s_dim, t_dim, n_data, alpha_it):
        super(ITLoss, self).__init__()
        self.s_dim = s_dim
        self.t_dim = t_dim
        self.n_data = n_data
        self.alpha_it = alpha_it
        self.embed = Embed(self.s_dim, self.t_dim, n=1)

    def forward_correlation_it(self, z_s, z_t):
        f_s = z_s
        f_t = z_t

        f_s = self.embed(f_s)

        n, d = f_s.shape

        f_s_norm = (f_s - f_s.mean(0)) / f_s.std(0)
        f_t_norm = (f_t - f_t.mean(0)) / f_t.std(0) 
        c_st = torch.einsum('bx,bx->x', f_s_norm, f_t_norm) / n
        c_diff = c_st - torch.ones_like(c_st)

        alpha = self.alpha_it
        c_diff = torch.abs(c_diff)
        c_diff = c_diff.pow(2.0)

        c_diff = c_diff.pow(alpha)

        loss = torch.log2(c_diff.sum())

        return loss

    def forward_mutual_it(self, z_s, z_t):
        f_s = z_s
        f_t = z_t

        if self.s_dim != self.t_dim:
            f_s = self.embed(f_s)

        f_s_norm = F.normalize(f_s)
        f_t_norm = F.normalize(f_t)

        # 1. Polynomial kernel
        G_s = torch.einsum('bx,dx->bd', f_s_norm, f_s_norm)
        G_t = torch.einsum('bx,dx->bd', f_t_norm, f_t_norm)
        G_st = G_s * G_t

        # Norm before difference
        z_s = torch.trace(G_s)
        z_st = torch.trace(G_st)

        G_s = G_s / z_s
        G_st = G_st / z_st

        g_diff = G_s.pow(2) - G_st.pow(2)
        loss = g_diff.sum()

        return loss
