import torch
import torch.nn as nn
import numpy as np

import torch.backends.cudnn as cudnn
cudnn.benchmark = True


class Shuffle(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(Shuffle, self).__init__()
        self.in_channels = in_channels
        idx = torch.randperm(in_channels)
        self.register_buffer('forward_shuffle_idx', nn.Parameter(idx, requires_grad=False))
        self.register_buffer('backward_shuffle_idx', nn.Parameter(torch.argsort(idx), requires_grad=False))

    def forward(self, x, reverse=False, conditioning=None):
        if not reverse:
            return x[:, self.forward_shuffle_idx, ...], 0
        else:
            return x[:, self.backward_shuffle_idx, ...]


class BasicFullyConnectedNet(nn.Module):
    def __init__(self, dim, depth, hidden_dim=256, use_tanh=False, use_bn=False, out_dim=None):
        super(BasicFullyConnectedNet, self).__init__()
        layers = []
        layers.append(nn.Linear(dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.LeakyReLU())
        for d in range(depth):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(hidden_dim, dim if out_dim is None else out_dim))
        if use_tanh:
            layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class DoubleVectorCouplingBlock(nn.Module):
    """In contrast to VectorCouplingBlock, this module assures alternating chunking in upper and lower half."""
    def __init__(self, in_channels, hidden_dim, depth=2, use_hidden_bn=False, n_blocks=2):
        super(DoubleVectorCouplingBlock, self).__init__()
        assert in_channels % 2 == 0
        self.s = nn.ModuleList([BasicFullyConnectedNet(dim=in_channels // 2, depth=depth, hidden_dim=hidden_dim,
                                                       use_tanh=True) for _ in range(n_blocks)])
        self.t = nn.ModuleList([BasicFullyConnectedNet(dim=in_channels // 2, depth=depth, hidden_dim=hidden_dim,
                                                       use_tanh=False) for _ in range(n_blocks)])

    def forward(self, x, reverse=False):
        if not reverse:
            logdet = 0
            for i in range(len(self.s)):
                idx_apply, idx_keep = 0, 1
                if i % 2 != 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                scale = self.s[i](x[idx_apply])
                x_ = x[idx_keep] * (scale.exp()) + self.t[i](x[idx_apply])
                x = torch.cat((x[idx_apply], x_), dim=1)
                logdet_ = torch.sum(scale.view(x.size(0), -1), dim=1)
                logdet = logdet + logdet_
            return x, logdet
        else:
            idx_apply, idx_keep = 0, 1
            for i in reversed(range(len(self.s))):
                if i % 2 == 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                x_ = (x[idx_keep] - self.t[i](x[idx_apply])) * (self.s[i](x[idx_apply]).neg().exp())
                x = torch.cat((x[idx_apply], x_), dim=1)
            return x


class VectorActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True, **kwargs):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        if len(input.shape) == 2:
            input = input[:, :, None, None]
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False, conditioning=None):
        if len(input.shape) == 2:
            input = input[:, :, None, None]
        if not reverse:
            _, _, height, width = input.shape
            if self.initialized.item() == 0:
                self.initialize(input)
                self.initialized.fill_(1)
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height * width * torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            if not self.logdet:
                return (self.scale * (input + self.loc)).flatten(start_dim=1)
            return (self.scale * (input + self.loc)).flatten(start_dim=1), logdet
        else:
            return self.reverse(input)

    def reverse(self, output, conditioning=None):
        return (output / self.scale - self.loc).flatten(start_dim=1)


class Flow(nn.Module):
    def __init__(self, module_list, in_channels, hidden_dim, hidden_depth):
        super(Flow, self).__init__()
        self.in_channels = in_channels
        self.flow = nn.ModuleList(
            [module(in_channels, hidden_dim=hidden_dim, depth=hidden_depth) for module in module_list])

    def forward(self, x, condition=None, reverse=False):
        if not reverse:
            logdet = 0
            for i in range(len(self.flow)):
                x, logdet_ = self.flow[i](x)
                logdet = logdet + logdet_
            return x, logdet
        else:
            for i in reversed(range(len(self.flow))):
                x = self.flow[i](x, reverse=True)
            return x


class EfficientVRNVP(nn.Module):
    def __init__(self, module_list, in_channels, n_flow, hidden_dim, hidden_depth):
        super().__init__()
        assert in_channels % 2 == 0
        self.flow = nn.ModuleList([Flow(module_list, in_channels, hidden_dim, hidden_depth) for n in range(n_flow)])

    def forward(self, x, condition=None, reverse=False):
        if not reverse:
            logdet = 0
            for i in range(len(self.flow)):
                x, logdet_ = self.flow[i](x, condition=condition)
                logdet = logdet + logdet_
            return x, logdet
        else:
            for i in reversed(range(len(self.flow))):
                
                x = self.flow[i](x, condition=condition, reverse=True)
            return x, None

    def reverse(self, x, condition=None):
        return self.flow(x, condition=condition, reverse=True)


class VectorTransformer(nn.Module):
    def __init__(self, in_channel, n_flow, hidden_depth, hidden_dim):
        super().__init__()
        self.in_channel = in_channel
        self.n_flow = n_flow
        self.depth_submodules = hidden_depth
        self.hidden_dim = hidden_dim
        modules = [VectorActNorm, DoubleVectorCouplingBlock, Shuffle]
        self.realnvp = EfficientVRNVP(modules, self.in_channel, self.n_flow, self.hidden_dim,
                                   hidden_depth=self.depth_submodules)

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        input = input.flatten(start_dim=1)
        out, logdet = self.realnvp(input)
        return out[:, :, None, None], logdet

    def reverse(self, out):
        out = out.flatten(start_dim=1)
        return self.realnvp(out, reverse=True)[0][:, :, None, None]


class FactorTransformer(VectorTransformer):
    def __init__(self, in_channel, n_flow, hidden_depth, hidden_dim, n_factors = 2, factor_config = None):
        super().__init__(in_channel, n_flow, hidden_depth, hidden_dim)
        self.n_factors = n_factors
        self.factor_config = factor_config

    def forward(self, input):
        out, logdet = super().forward(input)
        if self.factor_config is None:
            out = torch.chunk(out, self.n_factors, dim=1)
        else:
            out = torch.split(out, self.factor_config, dim=1)
        return out, logdet

    def reverse(self, out):
        out = torch.cat(out, dim=1)
        return super().reverse(out)