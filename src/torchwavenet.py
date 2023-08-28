import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt

from .config_torchwavenet import ModelConfig


Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, dilation):
        '''
        :param residual_channels: audio conv
        :param dilation: audio conv dilation
        '''
        super().__init__()
        self.dilated_conv = Conv1d(
            residual_channels, 2 * residual_channels, 
            3, padding=dilation, dilation=dilation)

        self.output_projection = Conv1d(
            residual_channels, 2 * residual_channels, 1)

    def forward(self, x):
        y = self.dilated_conv(x)

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class Wave(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.input_projection = Conv1d(
            cfg.input_channels, cfg.residual_channels, 1)

        self.residual_layers = nn.ModuleList([
            ResidualBlock(cfg.residual_channels, 2**(i %
                          cfg.dilation_cycle_length))
            for i in range(cfg.residual_layers)
        ])
        self.skip_projection = Conv1d(
            cfg.residual_channels, cfg.residual_channels, 1)
        self.output_projection = Conv1d(
            cfg.residual_channels, cfg.input_channels, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, input):
        x = input
        x = self.input_projection(x)
        x = F.relu(x)

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x)
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x