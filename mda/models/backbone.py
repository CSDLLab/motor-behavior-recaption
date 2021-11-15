import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


def conv_init(m):
    """Initilize the convolution kernel

    :param m: module
    :type m: nn.Module
    :return: initilized module
    :rtype: nn.Module
    """
    nn.init.kaiming_normal_(m.weight, mode='fan_out')
    if m.bias is not None:
        nn.init.constant_(m.bias, 0)


def bn_init(bn, scale):
    """initialize the batch norm kernel

    :param bn: module
    :type bn: nn.Module
    :param scale: the normalize scale
    :type scale: int
    :return: None
    :rtype: None
    """
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.reset_parameters()
    
    def reset_parameters(self):
        conv_init(self.conv)
        bn_init(self.bn, 1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)


class ConvGraph(nn.Module):
    def __init__(self, in_channels, proj_channels):
        super(ConvGraph, self).__init__()
        self.proj_channels = proj_channels
        self.proj1 = nn.Conv2d(in_channels, proj_channels, 1)
        self.proj2 = nn.Conv2d(in_channels, proj_channels, 1)
        self.soft = nn.Softmax(-2)
        self.reset_parameters()
    
    def reset_parameters(self):
        conv_init(self.proj1)
        conv_init(self.proj2)

    def forward(self, x):
        N, C, T, V = x.size()
        embed1 = self.proj1(x)
        embed2 = self.proj2(x)
        out = self.soft(torch.mm(embed1, embed2) / embed1.size(-1))
        return out


class ConvTemporalGraphical(nn.Module):
    """The basic module for applying a graph temporal model

    Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        """The basic module for applying a graph convolution.

        :param in_channels: Number of channels in the inputs sequence
        :type in_channels: int
        :param out_channels: Number of channels produced by the convolutoin
        :type out_channels: int
        :param kernel_size: Size of the graph convolution kernel
        :type kernel_size: int
        :param t_kernel_size: Size of the temporal covolution kernel, defaults to 1
        :type t_kernel_size: int, optional
        :param t_stride: Stride of the temporal convolution, defaults to 1
        :type t_stride: int, optional
        :param t_padding: Temporal zero-padding added to both sides of the inputs, defaults to 0
        :type t_padding: int, optional
        :param t_dilation: Spacing between temporal kernel elements, defaults to 1
        :type t_dilation: int, optional
        :param bias: If ``True``, adds a learnable bias to the outputs, defaults to True
        :type bias: bool, optional
        """
        super(ConvTemporalGraphical, self).__init__()
        self.kernel_size = kernel_size
        # TODO: is this multiple channels with kernel size is necessary
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias
        )
    
    def forward(self, x, A):
        """Forward function

        :param x: features inputs
        :type x: torch.Tensor
        :param A: adjacent matrix for modeling relationships
        :type A: torch.Tensor
        """
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous(), A
    

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence

    Shape:
        - Inputs[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Inputs[1]: Input graph adjacent matrix in :math:`(K, V, V)` format
        - Outputs[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Outputs[1]: Graph adjacent matrix for outputs data in :math:`(K, V, V)` format
        where
            :math:`N` is batch size
            :math:`K` is the spatial kernel size, as :math:`K` == kernel_size[1]`
            :math:`T_{in}/T_{out}` is a length of inputs/outputs sequence
            :math:`V` is the number of graph nodes
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        """Applies a spatial temporal graph convolution over an input graph sequence

        :param in_channels: Number of channels in the inputs sequence data
        :type in_channels: int
        :param out_channels: Number of channels produced by the convlution
        :type out_channels: int
        :param kernel_size: Size of the graph convolution kernel and temporal convolution kernel
        :type kernel_size: tuple
        :param stride: Stride of the temporal convolution, defaults to 1
        :type stride: int, optional
        :param dropout: Dropout rate of the final output, defaults to 0
        :type dropout: int, optional
        :param residual: If ``True``, applies a residual mechanism, defaults to True
        :type residual: bool, optional
        """
        super(st_gcn, self).__init__()

        assert len(kernel_size) == 2
        assert kernel_size[1] % 2 == 1
        padding = ((kernel_size[1] - 1) // 2, 0)
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[0])

        # inputs shape: N, C, T, V, outputs shape: N, C, T, V
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                # (temporal dim, vertices dimension)
                (kernel_size[1], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )

        if not residual:
            self.residual = lambda x: 0.0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)
                ),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU()
    
    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res.contiguous()
        x = self.lrelu(x)
        return x, A


class Weight(nn.Module):
    """Weight parameter generation for adjacent matrix

    """
    def __init__(self, channels, output_nodes):
        super(Weight, self).__init__()
        self.register_buffer('weight', nn.Parameter(torch.rand(2, output_nodes, requires_grad=True)))
        self.weight.data.uniform_(-1, 1)
    
    def forward(self, x):
        return torch.einsum('kij,ki->kij', (x, self.weight))


class WeightD(nn.Module):
    """Weight parameter generation for adjacent matrix

    """
    def __init__(self, channels, output_nodes):
        super(WeightD, self).__init__()
        self.register_buffer('weight', nn.Parameter(torch.rand(2, output_nodes, requires_grad=True)))
        self.weight.data.uniform_(-1, 1)
    
    def forward(self, x):
        return torch.einsum('kji,ki->kij', (x, self.weight))


class DataBN(nn.Module):
    def __init__(self, in_channels, num_nodes):
        super(DataBN, self).__init__()
        self.bn = nn.BatchNorm1d(in_channels * num_nodes)
        self.reset_parameters()
    
    def reset_parameters(self):
        bn_init(self.bn, 1)
    
    def forward(self, x):
        n, c, t, v = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(n, v * c, t)
        x = self.bn(x)
        x = x.view(n, v, c, t).permute(0, 1, 3, 2).contiguous().view(n, c, t, v)
        return x


class Upsampling(nn.Module):
    def __init__(self, input_nodes, output_nodes, adjacent_matrix, in_channels):
        super(Upsampling, self).__init__()
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.register_buffer('A', adjacent_matrix)
        self.w = Weight(in_channels, output_nodes)
    
    def forward(self, x):
        assert x.size(3) == self.input_nodes
        assert self.A.size(0) == 2
        assert self.A.size(1) == self.output_nodes

        res = self.w(self.A)
        res = torch.einsum('kij,nctj->ncti', (res, x))
        return res
