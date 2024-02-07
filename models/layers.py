import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any

from complexPyTorch.complexLayers import NaiveComplexBatchNorm1d
from numpy.random import RandomState
from torch.nn import Parameter, Module
from torch_geometric.nn.dense.linear import Linear
from torch.nn.init import _calculate_correct_fan, _calculate_fan_in_and_fan_out
from torch_geometric.nn import MessagePassing, HeteroDictLinear, HeteroLinear
from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.nn.parameter_dict import ParameterDict
from torch_geometric.typing import Metadata, NodeType, EdgeType, Adj, PairTensor, OptTensor
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros
from models.myFunctional import apply_complex, complex_leaky_relu, complex_dropout, complex_relu
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.inits import ones


class ELU(Module):
    def __init__(self, alpha: float = 1., inplace: bool = False) -> None:
        super(ELU, self).__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.selu(input.real, self.inplace).type(torch.complex64) + 1j * F.selu(input.imag, self.inplace).type(
            torch.complex64)


class DictELU(Module):
    def __init__(self, inplace: bool = False) -> None:
        super(DictELU, self).__init__()
        self.inplace = inplace

    def forward(self, x_dict: Dict[NodeType, Tensor]):
        for node_type, x in x_dict.items():
            x_dict[node_type] = F.selu(x_dict[node_type].real, self.inplace).type(torch.complex64) + 1j * F.selu(
                x_dict[node_type].imag, self.inplace).type(
                torch.complex64)
        return x_dict


class DictComBN(torch.nn.Module):

    def __init__(
            self,
            in_channels: Union[int, Dict[Any, int]]
    ):
        super().__init__()
        self.in_channels = in_channels
        self.BN = torch.nn.ModuleDict({
            key: NaiveComplexBatchNorm1d(channels)
            for key, channels in self.in_channels.items()
        })

    def forward(self, x_dict: Dict[NodeType, Tensor]):
        for node_type, x in x_dict.items():
            x = x.to(torch.complex64)
            x_dict[node_type] = self.BN[node_type](x)
        return x_dict


class HeteroDictComLinear(torch.nn.Module):

    def __init__(
            self,
            in_channels: Union[int, Dict[Any, int]],
            out_channels: Union[int, Dict[Any, int]]
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(out_channels, int):
            self.lin = torch.nn.ModuleDict({
                key: ComplexLinear(channels, self.out_channels)
                for key, channels in self.in_channels.items()
            })
        else:
            self.lin = torch.nn.ModuleDict({
                key: ComplexLinear(channels, self.out_channels[key])
                for key, channels in self.in_channels.items()
            })

    def forward(self, x_dict: Dict[NodeType, Tensor]):
        out_dict = {}
        for node_type, x in x_dict.items():
            x = x.to(torch.complex64)
            out_dict[node_type] = self.lin[node_type](x)
        return out_dict


class MM_Dict_ACT(Module):
    def __init__(self, args) -> None:
        super(MM_Dict_ACT, self).__init__()
        self.K = args.user_num
        self.Nt = args.antenna_num
        self.p_max = args.p_max

    def forward(self, x_dict: Dict[NodeType, Tensor]):
        # 先进行分割
        # 这相当于求的角度
        # RF [B*Nt,k]
        RF = x_dict['ant'].real
        RF = torch.sigmoid(RF) * 2 * torch.pi
        # FrfOUT [B*K,]
        RF = torch.complex(torch.cos(RF), torch.sin(RF)) * torch.sqrt(torch.tensor(1 / self.Nt))
        RF = RF.reshape(-1, self.Nt, self.K)

        # 再转化为
        # 这个需要先reshape一下,这里也是每一行是一个用户
        BB = x_dict['user'][:, :self.K].reshape(-1, self.K, self.K)
        BB = torch.transpose(BB, 1, 2)
        # 然后再进行一个归一化
        C = torch.bmm(RF, BB)
        C_norm_squared = torch.norm(C, p='fro', dim=[1, 2], keepdim=True) ** 2
        scaling_factor = torch.sqrt(self.K / C_norm_squared)
        BB = BB * scaling_factor
        # 最后处理能量直接softmax每次都用完

        P = x_dict['user'][:, -1].real
        P = P.reshape(-1, self.K)
        P = torch.softmax(P, dim=1) * self.p_max
        P = P.reshape(-1, self.K)
        return RF, BB, P


class MM_ACT(Module):
    def __init__(self, args) -> None:
        super(MM_ACT, self).__init__()
        self.K = args.user_num
        self.Nt = args.antenna_num
        self.p_max = args.p_max

    def forward(self, input: Tensor):
        """
        :param input: [B*K,F_{RF,k},F_{BB,k},B_k]
        :return:[B,K,...]
        """
        # 先进行分割
        # 这相当于求的角度
        RF = input[:, :self.Nt].real
        RF = torch.sigmoid(RF) * 2 * torch.pi
        # FrfOUT [B*K,]
        RF = torch.complex(torch.cos(RF), torch.sin(RF)) * torch.sqrt(torch.tensor(1 / self.Nt))
        RF = RF.reshape(-1, self.K, self.Nt)
        RF = torch.transpose(RF, 1, 2)
        # RF = input[:, :self.Nt]
        # # 归一化每个元素
        # RF = RF / torch.abs(RF) * (1 / 4)
        # RF = RF.reshape(-1, self.K, self.Nt)
        # RF = torch.transpose(RF, 1, 2)
        # 再转化为
        # 这个需要先reshape一下
        BB = input[:, self.Nt:self.Nt + self.K].reshape(-1, self.K, self.K)
        BB = torch.transpose(BB, 1, 2)
        # 然后再进行一个归一化
        C = torch.bmm(RF, BB)
        C_norm_squared = torch.norm(C, p='fro', dim=[1, 2], keepdim=True) ** 2
        scaling_factor = torch.sqrt(self.K / C_norm_squared)
        BB = BB * scaling_factor
        # 最后处理能量直接softmax每次都用完

        P = input[:, -1].real
        P = P.reshape(-1, self.K)
        # P = F.normalize(P, p=1, dim=1)
        P = torch.softmax(P, dim=1) * self.p_max
        P = P.reshape(-1, self.K)
        return RF, BB, P


class MM_ACTv2(Module):
    def __init__(self, args) -> None:
        super(MM_ACTv2, self).__init__()
        self.K = args.user_num
        self.Nt = args.antenna_num
        self.p_max = args.p_max
        self.model = args.model

    def forward(self, RF: Tensor, BB: Tensor, P: Tensor):
        """
        :param input: [B*K,F_{RF,k},F_{BB,k},B_k]
        :return:[B,K,...]
        """
        # 先进行分割
        # 这相当于求的角度
        if self.model == "GAT":
            RF = RF.real
        RF = torch.sigmoid(RF) * 2 * torch.pi
        # FrfOUT [B*K,]
        RF = torch.complex(torch.cos(RF), torch.sin(RF)) * torch.sqrt(torch.tensor(1 / self.Nt))
        RF = RF.reshape(-1, self.K, self.Nt)
        RF = torch.transpose(RF, 1, 2)

        BB = BB.reshape(-1, self.K, self.K)
        BB = torch.transpose(BB, 1, 2)
        # 然后再进行一个归一化
        C = torch.bmm(RF, BB)
        C_norm_squared = torch.norm(C, p='fro', dim=[1, 2], keepdim=True) ** 2
        scaling_factor = torch.sqrt(self.K / C_norm_squared)
        BB = BB * scaling_factor
        # 最后处理能量直接softmax每次都用完
        if self.model == "GAT":
            P = P.real
        P = P.reshape(-1, self.K)
        P = torch.softmax(P, dim=1) * self.p_max
        P = P.reshape(-1, self.K)
        return RF, BB, P


class MM_ACTv2(Module):
    def __init__(self, args) -> None:
        super(MM_ACTv2, self).__init__()
        self.K = args.user_num
        self.Nt = args.antenna_num
        self.p_max = args.p_max

    def forward(self, RF: Tensor, BB: Tensor, P: Tensor):
        """
        :param input: [B*K,F_{RF,k},F_{BB,k},B_k]
        :return:[B,K,...]
        """
        # 先进行分割
        # 这相当于求的角度
        if self.model == "GAT":
            RF = RF.real
        RF = torch.sigmoid(RF) * 2 * torch.pi
        # FrfOUT [B*K,]
        RF = torch.complex(torch.cos(RF), torch.sin(RF)) * torch.sqrt(torch.tensor(1 / self.Nt))
        RF = RF.reshape(-1, self.K, self.Nt)
        RF = torch.transpose(RF, 1, 2)

        BB = BB.reshape(-1, self.K, self.K)
        BB = torch.transpose(BB, 1, 2)
        # 然后再进行一个归一化
        C = torch.bmm(RF, BB)
        C_norm_squared = torch.norm(C, p='fro', dim=[1, 2], keepdim=True) ** 2
        scaling_factor = torch.sqrt(self.K / C_norm_squared)
        BB = BB * scaling_factor
        # 最后处理能量直接softmax每次都用完
        if self.model == "GAT":
            P = P.real
        P = P.reshape(-1, self.K)
        P = torch.softmax(P, dim=1) * self.p_max
        P = P.reshape(-1, self.K)
        return RF, BB, P


class MM_ACTv3(Module):
    def __init__(self, args) -> None:
        super(MM_ACTv3, self).__init__()
        self.Nt = args.antenna_num
        self.p_max = args.p_max

    def forward(self, dicts, users):
        """
        :param input: [B*K,F_{RF,k},F_{BB,k},B_k]
        :return:[B,K,...]
        """
        for user in users:
            RF, BB, P = dicts[str(user) + "_rf"], dicts[str(user) + "_bb"], dicts[str(user) + "_p"]
            RF = torch.sigmoid(RF) * 2 * torch.pi
            # FrfOUT [B*K,]
            RF = torch.complex(torch.cos(RF), torch.sin(RF)) * torch.sqrt(torch.tensor(1 / self.Nt))
            RF = RF.reshape(-1, user, self.Nt)
            RF = torch.transpose(RF, 1, 2)
            BB = BB.reshape(-1, user, user)
            BB = torch.transpose(BB, 1, 2)
            # 然后再进行一个归一化
            C = torch.bmm(RF, BB)
            C_norm_squared = torch.norm(C, p='fro', dim=[1, 2], keepdim=True) ** 2
            scaling_factor = torch.sqrt(user / C_norm_squared)
            BB = BB * scaling_factor
            # 最后处理能量直接softmax每次都用完
            P = P.reshape(-1, user)
            P = torch.softmax(P, dim=1) * self.p_max
            P = P.reshape(-1, user)
            dicts[str(user) + "_rf"], dicts[str(user) + "_bb"], dicts[str(user) + "_p"] = RF, BB, P
        return dicts


class ComplexLeakyRelu(Module):

    def __init__(self, negative_slope: float = 1e-2):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input):
        return complex_leaky_relu(input, self.negative_slope)


class ComplexDropout(Module):
    def __init__(self, p=0.5):
        super(ComplexDropout, self).__init__()
        self.p = p

    def forward(self, input):
        if self.training:
            return complex_dropout(input, self.p)
        else:
            return input


# 复数线性层
class ComplexLinear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ComplexLinear, self).__init__()
        self.fc_r = Linear(in_features, out_features, bias=bias)
        self.fc_i = Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

    def forward(self, input):
        return apply_complex(self.fc_r, self.fc_i, input)

    def reset_parameters(self):
        self.complex_kaiming_normal_()

    def complex_kaiming_normal_(self, mode="fan_in"):
        fan_in, fan_out = _calculate_fan_in_and_fan_out(self.fc_r.weight)
        s = 1. / fan_in
        rng = RandomState()
        modulus = rng.rayleigh(scale=s, size=self.fc_r.weight.shape)
        phase = rng.uniform(low=-np.pi, high=np.pi, size=self.fc_r.weight.shape)
        weight_real = modulus * np.cos(phase)
        weight_imag = modulus * np.sin(phase)
        self.fc_r.weight = torch.nn.Parameter(torch.tensor(weight_real, dtype=torch.float32))
        self.fc_i.weight = torch.nn.Parameter(torch.tensor(weight_imag, dtype=torch.float32))


# 复数线性层
class ComplexToReal(Module):
    def __init__(self, in_features, out_features):
        super(ComplexToReal, self).__init__()
        self.fc_r = Linear(in_features, out_features, bias=False)
        self.fc_i = Linear(in_features, out_features, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def forward(self, input):
        return self.fc_r(input.real) + self.fc_i(input.imag) + self.bias

    def reset_parameters(self):
        self.complex_kaiming_normal_()

    def complex_kaiming_normal_(self, mode="fan_in"):
        fan = _calculate_correct_fan(self.fc_r.weight, mode)
        s = 1. / fan
        rng = RandomState(0)
        modulus = rng.rayleigh(scale=s, size=self.fc_r.weight.shape)
        phase = rng.uniform(low=-np.pi, high=np.pi, size=self.fc_r.weight.shape)
        weight_real = modulus * np.cos(phase)
        weight_imag = modulus * np.sin(phase)
        self.fc_r.weight = torch.nn.Parameter(torch.tensor(weight_real, dtype=torch.float32))
        self.fc_i.weight = torch.nn.Parameter(torch.tensor(weight_imag, dtype=torch.float32))


class RealToComplex(Module):
    def __init__(self, in_features, out_features):
        super(RealToComplex, self).__init__()
        self.real_linear = Linear(in_features, out_features)
        self.imaginary_linear = Linear(in_features, out_features)

    def forward(self, x):
        real = self.real_linear(x)
        imaginary = self.imaginary_linear(x)
        output = torch.complex(real, imaginary)
        return output


class EdgeGATConv(MessagePassing):

    def __init__(
            self,
            in_channels,
            out_channels,
            heads: int = 1,
            concat: bool = True,
            negative_slope: float = 0.2,
            dropout: float = 0.0,
            residual: bool = False,
            add_self_loops: bool = True,
            bias: bool = True,
            share_weights: bool = False,
            edge_dim: Optional[int] = None,
            **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.share_weights = share_weights
        self.residual = residual
        self.edge_dim = edge_dim

        # 权重初始化
        self.lin_l = ComplexLinear(in_channels, heads * out_channels, bias=False)

        if share_weights:
            self.lin_r = self.lin_l
        else:
            self.lin_r = ComplexLinear(in_channels, heads * out_channels, bias=False)

        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = torch.nn.Parameter(torch.zeros(heads * out_channels, dtype=torch.complex64))
        elif bias and not concat:
            self.bias = torch.nn.Parameter(torch.zeros(heads * out_channels, dtype=torch.complex64))
        else:
            self.register_parameter('bias', None)

        if edge_dim != 0:
            self.lin_edge = ComplexLinear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = None

        if residual:
            if self.in_channels != out_channels * heads:
                self.res_fc = ComplexLinear(in_channels, heads * out_channels)
            else:
                self.res_fc = None
        else:
            self.res_fc = None

        self._alpha = None
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.fc_r.weight)
        glorot(self.lin_l.fc_i.weight)
        glorot(self.lin_r.fc_r.weight)
        glorot(self.lin_r.fc_i.weight)
        if self.lin_edge is not None:
            glorot(self.lin_edge.fc_r.weight)
            glorot(self.lin_edge.fc_i.weight)

        if self.residual:
            glorot(self.res_fc.fc_i.weight)
            glorot(self.res_fc.fc_i.weight)
        weight_real_att = torch.Tensor(1, self.heads, self.out_channels)
        weight_imag_att = torch.Tensor(1, self.heads, self.out_channels)
        glorot(weight_real_att)
        glorot(weight_imag_att)
        self.att.data = torch.nn.Parameter(torch.tensor(weight_real_att + 1j * weight_imag_att, dtype=torch.complex64))
        self.complex_kaiming_normal_()

    def complex_kaiming_normal_(self, mode="fan_in"):
        rng = RandomState()
        fan_l = _calculate_correct_fan(self.lin_l.fc_r.weight, mode)
        s_l = 1. / fan_l
        modulus_l = rng.rayleigh(scale=s_l, size=self.lin_l.fc_r.weight.shape)
        phase_l = rng.uniform(low=-np.pi, high=np.pi, size=self.lin_l.fc_r.weight.shape)
        weight_real_l = modulus_l * np.cos(phase_l)
        weight_imag_l = modulus_l * np.sin(phase_l)
        self.lin_l.fc_r.weight = torch.nn.Parameter(torch.tensor(weight_real_l, dtype=torch.float32))
        self.lin_l.fc_i.weight = torch.nn.Parameter(torch.tensor(weight_imag_l, dtype=torch.float32))

        fan_r = _calculate_correct_fan(self.lin_l.fc_r.weight, mode)
        s_r = 1. / fan_r
        modulus_r = rng.rayleigh(scale=s_r, size=self.lin_r.fc_r.weight.shape)
        phase_r = rng.uniform(low=-np.pi, high=np.pi, size=self.lin_r.fc_r.weight.shape)
        weight_real_r = modulus_r * np.cos(phase_r)
        weight_imag_r = modulus_r * np.sin(phase_r)
        self.lin_r.fc_r.weight = torch.nn.Parameter(torch.tensor(weight_real_r, dtype=torch.float32))
        self.lin_r.fc_i.weight = torch.nn.Parameter(torch.tensor(weight_imag_r, dtype=torch.float32))

        # 初始化注意力权重
        s_att = 1. / self.heads
        modulus_att = rng.rayleigh(scale=s_att, size=[self.heads, self.out_channels])
        phase_att = rng.uniform(low=-np.pi, high=np.pi, size=[self.heads, self.out_channels])
        weight_real_att = modulus_att * np.cos(phase_att)
        weight_imag_att = modulus_att * np.sin(phase_att)
        self.att.data = torch.nn.Parameter(torch.tensor(weight_real_att + 1j * weight_imag_att, dtype=torch.complex64))

    def forward(self, x, edge_index, edge_attr=None):

        H, C = self.heads, self.out_channels

        assert x.dim() == 2
        x_l = self.lin_l(x).view(-1, H, C)

        if self.share_weights:
            x_r = x_l
        else:
            x_r = self.lin_r(x).view(-1, H, C)
        out = self.propagate(edge_index, x=(x_l, x_r), size=None, edge_attr=edge_attr)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if self.res_fc is not None:
            resval = self.res_fc(x).view(-1, self.heads * self.out_channels)
            out = out + resval
        return out

    def message(self, x_j, x_i, index, edge_attr):
        x = x_i + x_j
        if self.lin_edge is not None:
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr

        x = complex_leaky_relu(x, self.negative_slope)

        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha.real, index)
        self._alpha = alpha

        alpha = complex_dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)


class GATv2Conv(MessagePassing):

    def __init__(
            self,
            in_channels,
            out_channels,
            heads: int = 1,
            concat: bool = True,
            negative_slope: float = 0.2,
            dropout: float = 0.0,
            residual: bool = False,
            add_self_loops: bool = True,
            bias: bool = True,
            share_weights: bool = False,
            edge_dim: Optional[int] = None,
            **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.share_weights = share_weights
        self.residual = residual
        self.edge_dim = edge_dim

        # 权重初始化
        self.lin_l = Linear(in_channels, heads * out_channels, bias=bias,
                            weight_initializer='glorot')

        if share_weights:
            self.lin_r = self.lin_l
        else:
            self.lin_r = Linear(in_channels, heads * out_channels,
                                bias=bias, weight_initializer='glorot')

        self.att = Parameter(torch.empty(1, heads, out_channels))

        if bias and concat:
            self.bias = torch.nn.Parameter(torch.zeros(heads * out_channels))
        elif bias and not concat:
            self.bias = torch.nn.Parameter(torch.zeros(heads * out_channels))
        else:
            self.register_parameter('bias', None)

        if edge_dim != 0:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
        else:
            self.lin_edge = None

        if residual:
            if self.in_channels != out_channels * heads:
                self.res_fc = Linear(in_channels, heads * out_channels,
                                     weight_initializer='glorot')
            else:
                self.res_fc = None
        else:
            self.res_fc = None

        self._alpha = None
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        self.res_fc.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr=None):

        H, C = self.heads, self.out_channels

        assert x.dim() == 2
        x_l = self.lin_l(x).view(-1, H, C)

        if self.share_weights:
            x_r = x_l
        else:
            x_r = self.lin_r(x).view(-1, H, C)
        out = self.propagate(edge_index, x=(x_l, x_r), size=None, edge_attr=edge_attr)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if self.res_fc is not None:
            resval = self.res_fc(x).view(-1, self.heads * self.out_channels)
            out = out + resval
        return out

    def message(self, x_j, x_i, index, edge_attr):
        x = x_i + x_j
        if self.lin_edge is not None:
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr

        x = F.leaky_relu(x, self.negative_slope)

        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index)
        self._alpha = alpha

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)


# 使用于二分图的原始GAT
class BipGATConv(MessagePassing):

    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            heads: int = 1,
            concat: bool = True,
            negative_slope: float = 0,
            dropout: float = 0.0,
            bias: bool = True,
            share_weights: bool = False,
            edge_dim: Optional[int] = None,
            **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.share_weights = share_weights
        self.edge_dim = edge_dim

        if isinstance(in_channels, int):
            # 权重初始化
            self.lin_l = ComplexLinear(in_channels, heads * out_channels, bias=bias)
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = ComplexLinear(in_channels, heads * out_channels, bias=bias)
        else:
            self.lin_l = ComplexLinear(in_channels[0], heads * out_channels,
                                       bias=bias)
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = ComplexLinear(in_channels[1], heads * out_channels, bias=bias)
        # 默认注意力参数
        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        # self.user_att = Parameter(torch.Tensor(1, heads, out_channels))

        # self.ant_att = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            # 这里是不要偏置的
            self.lin_edge = ComplexLinear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = None

        if bias and concat:
            self.bias = torch.nn.Parameter(torch.zeros(heads * out_channels, dtype=torch.complex64))
        elif bias and not concat:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels, dtype=torch.complex64))
        else:
            self.register_parameter('bias', None)

        # 默认是使用残差的
        self.res_l = ComplexLinear(in_channels[0], heads * out_channels)
        self.res_r = ComplexLinear(in_channels[1], heads * out_channels)

        self._alpha = None
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.fc_r.weight)
        glorot(self.lin_l.fc_i.weight)
        glorot(self.lin_r.fc_r.weight)
        glorot(self.lin_r.fc_i.weight)
        if self.lin_edge is not None:
            glorot(self.lin_edge.fc_r.weight)
            glorot(self.lin_edge.fc_i.weight)

        glorot(self.res_l.fc_r.weight)
        glorot(self.res_l.fc_i.weight)

        glorot(self.res_r.fc_r.weight)
        glorot(self.res_r.fc_i.weight)

        weight_real_att = torch.Tensor(1, self.heads, self.out_channels)
        weight_imag_att = torch.Tensor(1, self.heads, self.out_channels)
        glorot(weight_real_att)
        glorot(weight_imag_att)
        self.att.data = torch.nn.Parameter(torch.tensor(weight_real_att + 1j * weight_imag_att, dtype=torch.complex64))
        self.complex_kaiming_normal_()

    def complex_kaiming_normal_(self, mode="fan_in"):
        rng = RandomState()
        fan_l = _calculate_correct_fan(self.lin_l.fc_r.weight, mode)
        s_l = 1. / fan_l
        modulus_l = rng.rayleigh(scale=s_l, size=self.lin_l.fc_r.weight.shape)
        phase_l = rng.uniform(low=-np.pi, high=np.pi, size=self.lin_l.fc_r.weight.shape)
        weight_real_l = modulus_l * np.cos(phase_l)
        weight_imag_l = modulus_l * np.sin(phase_l)
        self.lin_l.fc_r.weight = torch.nn.Parameter(torch.tensor(weight_real_l, dtype=torch.float32))
        self.lin_l.fc_i.weight = torch.nn.Parameter(torch.tensor(weight_imag_l, dtype=torch.float32))

        fan_r = _calculate_correct_fan(self.lin_l.fc_r.weight, mode)
        s_r = 1. / fan_r
        modulus_r = rng.rayleigh(scale=s_r, size=self.lin_r.fc_r.weight.shape)
        phase_r = rng.uniform(low=-np.pi, high=np.pi, size=self.lin_r.fc_r.weight.shape)
        weight_real_r = modulus_r * np.cos(phase_r)
        weight_imag_r = modulus_r * np.sin(phase_r)
        self.lin_r.fc_r.weight = torch.nn.Parameter(torch.tensor(weight_real_r, dtype=torch.float32))
        self.lin_r.fc_i.weight = torch.nn.Parameter(torch.tensor(weight_imag_r, dtype=torch.float32))

        # 初始化注意力权重
        s_att = 1. / self.heads
        modulus_att = rng.rayleigh(scale=s_att, size=[self.heads, self.out_channels])
        phase_att = rng.uniform(low=-np.pi, high=np.pi, size=[self.heads, self.out_channels])
        weight_real_att = modulus_att * np.cos(phase_att)
        weight_imag_att = modulus_att * np.sin(phase_att)
        self.att.data = torch.nn.Parameter(torch.tensor(weight_real_att + 1j * weight_imag_att, dtype=torch.complex64))
        # self.user_att.data = torch.nn.Parameter(torch.tensor(weight_real_att + 1j * weight_imag_att, dtype=torch.complex64))
        # self.ant_att.data = torch.nn.Parameter(torch.tensor(weight_real_att + 1j * weight_imag_att, dtype=torch.complex64))

    def forward(
            self,
            x: Union[Tensor, PairTensor],
            edge_index: Adj,
            # user_edge_index : OptTensor = None,
            # antenna_edge_index : OptTensor = None,
            edge_attr: OptTensor = None):

        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None

        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
        # 这里进行一次消息传递机制，只会获得目标节点特征所以要进行两次消息传递机制，其实跟异构图是一样的
        # x_l起始节点特征 x_r 中心节点特征
        # 起始节点是天线，中心节点是用户
        out_r = self.propagate(edge_index, x=(x_l, x_r), size=None, edge_attr=edge_attr, mode="bip")
        # 修改一下起始节点和终止节点
        edge_index = edge_index[[1, 0], :]
        # 这里节点特征是不变的
        out_l = self.propagate(edge_index, x=(x_r, x_l), size=None, edge_attr=edge_attr, mode="bip")
        # 判断是否判断是否在子图上进行消息传递
        # if antenna_edge_index is not None:
        #     out_l2 = self.propagate(antenna_edge_index, x=(x_l, x_l), size=None,edge_attr= None,mode= "ant")
        #     out_l = out_l + out_l2

        # if user_edge_index is not None:
        #     out_r2 = self.propagate(user_edge_index, x=(x_r, x_r), size=None,edge_attr= None,mode= "user")
        #     out_r = out_r + out_r2

        if self.concat:
            out_r = out_r.view(-1, self.heads * self.out_channels)
            out_l = out_l.view(-1, self.heads * self.out_channels)
        else:
            out_l = out_l.mean(dim=1)
            out_r = out_r.mean(dim=1)

        if self.bias is not None:
            out_l += self.bias
            out_r += self.bias
        # 这里是默认是使用残差的
        out_l = out_l + self.res_l(x[0]).view(-1, self.heads * self.out_channels)
        out_r = out_r + self.res_r(x[1]).view(-1, self.heads * self.out_channels)

        return out_l, out_r

    def message(self, x_j: Tensor, x_i: Tensor, index: Tensor, edge_attr: OptTensor, mode: Any):
        x = x_i + x_j

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr

        x = complex_leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)

        # if mode == "bip":
        #     alpha = (x * self.att).sum(dim=-1)
        # elif mode == "user":
        #     alpha = (x * self.user_att).sum(dim=-1)
        # elif mode == "ant":
        #     alpha = (x * self.ant_att).sum(dim=-1)
        alpha = softmax(alpha.real, index)
        self._alpha = alpha

        alpha = complex_dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)


class HGTConv(MessagePassing):

    def __init__(
            self,
            in_channels: Union[int, Dict[str, int]],
            out_channels: int,
            metadata: Metadata,
            heads: int = 1,
            edge_dim: Optional[int] = None,
            **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)
        # 如果是int 的话则代表输入节点特征维度都相同
        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}
        # 参数赋值
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        # 为两种边进行线性变化
        self.lin = ModuleDict()
        self.res_fc = ModuleDict()
        self.bias = ParameterDict()
        self.metadata = metadata
        # 为不同的节点类型设置注意力
        for node_type, in_channels in self.in_channels.items():
            self.lin[node_type] = ComplexLinear(in_channels, heads * out_channels, bias=False)
            self.bias[node_type] = Parameter(torch.zeros(heads * out_channels, dtype=torch.complex64))
            self.res_fc[node_type] = ComplexLinear(in_channels, heads * out_channels)

        self.att = ParameterDict()
        for edge_type in metadata[1]:
            edge_type = '__'.join(edge_type)
            self.att[edge_type] = Parameter(torch.randn(1, heads, out_channels, dtype=torch.complex64))

        self.lin_edge = ComplexLinear(edge_dim, heads * out_channels, bias=False)

        # 参数初始化
        self.reset_parameters()

    def reset_parameters(self):
        rng = RandomState()
        for edge_type in self.metadata[1]:
            edge_type = '__'.join(edge_type)
            # 初始化注意力权重
            s_att = 1. / self.heads
            modulus_att = rng.rayleigh(scale=s_att, size=[self.heads, self.out_channels])
            phase_att = rng.uniform(low=-np.pi, high=np.pi, size=[self.heads, self.out_channels])
            weight_real_att = modulus_att * np.cos(phase_att)
            weight_imag_att = modulus_att * np.sin(phase_att)
            self.att[edge_type].data = torch.nn.Parameter(
                torch.tensor(weight_real_att + 1j * weight_imag_att, dtype=torch.complex64))

    def forward(
            self,
            x_dict: Dict[NodeType, Tensor],
            edge_index_dict: Dict[EdgeType, Tensor],
            edge_attr_dict: Dict[EdgeType, Tensor]
    ) -> Dict[NodeType, Optional[Tensor]]:

        H, D = self.heads, self.out_channels

        node_dict, res_dict, edge_dict, out_dict = {}, {}, {}, {}

        # Iterate over node-types:
        # 计算每个节点映射以及相关残差
        for node_type, x in x_dict.items():
            x = x.to(torch.complex64)
            node_dict[node_type] = self.lin[node_type](x).view(-1, H, D)
            res_dict[node_type] = self.res_fc[node_type](x)

        # Iterate over edge-types:
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            # 获得边特征
            edge_attr = edge_attr_dict[edge_type]
            edge_type = '__'.join(edge_type)
            # 对边特征进行线性变化
            edge_attr = self.lin_edge(edge_attr).view(-1, H, D)
            # propagate_type: (k: Tensor, q: Tensor, v: Tensor, rel: Tensor)
            out = self.propagate(edge_index, x=(node_dict[src_type], node_dict[dst_type]), edge_attr=edge_attr,
                                 edge_type=edge_type,
                                 size=(len(node_dict[src_type]), len(node_dict[dst_type])))
            out_dict[dst_type] = out
        # Iterate over node-types:
        for node_type, out in out_dict.items():
            out_dict[node_type] = out.view(-1, H * D) + res_dict[node_type]

        return out_dict

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr, edge_type, index: Tensor) -> Tensor:
        x = x_i + x_j + edge_attr
        x = complex_leaky_relu(x, 0)
        alpha = (x * self.att[edge_type]).sum(dim=-1)
        alpha = softmax(alpha.real, index)
        return x_j * alpha.unsqueeze(-1)
