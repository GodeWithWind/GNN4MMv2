import numpy as np
import torch
from typing import Dict, Optional, Union, Any
from torch.nn import PReLU
from numpy.random import RandomState
from torch.nn import Parameter, Module

from torch_geometric.nn.dense.linear import Linear
from torch.nn.init import _calculate_correct_fan, _calculate_fan_in_and_fan_out
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.nn.parameter_dict import ParameterDict
from torch_geometric.typing import Metadata, NodeType, EdgeType
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot, zeros,reset
from models.myFunctional import apply_complex
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.norm import GraphNorm




class EdgeDictBN(Module):
    def __init__(self, in_channels: int, metadata: Metadata,):
        super(EdgeDictBN, self).__init__()

        self.BN = ModuleDict()
        for edge_type in metadata[1]:
            edge_type = '__'.join(edge_type)
            self.BN[edge_type] = GraphNorm(in_channels)

    def forward(self, x_dict: Dict[NodeType, Tensor]):
        # 这里进行拼接
        for edge_type, x in x_dict.items():
            # print(edge_type)
            x_dict[edge_type] = self.BN["__".join(edge_type)](x)
        return x_dict

class NodeDictBN(torch.nn.Module):

    def __init__(
            self, in_channels: int, metadata: Metadata):
        super(NodeDictBN,self).__init__()

        self.BN = torch.nn.ModuleDict({
            node_type: GraphNorm(in_channels)
            for node_type in metadata[0]})

    def forward(self, x_dict: Dict[Any, Tensor]):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.BN[node_type](x)
        return x_dict


class DictELUV2(Module):
    def __init__(self, inplace: bool = False) -> None:
        super(DictELUV2, self).__init__()
        self.inplace = inplace
        # self.relu = ModuleDict()
        # self.relu["ant"] = torch.nn.PReLU(init= 0.1)
        # self.relu["user"] = torch.nn.PReLU(init= 0.1)
    def forward(self, x_dict: Dict[NodeType, Tensor]):
        for node_type, x in x_dict.items():
            x_dict[node_type] = F.leaky_relu(x_dict[node_type],negative_slope=0)
            # x_dict[node_type] =self.relu[node_type](x)
        return x_dict


class MM_ACTv3(Module):
    def __init__(self, args) -> None:
        super(MM_ACTv3, self).__init__()
        self.K = args.user_num
        self.Nt = args.antenna_num
        self.p_max = args.p_max
        self.model = args.model

    def forward(self, RF: Tensor,BB: Tensor,P: Tensor):
        """
        :param input: [B*K,F_{RF,k},F_{BB,k},B_k]
        :return:[B,K,...]
        """
        # 能量进行归一化
        P = torch.sigmoid(P)
        row_sums = torch.sum(P, dim=1,keepdim = True)
        scaling_factors = torch.max(torch.ones(row_sums.size()).to('cuda'), row_sums.to('cuda'))
        P = torch.div(P,scaling_factors)* self.p_max


        # 这里对RF进行归一化操作
        RF = RF / torch.abs(RF) * torch.sqrt(torch.tensor(1 / self.Nt))
        RF = RF.reshape(-1, self.K, self.Nt)
        RF = torch.transpose(RF, 1, 2)
        # 这里需要再进行一个量化处理
        RF = quantize_phase(4,RF)
        return RF, BB, P

# 定义量化函数
def quantize_phase(B, W):
    delta = 2 * torch.pi / 2 ** B  # 量化间隔
    # r = torch.zeros_like(W, dtype=torch.complex64)  # 初始化复数张量

    # 计算相位并量化
    phase = torch.angle(W)  # 获取相位
    phase_quantized = torch.round(phase / delta) * delta  # 优化后的量化相位
    magnitude = torch.abs(W)  # 获取幅度
    # 量化后的复数张量
    r = magnitude * torch.exp(1j * phase_quantized)
    return r



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
        self.fc_r.weight = Parameter(torch.tensor(weight_real, dtype=torch.float32))
        self.fc_i.weight = Parameter(torch.tensor(weight_imag, dtype=torch.float32))


# 复数线性层
class ComplexToReal(Module):
    def __init__(self, in_features, out_features):
        super(ComplexToReal, self).__init__()
        self.fc_r = torch.nn.Linear(in_features, out_features, bias=False)
        self.fc_i = torch.nn.Linear(in_features, out_features, bias=False)
        self.bias = Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def forward(self, input):
        return self.fc_r(input.real) + self.fc_i(input.imag) + self.bias

    def reset_parameters(self):
        pass
        # self.complex_kaiming_normal_()

    def complex_kaiming_normal_(self, mode="fan_in"):
        fan = _calculate_correct_fan(self.fc_r.weight, mode)
        s = 1. / fan
        rng = RandomState(0)
        modulus = rng.rayleigh(scale=s, size=self.fc_r.weight.shape)
        phase = rng.uniform(low=-np.pi, high=np.pi, size=self.fc_r.weight.shape)
        weight_real = modulus * np.cos(phase)
        weight_imag = modulus * np.sin(phase)
        self.fc_r.weight = Parameter(torch.tensor(weight_real, dtype=torch.float32))
        self.fc_i.weight = Parameter(torch.tensor(weight_imag, dtype=torch.float32))


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
            self.bias = Parameter(torch.zeros(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        if edge_dim != 0 :
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
        else:
            self.lin_edge = None

        if residual and concat:
             self.res_fc = Linear(in_channels, heads * out_channels,
                                   weight_initializer='glorot')
        elif bias and not concat:
             self.res_fc = Linear(in_channels, out_channels,
                                   weight_initializer='glorot')
        else:
             self.res_fc = None
        self.PReLU = PReLU()

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
            resval = self.res_fc(x)
            out = out + resval
        return out

    def message(self, x_j, x_i, index, edge_attr):
        x = x_i + x_j
        if self.lin_edge is not None:
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr

        # x = F.leaky_relu(x, self.negative_slope)
        x = self.PReLU(x)

        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index)
        self._alpha = alpha

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)



class GATv3Conv(MessagePassing):
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
            edge_out: Optional[int] = None,
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
            self.bias = Parameter(torch.zeros(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        if edge_dim != 0 :
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
            self.edge_up = Linear(edge_dim + 2* out_channels, edge_out, bias=True)
        else:
            self.lin_edge = None

        if residual and concat:
             self.res_fc = Linear(in_channels, heads * out_channels,
                                   weight_initializer='glorot')
        elif bias and not concat:
             self.res_fc = Linear(in_channels, out_channels,
                                   weight_initializer='glorot')
        else:
             self.res_fc = None
        self.PReLU = PReLU()

        self._alpha = None
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
            self.edge_up.reset_parameters()
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
            resval = self.res_fc(x)
            out = out + resval

        out_mean = out.view(-1, self.heads,self.out_channels).mean(dim=1)
        edge_attr = self.edge_up(torch.cat((out_mean[edge_index[0], :], edge_attr, out_mean[edge_index[1], :]), dim=1))
        return out , edge_attr

    def message(self, x_j, x_i, index, edge_attr):
        x = x_i + x_j
        if self.lin_edge is not None:
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr

        # x = F.leaky_relu(x, self.negative_slope)
        x = self.PReLU(x)

        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index)
        self._alpha = alpha

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)



class HGTConvV2(MessagePassing):
    def __init__(
            self,
            in_channels: Union[int, Dict[str, int]],
            out_channels: int,
            inter_edge_in_channels: Union[int, Dict[str, int]],
            edge_out_channels: int,
            metadata: Metadata,
            heads: int = 1,
            ext_edge_in_channels: Optional[int] = None,
            **kwargs,
    ):
        super().__init__(aggr='sum', node_dim=0, **kwargs)
        # 如果是int 的话则代表输入节点特征维度都相同
        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

 
        # 参数赋值
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_edge_in_channels = inter_edge_in_channels
        self.ext_edge_in_channels = ext_edge_in_channels
        self.edge_out_channels = edge_out_channels
        self.metadata = metadata
        self.heads = heads


        self.lin_src = ModuleDict()
        self.lin_dst = ModuleDict()
        self.res_fc = ModuleDict()

        self.bias = ParameterDict()
        self.lin_edge = ModuleDict()
        self.att = ParameterDict()
        self.edge_up = ModuleDict()

        for node_type, in_channels in self.in_channels.items():
            # 设置偏执
            self.bias[node_type] = Parameter(torch.zeros(self.heads * self.out_channels))
            # 设置残差
            self.res_fc[node_type] = Linear(in_channels, self.heads * self.out_channels, bias=False, weight_initializer='glorot')

        # 每个不同的边有一套注意力系数，总共有四套注意力机制
        for edge_type in metadata[1]:
            src_type, _, dst_type = edge_type
            edge_type = '__'.join(edge_type)
            self.lin_src[edge_type] = Linear(self.in_channels[src_type], heads * out_channels, bias=True,
                                                weight_initializer='glorot')
            self.lin_dst[edge_type] = Linear(self.in_channels[dst_type], heads * out_channels, bias=True,
                                                weight_initializer='glorot')
            self.att[edge_type] = Parameter(torch.empty(1, heads, out_channels))

            # 用于将边特征
            if src_type != dst_type:
                # 边特征用于计算注意力系数
                self.lin_edge[edge_type] = Linear(self.ext_edge_in_channels, heads * out_channels, bias=True,
                                                    weight_initializer='glorot')
                self.edge_up[edge_type] = Linear(self.ext_edge_in_channels + 2 * out_channels, self.edge_out_channels, bias=True,
                                                weight_initializer='glorot')
            else:
                self.lin_edge[edge_type] = Linear(self.inter_edge_in_channels, heads * out_channels, bias=True,
                                    weight_initializer='glorot')
                self.edge_up[edge_type] = Linear(self.inter_edge_in_channels + 2 * out_channels, self.edge_out_channels, bias=True,
                                                weight_initializer='glorot')
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.lin_src)
        reset(self.lin_dst)
        reset(self.res_fc)
        reset(self.lin_edge)
        glorot(self.att)
        zeros(self.bias)

    def forward(
            self,
            x_dict: Dict[NodeType, Tensor],
            edge_index_dict: Dict[EdgeType, Tensor],
            edge_attr_dict: Dict[EdgeType, Tensor]
    ) -> Dict[NodeType, Optional[Tensor]]:

        H, D = self.heads, self.out_channels

        res_dict, out_dict = {}, {}
        node_dict = {}

        # Iterate over node-types:
        # 计算每个节点映射以及相关残差
        for node_type, x in x_dict.items():
            # 计算残差
            res_dict[node_type] = self.res_fc[node_type](x)
            node_dict[node_type] = []

        # Iterate over edge-types:
        for edge_type, edge_index in edge_index_dict.items():
            edge_attr = edge_attr_dict[edge_type]
            src_type, _, dst_type = edge_type
            # 获得边特征
            edge_type = '__'.join(edge_type)
            # 节点特征进行映射
            src_x = self.lin_src[edge_type](x_dict[src_type]).view(-1, self.heads, self.out_channels)
            dst_x = self.lin_dst[edge_type](x_dict[dst_type]).view(-1, self.heads, self.out_channels)
            out = self.propagate(edge_index, x=(src_x, dst_x), edge_attr=edge_attr,
                                 edge_type=edge_type,
                                 size=(src_x.shape[0], dst_x.shape[0]))
            
            node_dict[dst_type].append(out.view(-1, self.heads * self.out_channels))

        # 所有有向边特征求和
        for node_type, outs in node_dict.items():
            out = torch.stack(outs)
            out_dict[node_type] = out.sum(dim = 0)  + res_dict[node_type]  + self.bias[node_type]
        
        # 这里依次遍历所有边
        for edge_type, edge_index in edge_index_dict.items():
            # 这里设置为边特征
            edge_attr = edge_attr_dict[edge_type]
            src_type, _, dst_type = edge_type
            src_x = out_dict[src_type].view(-1, self.heads, self.out_channels).mean(dim =1)
            dst_x = out_dict[dst_type].view(-1, self.heads, self.out_channels).mean(dim =1)
            edge_attr_dict[edge_type] = self.edge_up['__'.join(edge_type)](torch.cat((src_x[edge_index[0], :], edge_attr, dst_x[edge_index[1], :]), dim=1))
        
        return out_dict, edge_attr_dict

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr, edge_type, index: Tensor) -> Tensor:

        edge_attr = self.lin_edge[edge_type](edge_attr).view(-1, self.heads, self.out_channels)
        x = x_i + x_j + edge_attr
        x = F.leaky_relu(x, 0)

        alpha = (x * self.att[edge_type]).sum(dim=-1)
        alpha = softmax(alpha, index)

        alpha = F.dropout(alpha, p=0, training=self.training)
        return x_j * alpha.unsqueeze(-1)

