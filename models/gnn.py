import torch
from complexPyTorch.complexLayers import NaiveComplexBatchNorm1d
from munch import Munch
from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphNorm
from torch_geometric.nn.norm import BatchNorm
import torch_geometric.nn.norm.instance_norm
from models.layers import ComplexLinear, ELU, EdgeGATConv, MM_ACT, HGTConv, HeteroDictComLinear, MM_Dict_ACT, DictELU, \
    DictComBN, BipGATConv, MM_ACTv2, GATv2Conv, MM_ACTv3
from torch import nn
from torch.nn import Linear, BatchNorm1d, SELU, LeakyReLU, RReLU, PReLU
from torch_geometric.utils import to_undirected
from torch_geometric.nn.conv import GATv2Conv as GATConv

from solver.utils import unbatch_mm


# 直接学习beam
class GAT(nn.Module):
    def __init__(self, args):
        super(GAT, self).__init__()
        self.gat1 = EdgeGATConv(args.antenna_num, 16, 40, negative_slope=0, residual=True, edge_dim=args.edge_dim)
        self.gat2 = EdgeGATConv(16 * 40, 32, 40, negative_slope=0, residual=True, edge_dim=args.edge_dim)
        self.gat3 = EdgeGATConv(32 * 40, 64, 40, negative_slope=0, residual=True, edge_dim=args.edge_dim)
        self.relu = ELU()
        self.lin1 = ComplexLinear(64 * 40, 1024)
        self.bn1 = NaiveComplexBatchNorm1d(1024)
        self.lin2 = ComplexLinear(1024, 512)
        self.bn2 = NaiveComplexBatchNorm1d(512)

        self.lin3 = ComplexLinear(512, args.antenna_num)
        self.lin4 = ComplexLinear(512, 2)
        self.lin5 = ComplexLinear(512, 1)
        self.args = args
        # 初始化激活函数层
        self.act = MM_ACTv2(args)
        self.user_num = args.user_num

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr
        edge_attr = edge_attr.to(torch.complex64)
        x = x.to(torch.complex64)

        x = self.gat1(x, edge_index, edge_attr)
        x = self.relu(x)

        x = self.gat2(x, edge_index, edge_attr)
        x = self.relu(x)

        x = self.gat3(x, edge_index, edge_attr)
        x = self.relu(x)

        x = self.lin1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.lin2(x)
        x = self.bn2(x)
        x = self.relu(x)

        RF = self.lin3(x)
        BB = self.lin4(x)
        BB1 = BB[:, 0].reshape(-1, self.user_num, 1)
        BB2 = BB[:, 1].reshape(-1, 1, self.user_num)
        BB = torch.bmm(BB1, BB2)
        P = self.lin5(x)

        return self.act(RF, BB, P)


# 直接学习beam
class GATReal(nn.Module):
    def __init__(self, args):
        super(GATReal, self).__init__()
        self.gat1 = GATv2Conv(args.antenna_num * 2, 16, 40, negative_slope=0, residual=True, edge_dim=args.edge_dim)
        self.gat2 = GATv2Conv(16 * 40, 32, 40, negative_slope=0, residual=True, edge_dim=args.edge_dim)
        self.gat3 = GATv2Conv(32 * 40, 64, 40, negative_slope=0, residual=True, edge_dim=args.edge_dim)
        self.relu = SELU()
        self.lin1 = Linear(64 * 40, 1024)
        self.bn1 = BatchNorm1d(1024)
        self.lin2 = Linear(1024, 512)
        self.bn2 = BatchNorm1d(512)
        self.lin3 = Linear(512, args.antenna_num)
        # 这里就
        self.lin4 = Linear(512, 4)
        self.lin5 = Linear(512, 1)
        self.args = args
        # 初始化激活函数层
        self.act = MM_ACTv3(args)

        self.user_num = args.user_num

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr
        edge_attr = edge_attr.to(torch.float)
        x = x.to(torch.float)
        batch = data.batch

        x = self.gat1(x, edge_index, edge_attr)
        x = self.relu(x)

        x = self.gat2(x, edge_index, edge_attr)
        x = self.relu(x)

        x = self.gat3(x, edge_index, edge_attr)
        x = self.relu(x)

        x = self.lin1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.lin2(x)
        x = self.bn2(x)
        x = self.relu(x)

        RF = self.lin3(x)
        BB = self.lin4(x)
        P = self.lin5(x)
        # 进行一个 unbatch 操作
        dicts, users = unbatch_mm(RF, BB, P, data.x)
        dicts = self.act(dicts, users)
        return dicts, users


# 直接学习beam
class BipGAT(nn.Module):
    def __init__(self, args):
        super(BipGAT, self).__init__()
        self.bipGAT1 = BipGATConv(in_channels=(args.user_num, args.user_num), out_channels=32, heads=10, edge_dim=1)
        self.bipGAT2 = BipGATConv(in_channels=(32 * 10, 32 * 10), out_channels=64, heads=10, edge_dim=1)
        self.bipGAT3 = BipGATConv(in_channels=(64 * 10, 64 * 10), out_channels=128, heads=10, edge_dim=1)
        self.lin1 = HeteroDictComLinear(in_channels={'ant': 128 * 10, 'user': 128 * 10},
                                        out_channels={'ant': 512, 'user': 512})
        self.bn1 = DictComBN(in_channels={'ant': 512, 'user': 512})
        self.lin2 = HeteroDictComLinear(in_channels={'ant': 512, 'user': 512},
                                        out_channels={'ant': 256, 'user': 256})
        self.bn2 = DictComBN(in_channels={'ant': 256, 'user': 256})

        self.lin3 = HeteroDictComLinear(in_channels={'ant': 256, 'user': 256},
                                        out_channels={'ant': args.user_num, 'user': args.user_num + 1})

        # 初始化激活函数层
        self.relu = DictELU()
        self.selu = ELU()
        self.act = MM_Dict_ACT(args)
        self.args = args

        # 生成子图的边

    def generate_graph_edges(self, N, K):
        edge_index = []

        # Generate edges for each graph
        for i in range(N):
            start_node = i * K

            # Generate edges within the graph
            for j in range(K):
                for k in range(K):
                    if j != k:
                        edge_index.append([start_node + j, start_node + k])

            # Generate self-loop edges within the graph
            for j in range(K):
                edge_index.append([start_node + j, start_node + j])

        # Convert to PyG format
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_index = to_undirected(edge_index)
        return edge_index

    def forward(self, data):
        x_src, x_dst, edge_index, edge_attr = data.x_s, data.x_t, data.edge_index, data.edge_attr
        x_src = x_src.to(torch.complex64)
        x_dst = x_dst.to(torch.complex64)
        edge_attr = edge_attr.to(torch.complex64)
        # # print(self.args.batch_size,self.args.user_num)
        # user_edge_index = self.generate_graph_edges(self.args.batch_size , self.args.user_num).to('cuda')
        # antenna_edge_index = self.generate_graph_edges(self.args.batch_size , self.args.antenna_num).to('cuda')

        # 先经历过一系列GNN
        out_s, out_t = self.bipGAT1((x_src, x_dst), edge_index, edge_attr)
        out_s, out_t = self.selu(out_s), self.selu(out_t)

        out_s, out_t = self.bipGAT2((out_s, out_t), edge_index, edge_attr)
        out_s, out_t = self.selu(out_s), self.selu(out_t)

        out_s, out_t = self.bipGAT3((out_s, out_t), edge_index, edge_attr)
        out_s, out_t = self.selu(out_s), self.selu(out_t)
        x_dict = {'ant': out_s, 'user': out_t}

        x_dict = self.lin1(x_dict)
        x_dict = self.relu(x_dict)
        x_dict = self.bn1(x_dict)

        x_dict = self.lin2(x_dict)
        x_dict = self.relu(x_dict)
        x_dict = self.bn2(x_dict)

        x_dict = self.lin3(x_dict)

        return self.act(x_dict)


class HetGAT(nn.Module):
    def __init__(self, metadata, args):
        super(HetGAT, self).__init__()
        self.han1 = HGTConv(in_channels={'ant': args.user_num, 'user': args.user_num}, out_channels=16,
                            metadata=metadata, heads=10, edge_dim=1)
        self.han2 = HGTConv(in_channels=160, out_channels=32, metadata=metadata, heads=10, edge_dim=1)
        self.lin1 = HeteroDictComLinear(in_channels={'ant': 320, 'user': 320}, out_channels={'ant': 512, 'user': 512})
        self.bn1 = DictComBN(in_channels={'ant': 512, 'user': 512})
        self.lin2 = HeteroDictComLinear(in_channels={'ant': 512, 'user': 512},
                                        out_channels={'ant': args.user_num, 'user': args.user_num + 1})

        # 初始化激活函数层
        self.relu = DictELU()
        self.act = MM_Dict_ACT(args)

    def forward(self, data):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict
        x_dict = self.han1(x_dict, edge_index_dict, edge_attr_dict)
        x_dict = self.relu(x_dict)
        x_dict = self.han2(x_dict, edge_index_dict, edge_attr_dict)
        x_dict = self.relu(x_dict)

        x_dict = self.lin1(x_dict)
        x_dict = self.relu(x_dict)
        x_dict = self.bn1(x_dict)
        x_dict = self.lin2(x_dict)
        return self.act(x_dict)


class CNN_v3(nn.Module):
    def __init__(self, args):
        super(CNN_v3, self).__init__()
        # 用户数
        self.K = args.user_num
        # 天线数
        self.Nt = args.antenna_num

        self.batch_size = args.batch_size

        self.RF_dims = 2 * self.K * self.Nt
        self.BB_dims = 2 * self.K * self.K
        self.P_dims = self.K
        # Output_size = ((Input_size−Kernel_size+2×Padding)/Stride) + 1

        self.conv1 = nn.Sequential(  # input shape ( 3, K, Nt)
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # output shape (16, K, Nt)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=3, stride=(2, 1)),  # choose max value in 2x2 area, output shape (16, K/2, Nt/2)
            nn.BatchNorm2d(16)
        )
        self.conv2 = nn.Sequential(  # input shape (16, K/2, Nt/2)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            # output shape (32, K/2, Nt/2)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=3, stride=(2, 1)),  # output shape (32, K/4, Nt/4)
            nn.BatchNorm2d(32)
        )
        self.conv3 = nn.Sequential(  # input shape (32, K/4, Nt/4)
            nn.Conv2d(32, 64, 3, 1, 1),  # output shape (64, K/4, Nt/4)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=3, stride=(2, 1)),  # output shape (64, K/8, Nt/8) (64,1,2)
            nn.BatchNorm2d(64)
        )
        fc1_hidden1 = 64 * 2
        fc1_hidden2, fc1_hidden3 = 512, 256
        self.fc1 = nn.Sequential(
            nn.Linear(fc1_hidden1, fc1_hidden2),
            nn.ReLU(),
            nn.BatchNorm1d(fc1_hidden2),
            nn.Linear(fc1_hidden2, fc1_hidden3),
            nn.ReLU(),
            nn.BatchNorm1d(fc1_hidden3)
        )

        self.RFOut = nn.Linear(fc1_hidden3, self.RF_dims)
        self.BBOut = nn.Linear(fc1_hidden3, self.BB_dims)
        self.Pout = nn.Linear(fc1_hidden3, self.P_dims)

    def forward(self, x_csi):
        x = self.conv1(x_csi)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 64 * 2)
        x = self.fc1(x)

        RF = self.RFOut(x)  # output RF
        RF = RF.reshape(-1, self.K, 2 * self.Nt)
        RF = torch.complex(RF[:, :self.Nt], RF[:, :2 * self.Nt])

        BB = self.BBOut(x)  # output digital precoder
        BB = BB.reshape(-1, self.K, 2 * self.K)
        BB = torch.complex(BB[:, :self.K], BB[:, :2 * self.K])

        P = self.Pout(x).reshape(self.batch_size, self.user_num, 1)  # return feature_other and phase
        return RF, BB, P


class GAT(nn.Module):
    def __init__(self, args):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_channels=args.antenna_num * 2, out_channels=32, heads=40, negative_slope=0,
                            residual=True, edge_dim=args.edge_dim)
        self.gbn1 = GraphNorm(32 * 40)

        self.gat2 = GATConv(32 * 40, 64, 40, negative_slope=0, residual=True, edge_dim=args.edge_dim)
        self.gbn2 = GraphNorm(64 * 40)

        self.gat3 = GATConv(64 * 40, 128, 40, negative_slope=0, residual=True, edge_dim=args.edge_dim)
        self.gbn3 = GraphNorm(128 * 40)

        self.relu = ReLU()

        self.lin1 = Linear(128 * 40, 1024)
        self.bn1 = GraphNorm(1024)

        self.lin2 = Linear(1024, 512)
        self.bn2 = GraphNorm(512)

        self.RFOut = Linear(512, args.antenna_num * 2)
        # 这里就
        self.BBOut = Linear(1030, 2)

        self.POut = Linear(512, 1)
        self.args = args
        # 初始化激活函数层
        self.K = args.user_num
        self.batch_size = args.batch_size
        self.Nt = args.antenna_num
        self.p_max = args.p_max
        self.act = MM_ACTv3(args)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr
        edge_attr = edge_attr.to(torch.float)
        x = x.to(torch.float)

        x = self.gat1(x, edge_index, edge_attr)
        x = self.gbn1(x)
        x = self.relu(x)

        x = self.gat2(x, edge_index, edge_attr)
        x = self.gbn2(x)
        x = self.relu(x)

        x = self.gat3(x, edge_index, edge_attr)
        x = self.gbn3(x)
        x = self.relu(x)

        x = self.lin1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.lin2(x)
        x = self.bn2(x)
        x = self.relu(x)

        RF = self.RFOut(x)
        # 这里对RF 进行归一化操作
        RF = torch.complex(RF[:, : self.Nt], RF[:, self.Nt:])
        RF = RF.reshape(-1, self.K, self.Nt)

        # 首先根据有向边获得特征
        x_edge = torch.cat((x[edge_index[0]], edge_attr, x[edge_index[1]]), dim=1)
        BB = self.BBOut(x_edge)
        BB = torch.complex(BB[:, 0], BB[:, 1]).reshape(-1, self.K, self.K)

        # 强制用完
        P = self.POut(x).reshape(self.batch_size, self.K, 1)

        return self.act(RF, BB, P)


if __name__ == '__main__':
    pass
    # train_dataset = BipartiteDataset('../dataset/8u_16n_40w/val/')
    # loaders = DataLoader(dataset=train_dataset, batch_size=6, shuffle=True, num_workers=1)
    # data = train_dataset[0]
    # args = {'user_num': 8, 'antenna_num': 16, 'p_max': 1}
    # args = Munch(args)
    # model = BipGAT(data.metadata(), args)
    # for batch in loaders:
    #     print(batch.metadata())
    #     x, y, z = model(batch)
    #     print(x.shape)
    #     print(y.shape)
    #     print(z.shape)
    #     break

# print(data.metadata())
# model = BipGAT(data.metadata())
# for node_type, x in model(data).items():
#     print(node_type)
#     print(x.shape)
