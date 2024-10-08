import torch
from torch_geometric.nn.norm import BatchNorm,GraphNorm
from models.layers import GATv2Conv ,HGTConvV2 ,DictELUV2, MM_ACTv3 , NodeDictBN ,EdgeDictBN , ComplexToReal ,GATv3Conv
from torch import nn
from torch.nn import  Linear ,ReLU
from torch_geometric.nn.conv import GCN2Conv
from torch_geometric.nn.conv import GATv2Conv as GATConv


class GATReal(nn.Module):
    def __init__(self, args):
        super(GATReal, self).__init__()
        self.gat1 = GATv2Conv(args.antenna_num * 2, 32, 40, negative_slope=0, residual=True, edge_dim=args.edge_dim)
        self.gbn1 = GraphNorm(32*40)
   
        self.gat2 = GATv2Conv(32*40 , 64, 40, negative_slope=0, residual=True, edge_dim=args.edge_dim)
        self.gbn2 = GraphNorm(64*40)

        self.gat3 = GATv2Conv(64*40 , 128, 40, negative_slope=0, residual=True, edge_dim=args.edge_dim)
        self.gbn3 = GraphNorm(128*40)

        self.relu = ReLU()

        self.lin1 = Linear(128*40, 1024)
        self.bn1 = GraphNorm(1024)

        self.lin2 = Linear(1024, 512)
        self.bn2 = GraphNorm(512)

        self.RFOut = Linear(512, args.antenna_num * 2)
        # 这里就
        self.BBOut = Linear(1030,2)
        # self.EMLP = MLPBN(6,512)
        # self.BBOut = Linear(256,2)


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
        RF = torch.complex(RF[:,: self.Nt], RF[:,self.Nt:])
        RF = RF.reshape(-1, self.K, self.Nt)

        # 首先根据有向边获得特征
        x_edge = torch.cat((x[edge_index[0]],edge_attr,x[edge_index[1]]),dim = 1)
        BB = self.BBOut(x_edge)
        BB = torch.complex(BB[:,0], BB[:,1]).reshape(-1,self.K,self.K)
        # edge_attr = self.EMLP(edge_attr)
        # BB = self.BBOut(edge_attr)
        # BB = torch.complex(BB[:,0], BB[:,1]).reshape(-1,self.K,self.K)
        
        # 强制用完
        P = self.POut(x).reshape(self.batch_size,self.K,1)

        return self.act(RF, BB , P)


class GATBN(nn.Module):
    def __init__(self, in_channels, out_channels, heads,edge_dim,edge_out):
        super(GATBN, self).__init__()

        self.gat= GATv3Conv(in_channels, out_channels, heads, negative_slope=0, residual=True, edge_dim=edge_dim,edge_out=edge_out)
        # 设置节点BN
        self.nbn= GraphNorm(heads*out_channels)
        # 设置边BN
        self.ebn= GraphNorm(edge_out)

        # 初始化激活函数层
        self.relu = ReLU()

    def forward(self, x, edge_index, edge_attr):

        x,edge_attr = self.gat(x, edge_index, edge_attr)
        x = self.nbn(x)
        x = self.relu(x)

        edge_attr = self.ebn(edge_attr) 
        edge_attr = self.relu(edge_attr) 

        return x,edge_attr

class GATV3(nn.Module):
    def __init__(self, args):
        super(GATV3, self).__init__()
        self.gatBn1 = GATBN(args.antenna_num * 2, 32, 40,edge_dim=args.edge_dim,edge_out=256)
        self.gatBn2 = GATBN(32*40, 64, 40,edge_dim=256,edge_out=512)
        self.gatBn3 = GATBN(64*40, 128, 40,edge_dim=512,edge_out=1024)

        # 解码器
        self.NMlp = MLPBN(128*40,1024)
        self.EMLP = MLPBN(1024,512)

        self.RFOut = Linear(512, args.antenna_num * 2)
        self.POut = Linear(512, 1)

        # 这里就
        self.BBOut = Linear(256,2)

   
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

        x,edge_attr = self.gatBn1(x, edge_index, edge_attr)
        x,edge_attr = self.gatBn2(x, edge_index, edge_attr)
        x,edge_attr = self.gatBn3(x, edge_index, edge_attr)


        x = self.NMlp(x)



        RF = self.RFOut(x)
        # 这里对RF 进行归一化操作
        RF = torch.complex(RF[:,: self.Nt], RF[:,self.Nt:])
        RF = RF.reshape(-1, self.K, self.Nt)

        P = self.POut(x).reshape(self.batch_size,self.K,1)

        edge_attr = self.EMLP(edge_attr)
        BB = self.BBOut(edge_attr)
        BB = torch.complex(BB[:,0], BB[:,1]).reshape(-1,self.K,self.K)
        


        return self.act(RF, BB , P)

class EmbeddingBN(nn.Module):
    def __init__(self,out_channels, metadata,in_channels = 1):
        super(EmbeddingBN, self).__init__()

        self.embedding = nn.ModuleDict({key: ComplexToReal(in_channels, out_channels)for key in metadata[0]})
        self.BN = nn.ModuleDict({key: GraphNorm(out_channels)for key in metadata[0]})
        self.out_channels = out_channels

    def forward(self, x_dict):

        for node_type, x in x_dict.items():
            # 记录一下原始维度
            dim = x.shape[1]
            x = x.view(-1, 1)
            x = self.embedding[node_type](x)
            # 这里需要进行判断一下
            x = x.view(-1, dim, self.out_channels)
            x = torch.sum(x, dim=1)
            x = self.BN[node_type](x)
            x_dict[node_type] = x
        return x_dict

class HGATBN(nn.Module):
    def __init__(self, in_channels, out_channels, ext_edge_in_channels, edge_out_channels, metadata, heads, inter_edge_in_channels):
        super(HGATBN, self).__init__()

        self.han= HGTConvV2(in_channels=in_channels, out_channels=out_channels,ext_edge_in_channels = ext_edge_in_channels,edge_out_channels=edge_out_channels,
                    metadata=metadata, heads=heads, inter_edge_in_channels=inter_edge_in_channels)
        # 设置节点BN
        self.nbn= NodeDictBN(in_channels = heads*out_channels,metadata = metadata)
        # 设置边BN
        self.ebn= EdgeDictBN(in_channels = edge_out_channels,metadata = metadata)

        # 初始化激活函数层
        self.relu = DictELUV2()

    def forward(self, x_dict,edge_index_dict,edge_attr_dict):

        x_dict,edge_attr_dict = self.han(x_dict, edge_index_dict, edge_attr_dict)
        x_dict = self.nbn(x_dict)
        x_dict = self.relu(x_dict)

        edge_attr_dict = self.ebn(edge_attr_dict) 
        edge_attr_dict = self.relu(edge_attr_dict) 

        return x_dict,edge_attr_dict

class MLPBN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLPBN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.line1 = Linear(in_channels,out_channels)
        self.bn1 = GraphNorm(out_channels)
        self.line2 = Linear(out_channels,out_channels // 2)
        self.bn2 = GraphNorm(out_channels // 2)
        self.relu = ReLU()


    def forward(self, x):
        x = self.line1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.line2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class HetGATV2(nn.Module):
    def __init__(self, metadata, args):
        super(HetGATV2, self).__init__()

        self.node_embeding = EmbeddingBN(out_channels = 128 , metadata = metadata) 
        # 编码器
        self.hanBn1 = HGATBN(in_channels=128, out_channels=32,ext_edge_in_channels = 2,edge_out_channels=256,metadata=metadata, heads=40, inter_edge_in_channels=6)
        # self.hanBn1 = HGATBN(in_channels={'ant': 32, 'user': 48}, out_channels=32,ext_edge_in_channels = 2,edge_out_channels=256,metadata=metadata, heads=40, inter_edge_in_channels=6)
        self.hanBn2 = HGATBN(in_channels=32*40, out_channels=64,ext_edge_in_channels = 256,edge_out_channels=512, metadata=metadata, heads=40, inter_edge_in_channels=256)
        self.hanBn3 = HGATBN(in_channels=64*40, out_channels=128,ext_edge_in_channels = 512,edge_out_channels=1024, metadata=metadata, heads=40, inter_edge_in_channels=512)

        # 解码器
        self.PMlp = MLPBN(128*40,1024)
        self.POut = Linear(512,1)

        self.RFMlp = MLPBN(1024,512)
        self.RFOut = Linear(256,2)

        self.BBMlp = MLPBN(1024,512)
        self.BBOut = Linear(256 ,2)

        self.user_num = args.user_num
        self.antenna_num = args.antenna_num
        self.batch_size = args.batch_size
        self.p_max = args.p_max

        self.act = MM_ACTv3(args)


    def forward(self, data):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict

        # 进行节点嵌入
        x_dict = self.node_embeding(x_dict)

        x_dict,edge_attr_dict = self.hanBn1(x_dict, edge_index_dict, edge_attr_dict)
        x_dict,edge_attr_dict = self.hanBn2(x_dict, edge_index_dict, edge_attr_dict)
        x_dict,edge_attr_dict = self.hanBn3(x_dict, edge_index_dict, edge_attr_dict)


        user_x = x_dict["user"]
        user_x = self.PMlp(user_x)
        P = self.POut(user_x).reshape(self.batch_size,self.user_num,1)


        user_edge_attr = edge_attr_dict["user","u2u","user"]
        user_edge_attr = self.BBMlp(user_edge_attr)

        BB = self.BBOut(user_edge_attr)
        BB = torch.complex(BB[:,0], BB[:,1]).reshape(-1,self.user_num,self.user_num)


        edge_attr = edge_attr_dict["user","u2a","ant"] +  edge_attr_dict["ant","a2u","user"] 
        edge_attr = self.RFMlp(edge_attr)
        RF = self.RFOut(edge_attr)
        RF =  torch.complex(RF[:,0], RF[:,1]).reshape(-1,self.user_num,self.antenna_num) 

        return self.act(RF,BB,P)


class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
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
            nn.MaxPool2d(kernel_size=2, stride=(2, 2)),  # choose max value in 2x2 area, output shape (16, K/2, Nt/2)
            nn.BatchNorm2d(16)
        )
        self.conv2 = nn.Sequential(  # input shape (16, K/2, Nt/2)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            # output shape (32, K/2, Nt/2)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2, stride=(2, 2)),  # output shape (32, K/4, Nt/4)
            nn.BatchNorm2d(32)
        )
        self.conv3 = nn.Sequential(  # input shape (32, K/4, Nt/4)
            nn.Conv2d(32, 64, 3, 1, 1),  # output shape (64, K/4, Nt/4)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2, stride=(2, 2)),  # output shape (64, K/8, Nt/8) (64,1,2)
            nn.BatchNorm2d(64)
        )
        fc1_hidden1 = 128
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

        self.act = MM_ACTv3(args)

    def forward(self, x_csi):
        x = self.conv1(x_csi)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.conv3(x)
        
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 64 * 2)

        x = self.fc1(x)

        RF = self.RFOut(x)  # output RF
        RF = RF.reshape(-1, self.K, 2 * self.Nt)
        RF = torch.complex(RF[:,:, :self.Nt], RF[:,:, self.Nt:2 * self.Nt])
        RF = RF.reshape(-1,self.K,self.Nt) 


        BB = self.BBOut(x)  # output digital precoder
        BB = BB.reshape(-1, self.K, 2 * self.K)
        BB = torch.complex(BB[:,:, :self.K], BB[:,:, self.K:2 * self.K])
        BB = BB.reshape(-1,self.K,self.K)

        P = self.Pout(x).reshape(self.batch_size, self.K, 1)  # return feature_other and phase
        return self.act(RF,BB,P)

class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        # 用户数
        self.K = args.user_num
        # 天线数
        self.Nt = args.antenna_num

        self.batch_size = args.batch_size

        self.RF_dims = 2 * self.K * self.Nt
        self.BB_dims = 2 * self.K * self.K
        self.P_dims = self.K

        self.fc1 = nn.Sequential(
            nn.Linear(2 * self.K * self.Nt, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )

        self.RFOut = nn.Linear(256, self.RF_dims)
        self.BBOut = nn.Linear(256, self.BB_dims)
        self.Pout = nn.Linear(256, self.P_dims)

        self.act = MM_ACTv3(args)

    def forward(self, x_csi):
        x = x_csi.reshape(self.batch_size,-1)
        x = self.fc1(x)
        x = self.fc2(x)

        RF = self.RFOut(x)  # output RF
        RF = RF.reshape(-1, self.K, 2 * self.Nt)
        RF = torch.complex(RF[:,:, :self.Nt], RF[:,:, self.Nt:2 * self.Nt])
        RF = RF.reshape(-1,self.K,self.Nt) 


        BB = self.BBOut(x)  # output digital precoder
        BB = BB.reshape(-1, self.K, 2 * self.K)
        BB = torch.complex(BB[:,:, :self.K], BB[:,:, self.K:2 * self.K])
        BB = BB.reshape(-1,self.K,self.K)

        P = self.Pout(x).reshape(self.batch_size, self.K, 1)  # return feature_other and phase
        return self.act(RF,BB,P)

class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()

        # 用户数
        self.K = args.user_num
        # 天线数
        self.Nt = args.antenna_num

        self.batch_size = args.batch_size

        self.RF_dims = 2 * self.Nt
        self.BB_dims = 2 * self.K
        self.P_dims = 1

        self.lin_res = Linear(self.Nt * 2, 512)

        self.gcn1 = GCN2Conv(512, alpha=0.1, theta=0.4, layer=1, shared_weights=False)
        self.gbn1 = GraphNorm(512)

        self.gcn2 = GCN2Conv(512, alpha=0.1, theta=0.4, layer=2, shared_weights=False)
        self.gbn2 = GraphNorm(512)

        self.gcn3 = GCN2Conv(512, alpha=0.1, theta=0.4, layer=3, shared_weights=False)
        self.gbn3 = GraphNorm(512)

        fc1_hidden1 = 512
        fc1_hidden2, fc1_hidden3 = 256, 128
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

        self.act = MM_ACTv3(args)
        self.relu = ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.lin_res(x)
        x = self.relu(x)
        tem = x

        x = self.gcn1(x, tem, edge_index)
        x = self.gbn1(x)
        x = self.relu(x)

        x = self.gcn2(x, tem, edge_index)
        x = self.gbn2(x)
        x = self.relu(x)

        x = self.gcn3(x, tem, edge_index)
        x = self.gbn3(x)
        x = self.relu(x)

        x = self.fc1(x)

        RF = self.RFOut(x)
        # 这里对RF 进行归一化操作
        RF = torch.complex(RF[:,: self.Nt], RF[:,self.Nt:])
        RF = RF.reshape(-1,self.K,self.Nt) 


        BB = self.BBOut(x)  # output digital precoder
        BB = torch.complex(BB[:,: self.K], BB[:,self.K:])
        BB = BB.reshape(-1,self.K,self.K)

        P = self.Pout(x).reshape(self.batch_size, self.K, 1)  
        return self.act(RF,BB,P)

class GAT(nn.Module):
    def __init__(self, args):
        super(GAT, self).__init__()

        self.K = args.user_num
        self.batch_size = args.batch_size
        self.Nt = args.antenna_num
        self.p_max = args.p_max

        self.gat1 = GATv2Conv(in_channels=args.antenna_num * 2, out_channels=32, heads=40, negative_slope=0,
                            residual=True, edge_dim=args.edge_dim)
        self.gbn1 = GraphNorm(32 * 40)

        self.gat2 = GATv2Conv(32 * 40, 64, 40, negative_slope=0, residual=True, edge_dim=args.edge_dim)
        self.gbn2 = GraphNorm(64 * 40)

        self.gat3 = GATv2Conv(64 * 40, 128, 40, negative_slope=0, residual=True, edge_dim=args.edge_dim)
        self.gbn3 = GraphNorm(128 * 40)

        self.relu = ReLU()

        self.lin1 = Linear(128 * 40, 1024)
        self.bn1 = GraphNorm(1024)

        self.lin2 = Linear(1024, 512)
        self.bn2 = GraphNorm(512)

        self.RFOut = Linear(512, self.Nt * 2)
        # 这里就
        self.BBOut = Linear(512, self.K * 2)

        self.POut = Linear(512, 1)
        self.args = args
        # 初始化激活函数层
 
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

        BB = self.BBOut(x)
        BB = torch.complex(BB[:, : self.K], BB[:, self.K:])
        BB = BB.reshape(-1, self.K, self.K)
        
        # 强制用完
        P = self.POut(x).reshape(self.batch_size, self.K, 1)

        return self.act(RF, BB, P)

if __name__ == '__main__':
    pass