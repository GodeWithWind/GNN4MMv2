import numpy as np
import scipy.io as scio
import networkx as nx
import torch
import math
from scipy.sparse import coo_matrix
from torch_geometric.data import Data, InMemoryDataset, HeteroData
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import Dataset ,DataLoader as normDL
import os

class HomDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(HomDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['H.mat']

    @property
    def processed_file_names(self):
        return ['data.pt']

    # 处理数据
    def process(self):
        # 读取原始数据
        dataset = scio.loadmat(self.raw_paths[0])
        num = dataset["H_gather"].shape[0]
        print('数据集大小:{}'.format(num))
        tem = dataset["H_gather"]
        # 获取用户数和天线数
        K = dataset["H_gather"].shape[1]
        data_list = []
        # 依次处理所有数据
        for index in range(num):
            adj = np.zeros([K, K])
            for i in range(K):
                for j in range(K):
                    adj[i, j] = 1

            adj = coo_matrix(adj)
            indices = np.vstack((adj.row, adj.col))  # 我们真正需要的coo形式
            edge_index = torch.LongTensor(indices)  # PyG框架需要的coo形式
            
            h = torch.from_numpy(dataset["H_gather"][index])
            # 这里进行特征处理
            h = h / math.sqrt(1.99e-12)
            # rf = torch.from_numpy(dataset["RF"][index])
            # bb = torch.from_numpy(dataset["BB"][index])
            # sr = torch.from_numpy(dataset["SR"][index])

            edge1 = torch.einsum('in,in -> i', h.conj()[edge_index[0], :], h[edge_index[1], :]).reshape(
                edge_index.shape[1], -1)
            edge2 = torch.einsum('in,in -> i', h.conj()[edge_index[0], :], h[edge_index[0], :]).reshape(
                edge_index.shape[1], -1)
            edge3 = torch.einsum('in,in -> i', h.conj()[edge_index[1], :], h[edge_index[1], :]).reshape(
                edge_index.shape[1], -1)

            edge_attr = torch.cat((edge2.real, edge1.real, edge3.real, edge2.imag, edge1.imag, edge3.imag), dim=1)

            graph_data = Data(edge_index=edge_index, x=torch.cat((h.real, h.imag), dim=1), y=torch.zeros([1], dtype=torch.float),
                              edge_attr=edge_attr)
            data_list.append(graph_data)
        data_save, data_slices = self.collate(data_list)
        torch.save((data_save, data_slices), self.processed_paths[0])

class HetDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(HetDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['data.mat']

    @property
    def processed_file_names(self):
        return ['data.pt']

    # 处理数据
    def process(self):
        # 读取原始数据
        # 读取原始数据
        dataset = scio.loadmat(self.raw_paths[0])
        num = dataset["H"].shape[0]
        print('数据集大小:{}'.format(num))
        tem = dataset["H"]
        # 获取用户数和天线数
        K = dataset["H"].shape[1]
        Nt = dataset["H"].shape[2]

        # 获得天线节点边
        ant_adj = np.ones([Nt, Nt])
        ant_adj = coo_matrix(ant_adj)
        ant_indices = np.vstack((ant_adj.row, ant_adj.col))
        ant_index = torch.LongTensor(ant_indices)

        # 获得用户节点边
        user_adj = np.ones([K, K])
        user_adj = coo_matrix(user_adj)
        user_indices = np.vstack((user_adj.row, user_adj.col))
        user_index = torch.LongTensor(user_indices)

        # 获得天线到用户节点边
        ant_start_index = []
        user_end_index = []
        for s in range(Nt):
            for e in range(K):
                ant_start_index.append(s)
                user_end_index.append(e)
        # 转化为pyg形式
        ant_start_index = torch.from_numpy(np.asarray(ant_start_index)).to(torch.long)
        user_end_index = torch.from_numpy(np.asarray(user_end_index)).to(torch.long)
        ant_edge_index = torch.stack([ant_start_index, user_end_index], dim=0)

        # 设置从用户到天线边
        user_start_index = []
        ant_end_index = []
        for s in range(K):
            for e in range(Nt):
                user_start_index.append(s)
                ant_end_index.append(e)
        # 转化为pyg形式
        user_start_index = torch.from_numpy(np.asarray(user_start_index)).to(torch.long)
        ant_end_index = torch.from_numpy(np.asarray(ant_end_index)).to(torch.long)
        user_edge_index = torch.stack([user_start_index, ant_end_index], dim=0)
        # print(user_edge_index)
        # 处理后的数据
        data_list = []
        for index in range(num):
            # 获得用户信道
            h = torch.from_numpy(dataset["H"][index])
            # 这里进行特征处理，噪声归一化
            h = h / math.sqrt(1.99e-12)

            # snr_linear = 10 ** (-15 / 10)
            # signal_power = torch.norm(h, p=2, dim=1, keepdim=True) ** 2
            # noise_power = signal_power * snr_linear
            # # print("te")
            # noise = torch.randn(size=h.shape, dtype=torch.complex64)
            # noise = noise / torch.norm(noise, p=2, dim=1, keepdim=True)
            # noise = noise * torch.sqrt(noise_power)
            # oH = h
            # h = h + noise

            # 这里分别设置有噪声的h和原始H
            # rf = torch.from_numpy(dataset["RF"][index])
            # bb = torch.from_numpy(dataset["BB"][index])
            # sr = torch.from_numpy(dataset["SR"][index])

            # print(h.shape)
            graph_data = HeteroData()

            graph_data.y = torch.zeros([1], dtype=torch.float)
            # graph_data.oH = oH
            # graph_data.BB = bb

            # 获得天线节点特征
            ant_edge_attr1 = torch.einsum('in,in -> i', h.T.conj()[ant_index[0], :], h.T[ant_index[1], :]) \
                .reshape(ant_index.shape[1], -1)
            ant_edge_attr2 = torch.einsum('in,in -> i', h.T.conj()[ant_index[0], :], h.T[ant_index[0], :]) \
                .reshape(ant_index.shape[1], -1)
            ant_edge_attr3 = torch.einsum('in,in -> i', h.T.conj()[ant_index[1], :], h.T[ant_index[1], :]) \
                .reshape(ant_index.shape[1], -1)
            ant_edge_attr = torch.cat((ant_edge_attr2, ant_edge_attr1, ant_edge_attr3), dim=1)

            # 设置天线节点特征和有向边特征
            graph_data['ant'].x = torch.cat((h.T.real, h.T.imag), dim=1)
            # 此时ant 节点特征shape [n_t , 2K] , 填充到指定维度 填充到32
            graph_data['ant'].x = F.pad(graph_data['ant'].x, pad=(0, 32 - 2*K), mode='constant', value=0)
            # graph_data['ant'].x = h.T
            graph_data['ant', 'a2a', 'ant'].edge_index = ant_index
            graph_data['ant', 'a2a', 'ant'].edge_attr = torch.cat((ant_edge_attr.real, ant_edge_attr.imag), dim=1)
            # graph_data['ant', 'a2a', 'ant'].edge_attr = ant_edge_attr


            # 获得用户节点特征
            user_edge_attr1 = torch.einsum('in,in -> i', h.conj()[user_index[0], :], h[user_index[1], :]) \
                .reshape(user_index.shape[1], -1)
            user_edge_attr2 = torch.einsum('in,in -> i', h.conj()[user_index[0], :], h[user_index[0], :]) \
                .reshape(user_index.shape[1], -1)
            user_edge_attr3 = torch.einsum('in,in -> i', h.conj()[user_index[1], :], h[user_index[1], :]) \
                .reshape(user_index.shape[1], -1)
            user_edge_attr = torch.cat((user_edge_attr2, user_edge_attr1, user_edge_attr3), dim=1)
            graph_data['user'].x = torch.cat((h.real, h.imag), dim=1)
            graph_data['user'].x = F.pad(graph_data['user'].x, pad=(0, 48 - 2*Nt), mode='constant', value=0)
            # graph_data['user'].x = h

            graph_data['user', 'u2u', 'user'].edge_index = user_index
            graph_data['user', 'u2u', 'user'].edge_attr = torch.cat((user_edge_attr.real, user_edge_attr.imag), dim=1)
            # graph_data['user', 'u2u', 'user'].edge_attr = user_edge_attr

            # 获得天线到用户边特征
            a2u_edge_attr = h[ant_edge_index[1], ant_edge_index[0]].unsqueeze(-1)
            # 添加边
            graph_data['ant', 'a2u', 'user'].edge_index = ant_edge_index
            # 设置边特征
            graph_data['ant', 'a2u', 'user'].edge_attr = torch.cat((a2u_edge_attr.real, a2u_edge_attr.imag), dim=1)
            # graph_data['ant', 'a2u', 'user'].edge_attr = a2u_edge_attr

            # 获得用户到天线边特征
            u2a_edge_attr = h[user_edge_index[0], user_edge_index[1]].unsqueeze(-1)
            # 添加边
            graph_data['user', 'u2a', 'ant'].edge_index = user_edge_index
            # 设置边特征
            graph_data['user', 'u2a', 'ant'].edge_attr = torch.cat((u2a_edge_attr.real, u2a_edge_attr.imag), dim=1)
            # graph_data['user', 'u2a', 'ant'].edge_attr = u2a_edge_attr

            data_list.append(graph_data)
        data_save, data_slices = self.collate(data_list)
        torch.save((data_save, data_slices), self.processed_paths[0])

class HetDatasetV2(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(HetDatasetV2, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['data.mat']

    @property
    def processed_file_names(self):
        return ['data.pt']

    # 处理数据
    def process(self):
        # 读取原始数据
        # 读取原始数据
        dataset = scio.loadmat(self.raw_paths[0])
        num = dataset["H"].shape[0]
        print('数据集大小:{}'.format(num))
        tem = dataset["H"]
        # 获取用户数和天线数
        K = dataset["H"].shape[1]
        Nt = dataset["H"].shape[2]

        # 获得天线节点边
        ant_adj = np.ones([Nt, Nt])
        ant_adj = coo_matrix(ant_adj)
        ant_indices = np.vstack((ant_adj.row, ant_adj.col))
        ant_index = torch.LongTensor(ant_indices)

        # 获得用户节点边
        user_adj = np.ones([K, K])
        user_adj = coo_matrix(user_adj)
        user_indices = np.vstack((user_adj.row, user_adj.col))
        user_index = torch.LongTensor(user_indices)

        # 获得天线到用户节点边
        ant_start_index = []
        user_end_index = []
        for s in range(Nt):
            for e in range(K):
                ant_start_index.append(s)
                user_end_index.append(e)
        # 转化为pyg形式
        ant_start_index = torch.from_numpy(np.asarray(ant_start_index)).to(torch.long)
        user_end_index = torch.from_numpy(np.asarray(user_end_index)).to(torch.long)
        ant_edge_index = torch.stack([ant_start_index, user_end_index], dim=0)

        # 设置从用户到天线边
        user_start_index = []
        ant_end_index = []
        for s in range(K):
            for e in range(Nt):
                user_start_index.append(s)
                ant_end_index.append(e)
        # 转化为pyg形式
        user_start_index = torch.from_numpy(np.asarray(user_start_index)).to(torch.long)
        ant_end_index = torch.from_numpy(np.asarray(ant_end_index)).to(torch.long)
        user_edge_index = torch.stack([user_start_index, ant_end_index], dim=0)
        # print(user_edge_index)
        # 处理后的数据
        data_list = []
        for index in range(num):
            # 获得用户信道
            h = torch.from_numpy(dataset["H"][index])
            # 这里进行特征处理，噪声归一化
            h = h / math.sqrt(1.99e-12)
            rf = torch.from_numpy(dataset["RF"][index])
            bb = torch.from_numpy(dataset["BB"][index])
            sr = torch.from_numpy(dataset["SR"][index])

            # print(h.shape)
            graph_data = HeteroData()

            # 获得天线节点特征
            ant_edge_attr1 = torch.einsum('in,in -> i', h.T.conj()[ant_index[0], :], h.T[ant_index[1], :]) \
                .reshape(ant_index.shape[1], -1)
            ant_edge_attr2 = torch.einsum('in,in -> i', h.T.conj()[ant_index[0], :], h.T[ant_index[0], :]) \
                .reshape(ant_index.shape[1], -1)
            ant_edge_attr3 = torch.einsum('in,in -> i', h.T.conj()[ant_index[1], :], h.T[ant_index[1], :]) \
                .reshape(ant_index.shape[1], -1)
            ant_edge_attr = torch.cat((ant_edge_attr2, ant_edge_attr1, ant_edge_attr3), dim=1)

            # 天线特征
            # Ht_pad = torch.cat([h.T,torch.zeros([Nt, 16 - K], dtype=torch.complex64)],dim =1 )
            # 设置天线节点特征和有向边特征
            graph_data['ant'].x = torch.cat((h.T.real, h.T.imag), dim=1)
            # 此时ant 节点特征shape [n_t , 2K] , 填充到指定维度 填充到32
            graph_data['ant'].x = F.pad(graph_data['ant'].x, pad=(0, 32 - 2*K), mode='constant', value=0)

            graph_data['ant', 'a2a', 'ant'].edge_index = ant_index
            graph_data['ant', 'a2a', 'ant'].edge_attr = torch.cat((ant_edge_attr.real, ant_edge_attr.imag), dim=1)

            # 获得用户节点特征
            user_edge_attr1 = torch.einsum('in,in -> i', h.conj()[user_index[0], :], h[user_index[1], :]) \
                .reshape(user_index.shape[1], -1)
            user_edge_attr2 = torch.einsum('in,in -> i', h.conj()[user_index[0], :], h[user_index[0], :]) \
                .reshape(user_index.shape[1], -1)
            user_edge_attr3 = torch.einsum('in,in -> i', h.conj()[user_index[1], :], h[user_index[1], :]) \
                .reshape(user_index.shape[1], -1)
            user_edge_attr = torch.cat((user_edge_attr2, user_edge_attr1, user_edge_attr3), dim=1)

            # h_pad = torch.cat([h,torch.zeros([K, 24 - Nt], dtype=torch.complex64)],dim =1)

            graph_data['user'].x = torch.cat((h.real, h.imag), dim=1)

            graph_data['user'].x = F.pad(graph_data['user'].x, pad=(0, 48 - 2*Nt), mode='constant', value=0)

            graph_data['user', 'u2u', 'user'].edge_index = user_index
            graph_data['user', 'u2u', 'user'].edge_attr = torch.cat((user_edge_attr.real, user_edge_attr.imag), dim=1)

            # # 获得天线到用户边特征
            # ant_edge_attr = h[ant_edge_index[0], ant_edge_index[1]].unsqueeze(-1)
            # # 添加边
            # data['ant', 'a2u', 'user'].edge_index = ant_edge_index
            # # 设置边特征
            # data['ant', 'a2u', 'user'].edge_attr = torch.cat((ant_edge_attr.real, ant_edge_attr.imag), dim=1)

            # 获得用户到天线边特征
            user_edge_attr = h[user_edge_index[0], user_edge_index[1]].unsqueeze(-1)
            # 添加边
            graph_data['user', 'u2a', 'ant'].edge_index = user_edge_index
            # 设置边特征
            graph_data['user', 'u2a', 'ant'].edge_attr = torch.cat((user_edge_attr.real, user_edge_attr.imag), dim=1)
     
            graph_data.y = sr
            graph_data.RF = rf 
            graph_data.BB = bb

            data_list.append(graph_data)
        data_save, data_slices = self.collate(data_list)
        torch.save((data_save, data_slices), self.processed_paths[0])

def get_bath_rate(RF, BB, P, H):
    """

    :param RF: [b,N_T,k]
    :param BB: [b,k,k]
    :param P:  [b,k,1]
    :param H: [b*k,N_T]
    :return:
    """
    # 先计算出w
    # 这里是有问题的
    W = torch.bmm(RF, BB)
    # 对每个用户的beam 进行归一化即 列归一化
    W = W / torch.norm(W, p=2, dim=1, keepdim=True)
    W = torch.transpose(W, 1, 2)
    I = torch.real(torch.einsum('bim,bjm,bjn,bin -> bij', W.conj(), H, H.conj(), W))
    # print(I)
    I = P * I
    # 按行求和
    dr_temp1 = torch.einsum('bmi -> bi', I) - torch.einsum('bii -> bi', I)  + 1
    R = torch.log2(1 + torch.einsum('bii -> bi', I) / dr_temp1)
    return R

class CNNDataset(Dataset):
    def __init__(self, file_dir):
        super(CNNDataset, self).__init__()
        self.file_dir = file_dir
        # 获得数据集真正的路径
        self.data_file = os.path.join(self.file_dir, "raw/data.mat")
        # 加载获得的数据
        self.data = scio.loadmat(self.data_file)["H"] 
        self.num = self.data.shape[0]

    def __getitem__(self, index):
        h = self.data[index] / math.sqrt(1.99e-12)
        # 这里从numpy 转化为张量 shape [k,nt]
        h = torch.from_numpy(h).to(torch.complex64)
        h_abs = torch.abs(h)
        h_real = torch.real(h)
        h_imag = torch.imag(h)
        out = torch.stack([h_abs, h_real, h_imag], dim=0)
        return out

    def __len__(self):
        return self.num

class MLPDataset(Dataset):
    def __init__(self, file_dir):
        super(MLPDataset, self).__init__()
        self.file_dir = file_dir
        # 获得数据集真正的路径
        self.data_file = os.path.join(self.file_dir, "raw/data.mat")
        # 加载获得的数据
        self.data = scio.loadmat(self.data_file)["H"] 
        self.num = self.data.shape[0]

    def __getitem__(self, index):
        h = self.data[index] / math.sqrt(1.99e-12)
        # 这里从numpy 转化为张量 shape [k,nt]
        h = torch.from_numpy(h).to(torch.complex64)
        out = torch.cat((h.real,h.imag),dim = 1)
        return out

    def __len__(self):
        return self.num


if __name__ == '__main__':
    train_dataset = HetDataset('../dataset/BS_8u_16n_real/test_bs1/')
    train_dataset = HetDataset('../dataset/BS_8u_16n_real/train_bs5/')
    train_dataset = HetDataset('../dataset/BS_8u_16n_real/val_bs5/')
    train_dataset = HetDataset('../dataset/BS_8u_16n_real/test_bs5/')

    train_dataset = HetDataset('../dataset/BS_8u_16n_real/train_bs15/')
    train_dataset = HetDataset('../dataset/BS_8u_16n_real/val_bs15/')
    train_dataset = HetDataset('../dataset/BS_8u_16n_real/test_bs15/')


    # batch_size = 32
    # loader = normDL(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # for batch in loader:
    #     print(batch.shape)
    #     break
    # data = train_dataset[0]
    # x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict

    # print(edge_attr_dict["ant","a2a","ant"].shape)
    # print(x_dict.keys())
    # print(x_dict["user"].shape)
    # print(x_dict["ant"].shape)
    # print(edge_index_dict["user","u2a","ant"])
    # print(edge_attr_dict.keys())
    # print(type(edge_attr_dict.keys()))
    # print(data.metadata())
    # print(data.metadata()[0])
    # print(data.metadata()[1])

    # batch_size = 32
    # loader =  DataLoader(dataset=train_dataset,
    #                   batch_size=batch_size,
    #                   shuffle=True,
    #                   num_workers=0,
    #                   )
    # antenna_num = 16

    # for batch in loader:
    #     # print(batch.num_graphs)
    #     # 获得信道
    #     # H = batch.x_dict["user"]
    #     # H = torch.complex(H[:,:antenna_num], H[:,antenna_num:]).to(torch.complex64)
    #     # H = H.reshape(batch_size,8,-1)
    #     print(batch["user"].batch) 
    #     break

    #     # 获得RF 
    #     RF = batch.RF.reshape(batch_size,16,-1).to(torch.complex64)
    #     BB = batch.BB.reshape(batch_size,8,-1).to(torch.complex64)
    #     P = torch.ones([batch_size, 8,1]) / 8
    #     ur = get_bath_rate(RF,BB,P,H)
    #     print(ur.shape)
    #     print(torch.sum(ur, dim =1 ))
    #     print(batch.y)
    #     break
