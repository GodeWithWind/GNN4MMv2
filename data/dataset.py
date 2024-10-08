import numpy as np
import scipy.io as scio

import torch
from scipy.sparse import coo_matrix
import os
from torch_geometric.data import Data, InMemoryDataset, HeteroData
from torch.utils.data import Dataset, DataLoader


class HomogeneousDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(HomogeneousDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['data.npy']

    @property
    def processed_file_names(self):
        return ['data.pt']

    # 处理数据
    def process(self):
        # 读取原始数据
        dataset = np.load(self.raw_paths[0], allow_pickle=True)
        print('数据集大小:{}'.format(len(dataset)))
        tem = dataset[0].item()
        # 获取用户数和天线数
        Nt = tem["user"]["channel"].shape[2]
        K = tem["user"]["channel"].shape[0]
        data_list = []
        # 依次处理所有数据
        for i in range(len(dataset)):
            # 构建边
            # shape = [2,edgeNum] 从源节点到目标节点   1 2 3
            # 生成临接矩阵
            adj = np.zeros([K, K])
            for i in range(K):
                for j in range(K):
                    # 这里默认添加自环边
                    # if i != j:
                    adj[i, j] = 1

            adj = coo_matrix(adj)
            indices = np.vstack((adj.row, adj.col))  # 我们真正需要的coo形式
            edge_index = torch.LongTensor(indices)  # PyG框架需要的coo形式

            # 获得单个sample
            data = dataset[i].item()
            # 获得用户信道并减少维度 shape [k,n_t,path]
            h = data["user"]["channel"].squeeze()
            # 转化为张量
            h = torch.from_numpy(h)
            # 将每个path上的信道进行求和
            h = torch.einsum('knp -> kn', h)

            # 构建边特征
            edge1 = torch.einsum('in,in -> i', h.conj()[edge_index[0], :], h[edge_index[1], :]).reshape(
                edge_index.shape[1], -1)
            edge2 = torch.einsum('in,in -> i', h.conj()[edge_index[0], :], h[edge_index[0], :]).reshape(
                edge_index.shape[1], -1)
            edge3 = torch.einsum('in,in -> i', h.conj()[edge_index[1], :], h[edge_index[1], :]).reshape(
                edge_index.shape[1], -1)

            edge_attr = torch.cat((edge2.real, edge1.real, edge3.real, edge2.imag, edge1.imag, edge3.imag), dim=1)

            graph_data = Data(edge_index=edge_index, x=torch.cat((h.real, h.imag), dim=1), y=None, edge_attr=edge_attr)
            data_list.append(graph_data)
        data_save, data_slices = self.collate(data_list)
        torch.save((data_save, data_slices), self.processed_paths[0])


class HomRelGraphMat(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(HomRelGraphMat, self).__init__(root, transform, pre_transform)
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
        dataset = scio.loadmat(self.raw_paths[0])
        print('数据集大小:{}'.format(len(dataset)))
        tem = dataset["H"]
        # 获取用户数和天线数
        K = dataset["H"].shape[1]
        data_list = []
        # 依次处理所有数据
        for i in range(len(dataset)):
            adj = np.zeros([K, K])
            for i in range(K):
                for j in range(K):
                    adj[i, j] = 1

            adj = coo_matrix(adj)
            indices = np.vstack((adj.row, adj.col))  # 我们真正需要的coo形式
            edge_index = torch.LongTensor(indices)  # PyG框架需要的coo形式

            h = torch.from_numpy(dataset["H"][i])
            rf = torch.from_numpy(dataset["RF"][i])
            bb = torch.from_numpy(dataset["BB"][i])
            sr = torch.from_numpy(dataset["SR"][i])

            edge1 = torch.einsum('in,in -> i', h.conj()[edge_index[0], :], h[edge_index[1], :]).reshape(
                edge_index.shape[1], -1)
            edge2 = torch.einsum('in,in -> i', h.conj()[edge_index[0], :], h[edge_index[0], :]).reshape(
                edge_index.shape[1], -1)
            edge3 = torch.einsum('in,in -> i', h.conj()[edge_index[1], :], h[edge_index[1], :]).reshape(
                edge_index.shape[1], -1)

            edge_attr = torch.cat((edge2.real, edge1.real, edge3.real, edge2.imag, edge1.imag, edge3.imag), dim=1)

            graph_data = Data(edge_index=edge_index, x=torch.cat((h.real, h.imag), dim=1), y=sr, RF=rf, BB=bb,
                              edge_attr=edge_attr)
            data_list.append(graph_data)
        data_save, data_slices = self.collate(data_list)
        torch.save((data_save, data_slices), self.processed_paths[0])


class HomComDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(HomComDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['data.npy']

    @property
    def processed_file_names(self):
        return ['data.pt']

    # 处理数据
    def process(self):
        # 读取原始数据
        dataset = np.load(self.raw_paths[0], allow_pickle=True)
        print('数据集大小:{}'.format(len(dataset)))
        tem = dataset[0].item()
        # 获取用户数和天线数
        Nt = tem["user"]["channel"].shape[2]
        K = tem["user"]["channel"].shape[0]
        data_list = []
        # 依次处理所有数据
        for i in range(len(dataset)):
            # 构建边
            # shape = [2,edgeNum] 从源节点到目标节点   1 2 3
            # 生成临接矩阵
            adj = np.zeros([K, K])
            for i in range(K):
                for j in range(K):
                    # 这里默认添加自环边
                    # if i != j:
                    adj[i, j] = 1

            adj = coo_matrix(adj)
            indices = np.vstack((adj.row, adj.col))  # 我们真正需要的coo形式
            edge_index = torch.LongTensor(indices)  # PyG框架需要的coo形式

            # 获得单个sample
            data = dataset[i].item()
            # 获得用户信道并减少维度 shape [k,n_t,path]
            h = data["user"]["channel"].squeeze()
            # 转化为张量
            h = torch.from_numpy(h)
            # 将每个path上的信道进行求和
            h = torch.einsum('knp -> kn', h)

            # 构建边特征
            edge1 = torch.einsum('in,in -> i', h.conj()[edge_index[0], :], h[edge_index[1], :]).reshape(
                edge_index.shape[1], -1)
            edge2 = torch.einsum('in,in -> i', h.conj()[edge_index[0], :], h[edge_index[0], :]).reshape(
                edge_index.shape[1], -1)
            edge3 = torch.einsum('in,in -> i', h.conj()[edge_index[1], :], h[edge_index[1], :]).reshape(
                edge_index.shape[1], -1)

            edge_attr = torch.cat((edge2, edge1, edge3), dim=1)

            graph_data = Data(edge_index=edge_index, x=h, y=None, edge_attr=edge_attr)
            data_list.append(graph_data)
        data_save, data_slices = self.collate(data_list)
        torch.save((data_save, data_slices), self.processed_paths[0])


class HeterogeneousDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(HeterogeneousDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['data.npy']

    @property
    def processed_file_names(self):
        return ['data.pt']

    # 处理数据
    def process(self):
        # 读取原始数据
        # 读取原始数据
        dataset = np.load(self.raw_paths[0], allow_pickle=True)
        print('数据集大小:{}'.format(len(dataset)))
        tem = dataset[0].item()
        # 获取用户数和天线数
        num = len(dataset)
        Nt = tem["user"]["channel"].shape[2]
        K = tem["user"]["channel"].shape[0]

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
        for i in range(num):
            # 获得单个sample
            data = dataset[i].item()
            # 获得用户信道并减少维度 shape [k,n_t,path]
            h = data["user"]["channel"].squeeze()
            # 转化为张量
            h = torch.from_numpy(h)
            # 将每个path上的信道进行求和
            h = torch.einsum('knp -> kn', h)

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

            # 设置天线节点特征和有向边特征
            graph_data['ant'].x = torch.cat((h.T.real, h.T.imag), dim=1)
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
            graph_data['user'].x = torch.cat((h.real, h.imag), dim=1)
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

            data_list.append(graph_data)
        data_save, data_slices = self.collate(data_list)
        torch.save((data_save, data_slices), self.processed_paths[0])


class CNNDataset(Dataset):
    def __init__(self, file_dir):
        super(CNNDataset, self).__init__()
        self.file_dir = file_dir
        # 获得数据集真正的路径
        self.data_file = os.path.join(self.file_dir, "raw/data.mat")
        # 加载获得的数据
        self.data = scio.loadmat(self.data_file)["H_gather"]
        self.num = self.data.shape[0]

    def __getitem__(self, index):
        h = self.data[index]
        # 这里从numpy 转化为张量 shape [k,nt]
        h = torch.from_numpy(h).to(torch.complex64)
        h_abs = torch.abs(h)
        h_real = torch.real(h)
        h_imag = torch.imag(h)
        out = torch.stack([h_abs, h_real, h_imag], dim=0)
        return h, out

    def __len__(self):
        return self.num


if __name__ == '__main__':
    train_dataset = CNNDataset('../dataset/8u_16n/train/')
    batch_size = 32
    loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    for batch in loader:
        print(batch[0].shape)
        print(batch[1].shape)
        break
