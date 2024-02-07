import numpy as np
import scipy.io as scio
import networkx as nx
import torch
from scipy.sparse import coo_matrix
from torch_geometric.data import Data, InMemoryDataset, HeteroData
from torch_geometric.data import DataLoader


class HomogeneousRealDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(HomogeneousRealDataset, self).__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['data.pth']

    @property
    def processed_file_names(self):
        return ['data.pt']

    # 处理数据
    def process(self):
        # 读取原始数据
        data_list = []
        dict = torch.load(self.raw_paths[0])
        for key, value in dict.items():
            print(key)
            # 获得用户数
            print(value.shape)
            K = value.shape[2]
            # 构建边
            # shape = [2,edgeNum] 从源节点到目标节点   1 2 3
            # 生成临接矩阵
            adj = np.zeros([K, K])
            for i in range(K):
                for j in range(K):
                    # 这里默认添加自环边
                    adj[i, j] = 1

            adj = coo_matrix(adj)
            indices = np.vstack((adj.row, adj.col))  # 我们真正需要的coo形式
            edge_index = torch.LongTensor(indices)  # PyG框架需要的coo形式

            for i in range(value.shape[0]):
                # 获得单个sample
                h = value[i]
                # 构建节点特征
                x = h.T
                # 构建边特征
                edge_attr = torch.einsum('in,in -> i', x.conj()[edge_index[0], :], x[edge_index[1], :]).reshape(
                    edge_index.shape[1], -1)

                x = torch.cat((x.real, x.imag), dim=1)
                edge_attr = torch.cat((edge_attr.real, edge_attr.imag), dim=1)
                data = Data(edge_index=edge_index, x=x, edge_attr=edge_attr)
                data_list.append(data)

        data_save, data_slices = self.collate(data_list)
        torch.save((data_save, data_slices), self.processed_paths[0])


if __name__ == '__main__':
    train_dataset = HomogeneousRealDataset('../dataset/16n/test_9u/')
    print(len(train_dataset))
    # data = train_dataset[110000]
    # print(data.x.shape)
    # print(data.edge_attr.shape)


