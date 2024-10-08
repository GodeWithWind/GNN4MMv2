import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import unbatch
import scipy.io as scio
import torch.nn.functional as F


# # 创建一个空的数据列表
# dataset = []
# graph_size = []
# # 生成20张不同大小的图
# for i in range(20):
#     # 生成节点特征
#     num_nodes = torch.randint(low=5, high=15, size=(1,)).item()  # 随机生成节点数量
#     graph_size.append(num_nodes)
#     x = torch.randn(num_nodes, 16)  # 生成节点特征，这里假设每个节点有16个特征
#
#     # 生成边索引
#     num_edges = torch.randint(low=10, high=30, size=(1,)).item()  # 随机生成边数量
#     edge_index = torch.randint(low=0, high=num_nodes, size=(2, num_edges))  # 随机生成边的起点和终点索引
#
#     # 创建一个Data对象，并将其添加到数据集中
#     data = Data(x=x, edge_index=edge_index)
#     dataset.append(data)
#
# print(graph_size)
# # 创建一个数据加载器
# loader = DataLoader(dataset, batch_size=4, shuffle=False)
#
# # 遍历数据集
# for batch in loader:
#     tem = unbatch(batch.x, batch.batch)
#     for te in tem:
#         print(te.shape)
#     break


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
    W = torch.transpose(W, 1, 2)
    I = torch.real(torch.einsum('bim,bjm,bjn,bin -> bij', W.conj(), H, H.conj(), W))
    I = P.unsqueeze(-1) * I
    # 按行求和
    dr_temp1 = torch.einsum('bmi -> bi', I) - torch.einsum('bii -> bi', I) + 0.1
    R = torch.log2(1 + torch.einsum('bii -> bi', I) / dr_temp1)
    return R


RF = torch.randn([10, 16, 8], dtype=torch.complex64).to('cuda')
BB = torch.randn([10, 8, 8], dtype=torch.complex64).to('cuda')
P = torch.randn(10,8,1).to('cuda')
P = torch.softmax(P,dim =1)

BB_tem = torch.sqrt(P) * BB
W_tem = torch.bmm(RF, BB_tem)
W_tem_norm = torch.norm(W_tem, p='fro', dim=[1, 2], keepdim=True) ** 2
scaling_factor = torch.sqrt(4/ W_tem_norm)
BB = BB * scaling_factor


BB = torch.sqrt(P) * BB
W = torch.bmm(RF, BB)
W_norm = torch.norm(W, p='fro', dim=[1, 2], keepdim=True) ** 2

print(W_norm)





# W_norm = torch.norm(W, p='fro', dim=[1, 2], keepdim=True)

# W = torch.sqrt(torch.tensor(16).to('cuda')) * torch.div(W, torch.max(W_norm, torch.ones(
#     W_norm.size()).to('cuda')))

# print(torch.norm(W, p='fro', dim=[1, 2], keepdim=True)** 2)

# 按照列求模
# no = torch.norm(W, p=2, dim=1, keepdim=True)
# print(no.shape)
# # print(no.shape)
# # print(no)
# W = W / no
# # print(torch.norm(W, p=2, dim=1, keepdim=True))
# tem = W[0]
# print(tem.shape)
# print(torch.norm(tem, p=2, dim=0, keepdim=True))

# data = scio.loadmat("./test.mat")
# print(data.keys())
# H = torch.from_numpy(data["H"])
# print(H.shape)
# RF = torch.from_numpy(data["F"])
#
# BB = torch.from_numpy(data["Fb"])
#
# W = torch.mm(RF, BB)
# print(W.shape)
# # 按照列进行归一化
# W = W / torch.norm(W, p=2, dim=0, keepdim=True)
# # W = torch.from_numpy(data["WPR"])
# W = W.T
#
# # print(H[0, :])
# # print(W[0, :])
# # print(P)
# # Hj = H[0, :]
# # Wj = W[0, :]
# # # 计算 Hj*Wj
# # HW = Hj.conj() @ Wj
# # re = HW * HW.conj() * 0.25
# # print(re)
# # H = H.conj()
# I = torch.real(torch.einsum('im,jm,jn,in -> ij', W.conj(), H, H.conj(), W))
# P = torch.ones([4, 1]) / 4
# I = P * I
# # print(I)
#
# # I2 = torch.real(torch.multiply(torch.matmul(W.conj(), H.T), torch.matmul(W.conj(), H.T).conj()))
# # I2 = P * I2
# # print(I2)
# # print(I)
# # print(I)
# #
# # # 按行求和
# dr_temp1 = torch.einsum('mi -> i', I) - torch.einsum('ii -> i', I) + 1.99e-12
# R = torch.log2(1 + torch.einsum('ii -> i', I) / dr_temp1)
# print(R)
# print(torch.sum(R))
# print(data['sum_rate'])

