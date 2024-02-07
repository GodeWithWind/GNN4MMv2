import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import unbatch

# 创建一个空的数据列表
dataset = []
graph_size = []
# 生成20张不同大小的图
for i in range(20):
    # 生成节点特征
    num_nodes = torch.randint(low=5, high=15, size=(1,)).item()  # 随机生成节点数量
    graph_size.append(num_nodes)
    x = torch.randn(num_nodes, 16)  # 生成节点特征，这里假设每个节点有16个特征

    # 生成边索引
    num_edges = torch.randint(low=10, high=30, size=(1,)).item()  # 随机生成边数量
    edge_index = torch.randint(low=0, high=num_nodes, size=(2, num_edges))  # 随机生成边的起点和终点索引

    # 创建一个Data对象，并将其添加到数据集中
    data = Data(x=x, edge_index=edge_index)
    dataset.append(data)

print(graph_size)
# 创建一个数据加载器
loader = DataLoader(dataset, batch_size=4, shuffle=False)

# 遍历数据集
for batch in loader:
    tem = unbatch(batch.x, batch.batch)
    for te in tem:
        print(te.shape)
    break
