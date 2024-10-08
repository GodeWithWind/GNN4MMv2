import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import matplotlib as mpl

# 设置字体
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

# Define possible linestyles and markers
linestyles = ['-', '--', '-.', ':']
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '|', '_']

# Define a list of distinct colors with high contrast
distinct_colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF', '#800000', '#008000', '#000080',
                   '#808000', '#800080', '#008080', '#808080', '#C0C0C0', '#FFA500', '#FFC0CB']


# 加载数据
def load_data(file_path, key):
    return scio.loadmat(file_path)[key].squeeze()


data_files = [
    ("./PZF/bs15.mat", "SR_gather", "PZF"),
    ("./AO/bs15.mat", "SR_gather_test", "AO"),
    ("./Manifold/bs15.mat", "SR_gather", "Manifold"),
    ("./CNN/8u_16n_bs15.mat", "SR_gather", "CNN"),
    # ("./GAT/8u_16n.mat", "SR_gather", "GAT"),
    ("./GATReal/8u_16n_bs15.mat", "SR_gather", "GAT"),
    ("./GCN/8u_16n_bs15.mat", "SR_gather", "GCN"),
    ("./HetGATV2/8u_16n_bs15.mat", "SR_gather", "HGAT"),
    ("./MLP/8u_16n_bs15.mat", "SR_gather", "MLP")
]

# 固定颜色映射
color_map = {
    "PZF": '#000075',
    "AO": '#3cb44b',
    "Manifold": '#ffe119',
    "CNN": '#800000',
    # "GAT": '#00FFFF',
    "GAT": '#f58231',
    "GCN": '#9A6324',
    "HGAT": '#e6194B',
    "MLP": '#f032e6'
}

# 加载数据并计算均值
data = [(load_data(file, key), label, np.mean(load_data(file, key))) for file, key, label in data_files]

# 按均值排序
data.sort(key=lambda x: x[2])


# 计算 CDF
def compute_cdf(data):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, cdf


# 绘制 CDF 图
# plt.figure(figsize=(10, 10))

fig, ax = plt.subplots(1, 1, figsize=(6, 5.5))
# 设置背景
ax.grid(visible=True, which='major', linestyle='-')
ax.grid(visible=True, which='minor', linestyle='--', alpha=0.5)
# 显示小的刻度
ax.minorticks_on()

for dataset, label, mean in data:
    color = color_map[label]
    # distinct_colors.remove(color)
    sorted_data, cdf = compute_cdf(dataset)
    plt.plot(sorted_data, cdf, label=f'{label} (mean: {mean:.3f})', linewidth=2, color=color)

# 图例和标签
plt.tick_params(labelsize=15)
plt.xlabel('Sum Rate (bits/s/Hz)', fontsize=20)
plt.ylabel('CDF', fontsize=20)
# plt.title('Sum Rate CDF')
plt.legend(loc='best')
plt.grid(True)
plt.savefig('./BS15CDF.pdf', dpi=1200)
# 显示图形
plt.show()
