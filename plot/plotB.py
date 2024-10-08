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
    ("./GATReal/8u_16n.mat", "SR_gather", "GAT"),
    ("./GATReal/quantizeB3.mat", "SR_gather", "Quantized GAT, B=3"),
    ("./GATReal/quantizeB4.mat", "SR_gather", "Quantized GAT, B=4"),
    ("./HetGATV2/8u_16n.mat", "SR_gather", "HGAT"),
    ("./HetGATV2/quantizeB3.mat", "SR_gather", "Quantized HGAT, B=3"),
    ("./HetGATV2/quantizeB4.mat", "SR_gather", "Quantized HGAT, B=4"),
]


# 固定颜色映射
color_map = {
    "GAT": '#469990',
    "Quantized GAT, B=3": '#000075',
    "Quantized GAT, B=4": '#ffe119',
    "HGAT": '#9A6324',
    "Quantized HGAT, B=3": '#f58231',
    "Quantized HGAT, B=4": '#42d4f4'
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
# plt.figure(figsize=(10, 8))

fig, ax = plt.subplots(1, 1)
# 设置背景
ax.grid(visible=True, which='major', linestyle='-')
ax.grid(visible=True, which='minor', linestyle='--', alpha=0.5)
# 显示小的刻度
ax.minorticks_on()

for dataset, label, mean in data:
    # color = color_map[label]
    # distinct_colors.remove(color)
    sorted_data, cdf = compute_cdf(dataset)
    plt.plot(sorted_data, cdf, label=f'{label} (mean: {mean:.3f})', linewidth=2)

# 图例和标签
plt.tick_params(labelsize=15)
plt.xlabel('Sum Rate (bits/s/Hz)', fontsize=15)
plt.ylabel('CDF', fontsize=15)
# plt.title('Sum Rate CDF')
plt.legend(loc='best')
plt.grid(True)
plt.savefig('./Quantized.pdf', dpi=1200)
# 显示图形
plt.show()
