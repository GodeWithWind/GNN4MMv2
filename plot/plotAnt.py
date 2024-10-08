import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# 设置字体
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

# Data from the table
# approaches = ['AO', 'Manifold', 'PZF', 'GATV2', 'HGAT']
approaches = ['AO', 'Manifold', 'PZF', 'HGAT']

K_Te_values = [14, 15, 16, 17, 18]

# Sum Rates for different approaches
sum_rates = {
    'AO': [18.47, 21.42, 24.40, 25.76, 29.23],
    'Manifold': [13.71, 14.11, 15.63, 14.86, 16.34],
    'PZF': [20.95, 24.22, 27.49, 29.97, 32.46],
    # 'GATV2': [28.71, 30.88, 32.40, 33.29, 34.02],
    'HGAT': [35.46, 36.77, 38.08, 39.00, 39.92]
}

# Specific offsets to avoid overlap at K = 6 and K = 10
# annotation_offsets = {
#     'AO': [5, -15, 10, 10, -10],
#     'Manifold': [10, 10, 10, 10, 10],
#     'PZF': [10, -12, 10, 10, 10],
#     # 'GATV2': [-10, 10, 10, 10, 10],
#     'HGAT': [-10, 15, 10, 10, -15]
# }

# 固定颜色映射
# color_map = {
#     "PZF": '#ffee6f',
#     "AO": '#99bcac',
#     "Manifold": '#5f4321',
#     "CNN": '#c12c1f',
#     # "GAT": '#00FFFF',
#     "GATV2": '#dd7694',
#     "GCN": '#ed6d46',
#     "HGAT": '#2a6e3f',
#     "MLP": '#000080'
# }

annotation_offsets = {
    'AO': [5, 10, 10, 10, 10],
    'Manifold': [10, 10, 10, 10, 10],
    'PZF': [10, 10, 10, 10, 10],
    # 'GATV2': [-10, 10, 10, 10, 10],
    'HGAT': [10, 10, 10, 10, -15]
}

# Plotting
fig, ax = plt.subplots(1, 1)
# 设置背景
ax.grid(visible=True, which='major', linestyle='-')
ax.grid(visible=True, which='minor', linestyle='--', alpha=0.5)
# 显示小的刻度
ax.minorticks_on()

for approach in approaches:
    plt.plot(K_Te_values, sum_rates[approach], marker='o', label=approach)
    for i, txt in enumerate(sum_rates[approach]):
        offset = annotation_offsets[approach][i]
        plt.annotate(f'{txt}', (K_Te_values[i], sum_rates[approach][i]), textcoords="offset points", xytext=(0, offset),
                     ha='center')
plt.tick_params(labelsize=12)
plt.xlabel('Number of antennas', fontsize=15)
plt.ylabel('Sum Rate (bits/s/Hz)', fontsize=15)
# plt.title('Scalability to Different Numbers of antennas', fontsize=15)
plt.xticks(K_Te_values)  # Ensuring x-axis only has integer values
plt.legend()
plt.grid(True)
plt.savefig('./SCAnts.pdf', dpi=1200)
plt.show()
