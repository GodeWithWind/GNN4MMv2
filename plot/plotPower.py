import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# 设置字体
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

# Data from the table
approaches = ['AO', 'Manifold', 'PZF', 'GCN', 'MLP', 'CNN', 'GAT', 'HGAT']

K_Te_values = [1, 2, 3, 4]

# Sum Rates for different approaches
sum_rates = {
    'AO': [24.40, 29.51, 32.56, 34.82],
    'Manifold': [14.97, 15.18, 15.26, 15.30],
    'PZF': [27.49, 32.91, 36.26, 38.71],
    'GCN': [22.06, 23.59, 20.50, 23.21],
    'MLP': [25.10, 27.10, 26.97, 29.39],
    'CNN': [26.36, 28.23, 30.60, 29.18],
    'GAT': [32.40, 34.38, 36.02, 36.70],
    'HGAT': [38.08, 41.06, 43.10, 44.05]
}

annotation_offsets = {
    'AO': [5, 10, 10, 10, 10],
    'Manifold': [10, 10, 10, 10, 10],
    'PZF': [10, 10, 10, 10, 10],
    'GCN': [10, 10, 10, 10, 10],
    'MLP': [10, 10, 10, 10, 10],
    'CNN': [10, 10, 10, 10, 10],
    'GAT': [10, 10, 10, 10, 10],
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
plt.xlabel('$Nt_{\\rm Te}$', fontsize=15)
plt.ylabel('Sum Rate (bits/s/Hz)', fontsize=15)
plt.title('Scalability to Different Numbers of antennas', fontsize=15)
plt.xticks(K_Te_values)  # Ensuring x-axis only has integer values
plt.legend()
plt.grid(True)
# plt.savefig('./SCUsers.pdf', dpi=1200)
plt.show()
