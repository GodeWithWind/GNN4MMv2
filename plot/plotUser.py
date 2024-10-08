import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# 设置字体
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

# Data from the table
approaches = ['AO', 'Manifold', 'PZF', 'GAT', 'HGAT']
K_Te_values = [6, 7, 8, 9, 10]

# Sum Rates for different approaches
sum_rates = {
    'AO': [30.22, 28.16, 24.40, 19.02, 14.79],
    'Manifold': [14.69, 14.84, 14.97, 14.86, 14.92],
    'PZF': [31.92, 30.22, 27.49, 22.50, 17.36],
    'GAT': [28.71, 30.88, 32.40, 33.29, 34.02],
    'HGAT': [30.20, 35.53, 38.08, 39.21, 40.00]
}

# Specific offsets to avoid overlap at K = 6 and K = 10
annotation_offsets = {
    'AO': [5, -15, 10, 10, -10],
    'Manifold': [10, 10, 10, 10, 10],
    'PZF': [10, -12, 10, 10, 10],
    'GAT': [-10, 10, 10, 10, 10],
    'HGAT': [-10, 15, 10, 10, -15]
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
plt.tick_params(labelsize=10)
plt.xlabel('Numbers of Users', fontsize=15)
plt.ylabel('Sum Rate (bits/s/Hz)', fontsize=15)
# plt.title('Scalability to Different Numbers of Users', fontsize=15)
plt.xticks(K_Te_values)  # Ensuring x-axis only has integer values
plt.legend()
plt.grid(True)
plt.savefig('./SCUsers.pdf', dpi=1200)
plt.show()
