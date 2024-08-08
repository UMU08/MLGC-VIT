import matplotlib.pyplot as plt
import numpy as np


# 设置中文字体为SimHei
plt.rcParams['font.family'] = 'SimHei'
# 假设您有 5 层 GCN 和带有密集连接的 GCN 的节点表示的差异平均值数据
layers = [1, 2, 3, 4, 5]  # 层数
mean_diffs_gcn = [0.27, 0.19, 0.13, 0.09, 0.02]  # 多层 GCN 的节点表示的差异平均值
mean_diffs_dense = [0.27, 0.25, 0.23, 0.19, 0.16]  # 带有密集连接的 GCN 的节点表示的差异平均值

# 创建折线图
plt.figure(figsize=(8, 6))
plt.plot(layers, mean_diffs_gcn, marker='o', label='多层 GCN')
plt.plot(layers, mean_diffs_dense, marker='o', label='带有密集连接的 GCN')

# 添加标题和标签
plt.title('多层 GCN 与带有密集连接的 GCN 节点表示的差异平均值')
plt.xlabel('层数')
plt.ylabel('差异平均值')

# 设置横坐标刻度
plt.xticks(layers, layers)
# 显示图例
plt.legend()

# 显示网格线
plt.grid(True)

# 显示图形
plt.show()