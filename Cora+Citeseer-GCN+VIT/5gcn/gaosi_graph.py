# import numpy as np
# import matplotlib.pyplot as plt
#
# # 定义图像大小和高斯噪声参数
# n = 20  # 图像大小为 n x n
# mean = 1  # 均值
# std = 0.2   # 标准差
#
# # 生成正态分布的随机数作为高斯噪声（对应三个通道）
# gaussian_noise_r = np.random.normal(mean, std, (n, n)) * 0.1  # 少许红色通道
# gaussian_noise_g = np.random.normal(mean, std, (n, n)) * 0.8  # 主要绿色通道
# gaussian_noise_b = np.random.normal(mean, std, (n, n)) * 0.3  # 少许蓝色通道
#
# # 确保通道值在 0 到 1 之间
# gaussian_noise_r = np.clip(gaussian_noise_r, 0, 1)
# gaussian_noise_g = np.clip(gaussian_noise_g, 0, 1)
# gaussian_noise_b = np.clip(gaussian_noise_b, 0, 1)
#
# # 将三个通道的数据合并成彩色图像
# gaussian_noise_rgb = np.stack([gaussian_noise_r, gaussian_noise_g, gaussian_noise_b], axis=-1)
#
# # 显示彩色高斯噪声图像
# plt.imshow(gaussian_noise_rgb)
# plt.title('Green-Dominant Gaussian Noise')
# plt.axis('off')  # 关闭坐标轴
# plt.show()
import numpy as np
import matplotlib.pyplot as plt

# 定义图像大小和高斯噪声参数
n = 20  # 图像大小为 n x n
mean = 1  # 均值
std = 0.6   # 标准差

# 生成正态分布的随机数作为高斯噪声（对应三个通道）
gaussian_noise_r = np.random.normal(mean, std, (n, n)) * 0.3  # 少许红色通道
gaussian_noise_g = np.random.normal(mean, std, (n, n)) * 0.8  # 主要绿色通道
gaussian_noise_b = np.random.normal(mean, std, (n, n)) * 0.3  # 少许蓝色通道

# 确保通道值在 0 到 1 之间
gaussian_noise_r = np.clip(gaussian_noise_r, 0, 1)
gaussian_noise_g = np.clip(gaussian_noise_g, 0, 1)
gaussian_noise_b = np.clip(gaussian_noise_b, 0, 1)

# 将三个通道的数据合并成彩色图像
gaussian_noise_rgb = np.stack([gaussian_noise_r, gaussian_noise_g, gaussian_noise_b], axis=-1)

# 创建子图
fig, axs = plt.subplots(2, 2, figsize=(8, 8))

# 在不同子图中显示图像的四个部分
axs[0, 0].imshow(gaussian_noise_rgb[:n//2, :n//2])
axs[0, 0].axis('off')

axs[0, 1].imshow(gaussian_noise_rgb[:n//2, n//2:])
axs[0, 1].axis('off')

axs[1, 0].imshow(gaussian_noise_rgb[n//2:, :n//2])
axs[1, 0].axis('off')

axs[1, 1].imshow(gaussian_noise_rgb[n//2:, n//2:])
axs[1, 1].axis('off')

plt.show()