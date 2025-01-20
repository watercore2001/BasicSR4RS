import matplotlib.pyplot as plt
import numpy as np

# 生成假的数据
data1 = np.random.rand(600, 600, 3)  # 第一个 ndarray
data2 = np.random.rand(600, 600, 3)  # 第二个 ndarray
data3 = np.random.rand(600, 600, 3)  # 第三个 ndarray

# 调整波段的顺序为 BGR
data1_bgr = data1[..., ::-1]
data2_bgr = data2[..., ::-1]
data3_bgr = data3[..., ::-1]

# 创建一个包含三个子图的画布
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# 显示每个 ndarray 的三个波段
axs[0].imshow(data1_bgr)
axs[0].set_title('Data 1')
axs[0].axis('off')

axs[1].imshow(data2_bgr)
axs[1].set_title('Data 2')
axs[1].axis('off')

axs[2].imshow(data3_bgr)
axs[2].set_title('Data 3')
axs[2].axis('off')
plt.savefig('test.png')
plt.show()