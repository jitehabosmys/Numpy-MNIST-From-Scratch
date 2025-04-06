# codes to make visualization of your weights.
import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

model = nn.models.Model_CNN()
model.load_model(r'.\codes\saved_models\best_model_3.pickle')

test_images_path = r'.\codes\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\codes\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / test_imgs.max()

# logits = model(test_imgs)

# mats = []
# mats.append(model.layers[0].params['W'])
# mats.append(model.layers[2].params['W'])        

# _, axes = plt.subplots(30, 20)
# _.set_tight_layout(1)
# axes = axes.reshape(-1)
# for i in range(600):
#         axes[i].matshow(mats[0].T[i].reshape(28,28))
#         axes[i].set_xticks([])
#         axes[i].set_yticks([])

# plt.figure()
# plt.matshow(mats[1])
# plt.xticks([])
# plt.yticks([])
# plt.show()

# # 打印模型每层的名称
# for i, layer in enumerate(model.layers):
#     print(f"Layer {i} ({layer.__class__.__name__})")

# 获取卷积核参数
conv1_kernels = model.layers[0].params['W'][:, 0, :, :] # [16, 5, 5]
conv2_kernels = model.layers[3].params['W'].mean(axis=1) # [32, 5, 5]

# 可视化 conv1 的卷积核
plt.figure(figsize=(10, 6))
plt.suptitle('Conv1 Kernels (16@5x5)')
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(conv1_kernels[i], cmap='gray')
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.show()

# 可视化 conv2 的卷积核
plt.figure(figsize=(10, 6))
plt.suptitle('Conv2 Kernels (32@5x5, Mean over Input Channels)')
for i in range(32):
    plt.subplot(8, 4, i+1)
    plt.imshow(conv2_kernels[i], cmap='gray')
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.show()