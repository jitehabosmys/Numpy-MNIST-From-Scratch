import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

# model = nn.models.Model_MLP()
# model.load_model(r'.\codes\saved_models\best_model_26.pickle')
model = nn.models.Model_CNN(bn=True)
model.load_model(r'.\codes\saved_models\best_model_363.pickle')

test_images_path = r'.\codes.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\codes.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / test_imgs.max()

# 评估模式
model.eval()
logits = model(test_imgs)
# 预测标签
pred_labs = np.argmax(logits, axis=-1)
# 计算准确率
print(nn.metric.accuracy(logits, test_labs))

# 绘制混淆矩阵
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labs, pred_labs)
plt.imshow(cm, cmap=plt.cm.Blues)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.xticks([], [])
plt.yticks([], [])
plt.title('Confusion matrix')
plt.colorbar()

# 在每个单元格中添加数字
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > cm.max()/2 else "black")
plt.show()
