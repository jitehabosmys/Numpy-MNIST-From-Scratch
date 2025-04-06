# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

# fixed seed for experiment
np.random.seed(309)

train_images_path = r'codes.\dataset\MNIST\train-images-idx3-ubyte.gz'  
train_labels_path = r'codes.\dataset\MNIST\train-labels-idx1-ubyte.gz'



with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)


# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

# # 1. Linear + SGD = 0.9361
# linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4])
# optimizer = nn.optimizer.SGD(init_lr=0.06, model=linear_model)
# scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
# loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)
# runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)
# print("Start training model 1")
# runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=100, save_dir=r'.\codes\saved_models', save_filename='best_model_1.pickle')

# fig, axes = plt.subplots(1, 2)
# fig.set_tight_layout(True)
# plot(runner, axes)
# plt.savefig(r'.\codes\figs\Figure_1.png')
# plt.close()

# # 2.Linear + MomentumGD = 0.9403
# linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4])
# optimizer = nn.optimizer.MomentGD(init_lr=0.06, model=linear_model, mu=0.9)
# scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
# loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)
# runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)
# print("Start training model 2")
# runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=100, save_dir=r'.\codes\saved_models',save_filename='best_model_2.pickle')

# fig, axes = plt.subplots(1, 2)
# fig.set_tight_layout(True)
# plot(runner, axes)
# plt.savefig(r'.\codes\figs\Figure_2.png')
# plt.close()

# # 3. Conv2D + SGD = 0.9556
# conv_model = nn.models.Model_CNN()
# optimizer = nn.optimizer.SGD(init_lr=0.01, model=conv_model)
# scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
# loss_fn = nn.op.MultiCrossEntropyLoss(model=conv_model, max_classes=train_labs.max()+1)
# # loss_fn.cancel_soft_max()
# runner = nn.runner.RunnerM(conv_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)
# print("Start training model 3")
# runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=1, log_iters=300, save_dir=r'.\codes\saved_models', save_filename='best_model_3.pickle')

# fig, axes = plt.subplots(1, 2)
# fig.set_tight_layout(True)
# plot(runner, axes)
# plt.savefig(r'.\codes\figs\Figure_3.png')
# plt.close()


# # 4. Linear + Adam + Data Augmentation 
# # 测试每种数据增强会不会提升模型性能
# """
# 4: {'rotation':True, 'flip':True,'shift':True} = 0.8853 -> 0.8787
# 5. {rotation:True} = 0.9309 -> 0.9155
# 6. {flip:True} = 0.939 -> 0.9215
# 7. {shift:True} = 0.9451 -> 0.9556
# 8. {} = 0.9572 -> 0.9556
# 结论：数据增强反而会降低模型性能
# """
# transform_dicts = [{'rotation':True, 'flip':True,'shift':True}, {'rotation':True}, {'flip':True}, {'shift':True}, {}]

# for i, transform_dict in enumerate(transform_dicts):
#         print(f"Start training with transform_dict {transform_dict}")
#         linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4])
#         optimizer = nn.optimizer.Adam(init_lr=0.06, model=linear_model)
#         scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
#         loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)
        
#         runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=100, save_dir=r'.\codes\saved_models', save_filename=f'best_model_{i+4}.pickle', transform_dcit=transform_dict)

#         fig, axes = plt.subplots(1, 2)
#         fig.set_tight_layout(True)
#         plot(runner, axes)
#         plt.savefig(fr'.\codes\figs\Figure_{i+4}.png')
#         plt.close()

# # 5. Linear + Adam + nHidden(100, 1000)
# # 测试不同隐藏层的效果
# """
# 9. 100 = 0.947 -> 0.9422 -> 0.9573
# 8. 600 = 0.9572 -> 0.9556
# 10. 1000 = 0.9494 -> 0.9541 -> 0.9281
# 结论：增加隐藏层的数量不一定会提升模型性能。可能原因：优化更复杂。
# """
# nHiddens = [100, 1000]
# for i, nHidden in enumerate(nHiddens):
#         print(f"Start training with nHidden {nHidden}")
#         linear_model = nn.models.Model_MLP([train_imgs.shape[-1], nHidden, 10], 'ReLU', [1e-4, 1e-4])
#         optimizer = nn.optimizer.Adam(init_lr=0.06, model=linear_model)
#         scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
#         loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)
#         runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)
#         runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=100, save_dir=r'.\codes\saved_models', save_filename=f'best_model_{i+9}.pickle')

#         fig, axes = plt.subplots(1, 2)
#         fig.set_tight_layout(True)
#         plot(runner, axes)
#         plt.savefig(fr'.\codes\figs\Figure_{i+9}.png')
#         plt.close()

# # 6. Linear + Adam - RMSProp
# """
# 8. use RMSProp = 0.9572 -> 0.9556
# 11. not use RMSProp = 0.9374 -> 0.9379
# 结论：RMSProp 提供了自适应学习率，有助于优化。
# """

# linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4])
# optimizer = nn.optimizer.Adam(init_lr=0.06, model=linear_model, rms_prop=False)
# scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
# loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)
# runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)
# print("Start training model 11")
# runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=100, save_dir=r'.\codes\saved_models', save_filename='best_model_11.pickle')

# fig, axes = plt.subplots(1, 2)
# fig.set_tight_layout(True)
# plot(runner, axes)
# plt.savefig(r'.\codes\figs\Figure_11.png')
# plt.close()


# # 7. Linear + Adam + nHidden in {600, 1000} (train longer)
# """
# 在实验5中，我们发现增加隐藏层的数量并没有提升模型性能。可能是因为参数量增加，需要更多的训练时间。因此，这里训练更长的时间，看看是否有提升。
# 600: 0.9443 -> 0.967
# 1000: 0.9374 -> 0.9676
# 可能发生了过拟合，或者是优化问题。尝试降低学习率。
# 可见0.06的学习率过高，导致优化问题。
# """
# nHiddens = [600, 1000]
# for i, nHidden in enumerate(nHiddens):
#         print(f"Start training with nHidden {nHidden}")
#         linear_model = nn.models.Model_MLP([train_imgs.shape[-1], nHidden, 10], 'ReLU', [1e-4, 1e-4])
#         optimizer = nn.optimizer.Adam(init_lr=0.01, model=linear_model) # 学习率降低为0.01
#         scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
#         loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)
#         runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)
#         runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=10, log_iters=300, save_dir=r'.\codes\saved_models', save_filename=f'best_model_{i+12}.pickle')

#         fig, axes = plt.subplots(1, 2)
#         fig.set_tight_layout(True)
#         plot(runner, axes)
#         plt.savefig(fr'.\codes\figs\Figure_{i+12}.png')
#         plt.close()

# # 8. Linear + SGD + L2 regularization
# """
# 在 SGD 中，权重衰减等价于 L2 正则化。本实验比较使用 L2 正则化和不使用 L2 正则化的效果。
# no L2 regularization: 0.9381
# L2 regularization: 0.9361
# """
# linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4])
# optimizer = nn.optimizer.SGD(init_lr=0.06, model=linear_model)
# scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
# loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)
# runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)
# print("Start training model 14")
# runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=100, save_dir=r'.\codes\saved_models', save_filename='best_model_14.pickle')

# fig, axes = plt.subplots(1, 2)
# fig.set_tight_layout(True)
# plot(runner, axes)
# plt.savefig(r'.\codes\figs\Figure_14.png')
# plt.close()

# # 9. Linear + Adam + lower learning rate
# """
# 在实验7中，我们发现学习率过高会导致优化问题。因此，这里尝试使用 1e-3 和 1e-4 的学习率。
# 1e-3:0.9142
# 1e-4:0.7634
# 结论：更小的学习率没有减小训练损失的震荡，反而可能陷入局部最小值。可能0.01就是一个不错的选择。
# """
# lr_rates = [1e-3, 1e-4]
# for i, lr_rate in enumerate(lr_rates):
#         print(f"Start training with lr_rate {lr_rate}")
#         linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4])
#         optimizer = nn.optimizer.Adam(init_lr=lr_rate, model=linear_model)
#         scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
#         loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)
#         runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)
#         runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=300, save_dir=r'.\codes\saved_models', save_filename=f'best_model_{i+15}.pickle')

#         fig, axes = plt.subplots(1, 2)
#         fig.set_tight_layout(True)
#         plot(runner, axes)
#         plt.savefig(fr'.\codes\figs\Figure_{i+15}.png')
#         plt.close()

# # 10. Linear + Adam + bacth_size
# """
# 探究不同 batch_size 对于模型的影响。
# 16:0.9615
# 32:0.9595
# 64:0.9612
# 128:0.9629
# 结论：batch_size 影响不大，但是 128 效果最好。
# """
# batch_sizes = [16, 32, 64, 128]
# for i, batch_size in enumerate(batch_sizes):
#         print(f"Start training with batch_size {batch_size}")
#         linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4])
#         optimizer = nn.optimizer.Adam(init_lr=0.01, model=linear_model)
#         scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
#         loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)
#         runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)
#         runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=300, save_dir=r'.\codes\saved_models', save_filename=f'best_model_{i+17}.pickle', batch_size=batch_size)

#         fig, axes = plt.subplots(1, 2)
#         fig.set_tight_layout(True)
#         plot(runner, axes)
#         plt.savefig(fr'.\codes\figs\Figure_{i+17}.png')
#         plt.close()

# 11. Linear + Adam + bacth_size(with corresponding learning rate)
"""
按照线性缩放的原则，batch_size 变为 k 倍，学习率也需要变为 k 倍。
16 + 0.005 :0.9541
64 + 0.02 :0.9549
128 + 0.04 :0.9554
基本没有任何区别，还不如全用0.01.
"""
batch_sizes = [16, 64, 128]
lr_rates = [0.005, 0.02, 0.04]
for i, (batch_size, lr_rate) in enumerate(zip(batch_sizes, lr_rates)):
        print(f"Start training with batch_size {batch_size} and lr_rate {lr_rate}")
        linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4])
        optimizer = nn.optimizer.Adam(init_lr=lr_rate, model=linear_model)
        scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
        loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)
        runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)
        runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=300, save_dir=r'.\codes\saved_models', save_filename=f'best_model_{i+21}.pickle', batch_size=batch_size)

        fig, axes = plt.subplots(1, 2)
        fig.set_tight_layout(True)
        plot(runner, axes)
        plt.savefig(fr'.\codes\figs\Figure_{i+21}.png')

# _, axes = plt.subplots(1, 2)
# axes.reshape(-1)
# _.set_tight_layout(1)
# plot(runner, axes)

# plt.show()

