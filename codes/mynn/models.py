from .op import *
import pickle


class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for i, layer in enumerate(reversed(self.layers)):
            grads = layer.backward(grads)
        # if hasattr(layer, 'grads'):
        #     print(f"Layer {i} ({layer.__class__.__name__}) grad norm: {np.linalg.norm(grads)}")
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        self.layers = []    # 初始化移到这里，避免重复初始化

        for i in range(len(self.size_list) - 1):
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            layer.W = param_list[i + 2]['W']
            layer.b = param_list[i + 2]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = param_list[i + 2]['weight_decay']
            layer.weight_decay_lambda = param_list[i+2]['lambda']
            if self.act_func == 'Logistic':
                raise NotImplemented
            elif self.act_func == 'ReLU':
                layer_f = ReLU()
            self.layers.append(layer)
            if i < len(self.size_list) - 2:
                self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self, input_shape=(1, 28, 28), num_classes=10):
        """
        初始化一个简单的CNN模型，结构为：
        Conv1 → ReLU → MaxPool → Conv2 → ReLU → MaxPool → Flatten → Linear → Softmax

        参数:
            input_shape: 输入张量形状 (C, H, W)，默认单通道28x28（MNIST）
            num_classes: 输出类别数
        """
        super().__init__()
        C, H, W = input_shape
        self.input_shape = input_shape
        self.num_classes = num_classes

        # 卷积层1，卷积核大小5x5，输出通道数16，步长1，填充0。->[16, 24, 24]
        self.conv1 = conv2D(
            in_channels=C,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0,
            initialize_method=np.random.normal,
            weight_decay=False,
            weight_decay_lambda=1e-8
        )
        # ReLU激活
        self.relu1 = ReLU()
        # 最大池化层1，池化核大小2x2，步长2，填充0。->[16, 12, 12]
        self.maxpool1 = MaxPool2D(kernel_size=2, stride=2, padding=0)

        # 卷积层2，卷积核大小5x5，输出通道数32，步长1，填充0。->[32, 8, 8]
        self.conv2 = conv2D(
            in_channels=16,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=0,
            initialize_method=np.random.normal,
            weight_decay=False,
            weight_decay_lambda=1e-8
        )
        # ReLU激活
        self.relu2 = ReLU()
        # 最大池化层2，池化核大小2x2，步长2，填充0。->[32, 4, 4]
        self.maxpool2 = MaxPool2D(kernel_size=2, stride=2, padding=0)

        # 全连接层，输入维度32x4x4，输出维度10。
        self.fc1 = Linear(
            in_dim=32 * 4 * 4,
            out_dim=10,
            initialize_method=np.random.normal,
            weight_decay=False,
            weight_decay_lambda=1e-8
        )
        # 存储所有层
        self.layers = [self.conv1, self.relu1, self.maxpool1, self.conv2, self.relu2, self.maxpool2, self.fc1]
        # 存储所有可训练层（用于梯度更新）
        self.trainable_layers = [self.conv1, self.conv2, self.fc1]


    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):

        # 调整输入形状 [batch, 28*28] -> [batch, 1, 28, 28]
        X = X.reshape(X.shape[0], *self.input_shape)

        X = self.conv1(X)
        X = self.relu1(X)
        X = self.maxpool1(X)
        X = self.conv2(X)
        X = self.relu2(X)
        X = self.maxpool2(X)

        # 展平输入全连接层
        X = X.reshape(X.shape[0], -1)
        X = self.fc1(X)

        # 输出类别概率
        # X = softmax(X)
        return X

    # def backward(self, loss_grad):
    #     grads = loss_grad
    #     # print(f"Initial grad shape: {grads.shape}")  # 应为 (batch_size, 10)
        
    #     for layer in reversed(self.layers):
    #         grads = layer.backward(grads)
    #         # print(f"After {layer.__class__.__name__}: {grads.shape}")
            
    #         if layer == self.fc1:
    #             grads = grads.reshape(grads.shape[0], 32, 4, 4)
    #             # print(f"After reshape: {grads.shape}")  # 应为 (batch_size, 32, 4, 4)
    #     return grads
    def backward(self, loss_grad):
        grads = loss_grad
        for i, layer in enumerate(reversed(self.layers)):
            grads = layer.backward(grads)
            if layer == self.fc1:
                grads = grads.reshape(grads.shape[0], 32, 4, 4)
            # if hasattr(layer, 'grads'):
            #     print(f"Layer {i} ({layer.__class__.__name__}) grad norm: {np.linalg.norm(grads)}")
        return grads
    
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)

        # 加载参数
        self.conv1.params['W'] = param_list['conv1_params']['W']
        self.conv1.params['b'] = param_list['conv1_params']['b']
        self.conv1.weight_decay = param_list['conv1_params']['weight_decay']
        self.conv1.weight_decay_lambda = param_list['conv1_params']['lambda']

        self.conv2.params['W'] = param_list['conv2_params']['W']
        self.conv2.params['b'] = param_list['conv2_params']['b']
        self.conv2.weight_decay = param_list['conv2_params']['weight_decay']
        self.conv2.weight_decay_lambda = param_list['conv2_params']['lambda']

        self.fc1.params['W'] = param_list['fc1_params']['W']
        self.fc1.params['b'] = param_list['fc1_params']['b']
        self.fc1.weight_decay = param_list['fc1_params']['weight_decay']
        self.fc1.weight_decay_lambda = param_list['fc1_params']['lambda']

        
    def save_model(self, save_path):
        model_info = {
            'input_shape':self.input_shape,
            'num_classes':self.num_classes,
            'conv1_params':{
                'W':self.conv1.params['W'],
                'b':self.conv1.params['b'],
                'weight_decay':self.conv1.weight_decay,
                'lambda':self.conv1.weight_decay_lambda
            },
            'conv2_params':{
                'W':self.conv2.params['W'],
                'b':self.conv2.params['b'],
                'weight_decay':self.conv2.weight_decay,
                'lambda':self.conv2.weight_decay_lambda
            },
            'fc1_params':{
                'W':self.fc1.params['W'],
                'b':self.fc1.params['b'],
                'weight_decay':self.fc1.weight_decay,
                'lambda':self.fc1.weight_decay_lambda
            }
        }   

        with open(save_path, 'wb') as f:
            pickle.dump(model_info, f)