from abc import abstractmethod, ABC
import numpy as np
np.random.seed(309)

class Layer(ABC):
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, grads):
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()

        self.params = {}
        if initialize_method == 'Xavier':
            fan_in = in_dim
            fan_out = out_dim
            scale = np.sqrt(2.0 / (fan_in + fan_out))  # ReLU的修正因子
            self.params['W'] = np.random.normal(scale=scale, size=(in_dim, out_dim))
            self.params['b'] = np.random.normal(size=(1, out_dim))
        else:    
            self.params['W'] = initialize_method(size=(in_dim, out_dim))
            self.params['b'] = initialize_method(size=(1, out_dim))

        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return np.dot(X, self.params['W']) + self.params['b']   

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        self.grads['W'] = np.dot(self.input.T, grad)
        # add weight decay
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.params['W']

        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)   # 等价于 1_B * grad, 1_B 是行向量
        grad_input = np.dot(grad, self.params['W'].T)
        return grad_input

    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.params = {}

        # Xavier初始化
        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size
        scale = np.sqrt(2.0 / (fan_in + fan_out))  # ReLU的修正因子
        self.params['W'] = np.random.normal(scale=scale, size=(out_channels, in_channels, kernel_size, kernel_size))
        self.params['b'] = np.zeros(out_channels)  # 偏置初始化为0更稳定

        # self.params['W'] = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
        # self.params['b'] = initialize_method(size=(out_channels, )) 
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.


        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """

        self.input = X
        batch_size, in_channels, H, W = X.shape
        out_channels = self.params['W'].shape[0]
        k = self.kernel_size

        # 计算输出形状

        new_H = (H - k) // self.stride + 1
        new_W = (W - k) // self.stride + 1
        # 检查最后一个窗口是否越界
        if (new_H - 1) * self.stride + k > H:
            new_H -= 1
        if (new_W - 1) * self.stride + k > W:
            new_W -= 1
        output = np.zeros((batch_size, out_channels, new_H, new_W))

        # 卷积核权重调整为 [1, out, in, k, k]
        W_expanded = self.params['W'][None, ...]

        for i in range(new_H):
            for j in range(new_W):
                h_start = i * self.stride
                h_end = h_start + k
                w_start = j * self.stride
                w_end = w_start + k

                # 提取当前窗口区域 [batch, in, k, k]
                patch = X[:, :, h_start:h_end, w_start:w_end]
                # 扩展维度以支持广播 [batch, 1, in, k, k]
                patch_expanded = patch[:, None, :, :, :]

                # 计算卷积
                product = patch_expanded * W_expanded  # [batch, out, in, k, k]
                summed = np.sum(product, axis=(2, 3, 4))  # [batch, out]

                # 加上偏置（自动广播到[batch, out]）
                output[:, :, i, j] = summed + self.params['b']

        return output


    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        X = self.input 
        H, W = X.shape[2:]
        _, _, new_H, new_W = grads.shape
        k = self.kernel_size

        # 初始化梯度容器
        self.grads['W'] = np.zeros_like(self.params['W']) # [out, in, k, k]
        self.grads['b'] = np.zeros_like(self.params['b']) # [out]
        grad_input = np.zeros_like(X)

        # 计算偏置梯度
        self.grads['b'] = np.sum(grads, axis=(0, 2, 3))  # [out]

        # 计算卷积核梯度
        for i in range(new_H):
            for j in range(new_W):
                h_start = i * self.stride
                h_end = h_start + k
                w_start = j * self.stride
                w_end = w_start + k

                # 边界检查
                if h_end > H or w_end > W:
                    continue

                # 提取当前窗口区域 [batch, in, k, k]
                patch = X[:, :, h_start:h_end, w_start:w_end]
                # 扩展窗口维度以支持广播 [batch, 1, in, k, k]
                patch_expanded = patch[:, None, :, :, :]

                # 扩展梯度维度以支持广播 [batch, out, 1, 1, 1]
                grad_expanded = grads[:, :, i, j][:, :, None, None, None]

                # 累加卷积核梯度
                self.grads['W'] += np.sum(grad_expanded * patch_expanded, axis=0)  # [out, in, k, k]

        # add weight decay
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.params['W']

        # 计算输入梯度
        for i in range(new_H):
            for j in range(new_W):
                h_start = i * self.stride
                h_end = h_start + k
                w_start = j * self.stride
                w_end = w_start + k

                # 边界检查
                if h_end > H or w_end > W:
                    continue

                # 扩展卷积核维度以支持广播 [1, out, in, k, k]
                W_expanded = self.params['W'][None, :, :, :, :]
                # 扩展梯度维度以支持广播 [batch, out, 1, 1, 1]
                grad_expanded = grads[:, :, i, j][:, :, None, None, None]

                # 计算当前窗口对输入梯度的贡献
                grad_contribution = np.sum(grad_expanded * W_expanded, axis=1)  # [batch, in, k, k]

                # 累加窗口对输入梯度的贡献
                grad_input[:, :, h_start:h_end, w_start:w_end] += grad_contribution

        return grad_input

    # def backward(self, grads):
    #     # 仅保留卷积核梯度计算，跳过输入梯度计算（临时测试）
    #     self.grads['W'] = np.zeros_like(self.params['W'])
    #     self.grads['b'] = np.sum(grads, axis=(0, 2, 3))
    #     grad_input = np.zeros_like(self.input)  # 返回全零梯度
    #     return grad_input

    

    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.has_softmax = True
        self.max_classes = max_classes
        self.grads = None
        self.input = None
        self.labels = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        # / ---- your codes here ----/
        self.input = predicts
        self.labels = labels
        if self.has_softmax:
            predicts = softmax(predicts)
        
        # 将 labels 转化为 one-hot 编码
        one_hot_labels = np.zeros_like(predicts)
        one_hot_labels[np.arange(labels.shape[0]), labels] = 1
        
        # 计算交叉熵
        loss = -np.sum(one_hot_labels * np.log(predicts + 1e-10), axis=1)
        loss = np.mean(loss)

        return loss
                
    
    def backward(self):
        # first compute the grads from the loss to the input
        # / ---- your codes here ----/

        # 将 labels 转化为 one-hot 编码
        one_hot_labels = np.zeros_like(self.input)
        one_hot_labels[np.arange(self.labels.shape[0]), self.labels] = 1

        # 计算损失函数对输入的梯度
        if self.has_softmax:
            grads = softmax(self.input) - one_hot_labels
        else:
            grads = self.input - one_hot_labels
        
        # 除以 batch_size 进行归一化
        grads /= self.input.shape[0]
        self.grads = grads

        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition

class MaxPool2D(Layer):
    """
    A max pooling layer.
    """
    def __init__(self, kernel_size=3, stride=1, padding=0) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input = None # Record the input for backward process.
        self.grads = None
        self.max_idx = None # 记录最大值位置

        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, C, H, W]
        no padding
        """

        self.input = X
        batch_size, C, H, W = X.shape
        k = self.kernel_size

        # 计算输出形状

        new_H = (H - k) // self.stride + 1
        new_W = (W - k) // self.stride + 1
        # 检查最后一个窗口是否越界
        if (new_H - 1) * self.stride + k > H:
            new_H -= 1
        if (new_W - 1) * self.stride + k > W:
            new_W -= 1

        # 初始化输出和最大值位置
        output = np.zeros((batch_size, C, new_H, new_W))
        self.max_idx = np.zeros((batch_size, C, new_H, new_W, 2), dtype=int)

        for i in range(new_H):
            for j in range(new_W):
                h_start = i * self.stride
                h_end = h_start + k
                w_start = j * self.stride
                w_end = w_start + k

                # 提取当前窗口区域 [batch, in, k, k]
                patch = X[:, :, h_start:h_end, w_start:w_end]
                # 最大值
                output[:, :, i, j] = np.max(patch, axis=(2, 3))

                # 记录最大值位置，用于反向传播
                flat_patch = patch.reshape(batch_size, C, -1)  # [batch, C, k*k]
                max_pos_flat = np.argmax(flat_patch, axis=2)   # [batch, C]
                max_pos_h = max_pos_flat // k                  # 行坐标
                max_pos_w = max_pos_flat % k                   # 列坐标

                self.max_idx[:, :, i, j, 0] = h_start + max_pos_h
                self.max_idx[:, :, i, j, 1] = w_start + max_pos_w
        return output


    def backward(self, grads):
        """
        grads : [batch_size, C, new_H, new_W]
        """
        X = self.input 
        batch_size, C, _, _ = X.shape
        _, _, new_H, new_W = grads.shape
        k = self.kernel_size

        # 初始化梯度容器
        grad_input = np.zeros_like(X)

        # 计算输入梯度
        for i in range(new_H):
            for j in range(new_W):
                # 获取当前窗口最大值位置 [batch, C]
                max_pos_h = self.max_idx[:, :, i, j, 0]
                max_pos_w = self.max_idx[:, :, i, j, 1]

                # 将梯度累加到对应位置
                batch_indices = np.arange(batch_size)[:, None]  # [batch, 1]
                channel_indices = np.arange(C)[None, :]        # [1, C]
                grad_input[batch_indices, channel_indices, self.max_idx[:, :, i, j, 0], self.max_idx[:, :, i, j, 1]] += grads[:, :, i, j]      


        return grad_input

def clear_grad(self):
    pass


# class BatchNorm1D(Layer):
#     """
#     A batch normalization layer.
#     """
#     def __init__(self, input_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
#         super().__init__()
#         self.input_dim = input_dim
#         self.initialize_method = initialize_method
#         self.weight_decay = weight_decay
#         self.weight_decay_lambda = weight_decay_lambda
#         self.params = {}
#         self.params['gamma'] = np.ones(input_dim)
#         self.params['beta'] = np.zeros(input_dim)
#         self.params['running_mean'] = np.zeros(input_dim)
#         self.grads = {}


# 测试代码
if __name__ == '__main__':

    # # 1. 测试 MaxPool2D 层
    # # 测试前向传播
    # X = np.random.randn(2, 3, 6, 6)  # [batch=2, C=3, H=6, W=6]
    # maxpool = MaxPool2D(kernel_size=2, stride=2)
    # output = maxpool.forward(X)
    # print("Forward output shape:", output.shape)  # 应输出 (2, 3, 3, 3)

    # # 测试反向传播
    # grads = np.ones_like(output)
    # dX = maxpool.backward(grads)
    # print("Backward gradient sum:", np.sum(dX))  # 应等于 np.prod(grads.shape)
    # print(np.prod(grads.shape))  # 应该等于 2*3*3*3 = 54

    # # 再测试前向传播
    # Y = np.arange(16).reshape(1, 1, 4, 4)
    # maxpool = MaxPool2D(kernel_size=2, stride=2)
    # output = maxpool.forward(Y)
    # # print(Y)
    # print("Forward output:", output)    # 应输出 [5, 7; 13, 15]

    # # 2. 测试 Linear 层
    # # 测试前向传播
    # X = np.random.randn(2, 3)  # [batch=2, input_dim=3]
    # linear = Linear(in_dim=3, out_dim=5)
    # output = linear.forward(X)
    # print("Forward output shape:", output.shape)  # 应输出 [2, 5]

    # # 3. 测试 conv 层
    # # 测试前向传播
    # X = np.random.randn(2, 3, 6, 6)  # [batch=2, in_channel=3, H=6, W=6]
    # conv = conv2D(in_channels=3, out_channels=5, kernel_size=3, stride=1, padding=0)
    # output = conv.forward(X)
    # print("Forward output shape:", output.shape)  # 应输出 [2, 5, 4, 4]

    # 测试卷积梯度
    X = np.random.randn(32, 1, 28, 28)
    conv = conv2D(in_channels=1, out_channels=16, kernel_size=5)
    out = conv.forward(X)
    grads = np.random.randn(*out.shape)
    dX = conv.backward(grads)
    print("Conv grad norms - W:", np.linalg.norm(conv.grads['W']), "input:", np.linalg.norm(dX))

    






