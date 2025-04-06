from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]

                # layer.clear_grad()


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu):
        super().__init__(init_lr, model)
        self.mu = mu
        self.v = {}  # 格式: {id(layer): {'W': v_W, 'b': v_b}}
        
        # 初始化动量缓存
        for layer in self.model.layers:
            if layer.optimizable:
                self.v[id(layer)] = {
                    key: np.zeros_like(param)
                    for key, param in layer.params.items()
                }
    
    def step(self):
        for layer in self.model.layers:
            if not layer.optimizable:
                continue
                
            layer_v = self.v[id(layer)]  # 获取该层的动量缓存
            
            for key, param in layer.params.items():
                # 合并权重衰减到梯度
                if layer.weight_decay:
                    layer.grads[key] += layer.weight_decay_lambda * param
                
                # 更新动量
                layer_v[key] = self.mu * layer_v[key] - self.init_lr * layer.grads[key]
                
                # 更新参数
                param += layer_v[key]


# Adam optimizer
class Adam(Optimizer):
    def __init__(self, init_lr, model, beta1=0.9, beta2=0.999, epsilon=1e-8, rms_prop=True):
        super().__init__(init_lr, model)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # 格式: {id(layer): {'W': m_W, 'b': m_b}}
        self.v = {}  # 格式: {id(layer): {'W': v_W, 'b': v_b}}
        self.step_count = 0 # 初始化迭代次数
        self.rms_prop = rms_prop # 是否考虑二阶动量
        
        # 初始化动量缓存
        for layer in self.model.layers:
            if layer.optimizable:
                self.m[id(layer)] = {
                    key: np.zeros_like(param)
                    for key, param in layer.params.items()
                }
                self.v[id(layer)] = {
                    key: np.zeros_like(param)
                    for key, param in layer.params.items()
                }
    
    def step(self):

        # 更新迭代次数
        self.step_count += 1

        for layer in self.model.layers:
            if not layer.optimizable:
                continue
                
            layer_m = self.m[id(layer)]  # 获取该层的一阶动量缓存
            layer_v = self.v[id(layer)]  # 获取该层的二阶动量缓存
            
            for key, param in layer.params.items():
                # 合并权重衰减到梯度
                if layer.weight_decay:
                    layer.grads[key] += layer.weight_decay_lambda * param
                
                # 更新动量
                layer_m[key] = self.beta1 * layer_m[key] + (1 - self.beta1) * layer.grads[key]
                layer_v[key] = self.beta2 * layer_v[key] + (1 - self.beta2) * layer.grads[key] ** 2
                
                # 计算 bias-corrected 动量
                m_hat = layer_m[key] / (1 - self.beta1 ** (self.step_count + 1))
                v_hat = layer_v[key] / (1 - self.beta2 ** (self.step_count + 1))
                
                # 更新参数
                if self.rms_prop:
                    param -= self.init_lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
                else:
                    param -= self.init_lr * m_hat 
                
