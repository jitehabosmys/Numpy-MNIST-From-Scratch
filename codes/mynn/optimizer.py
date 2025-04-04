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
