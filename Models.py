import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional, layer, surrogate
from spikingjelly.activation_based.neuron import BaseNode, LIFNode
from torchvision import transforms
import math

import LearnableSurrogateFunction





# PLIF
class PLIFNode(BaseNode):
    def __init__(self, init_tau=2.0, v_threshold=1.0, v_reset=0.0, detach_reset=True,
                 surrogate_function=surrogate.ATan(), step_mode='s'):
        super().__init__(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate_function,
                         detach_reset=detach_reset, step_mode='s')
        # a: learnable parameter; tau = 1/k(a); k(a) = 1/(1+e^-a) => a = - ln(tau - 1)
        # sigmoid(x) = 1/(1 + e ^ (-x))
        init_a = - math.log(init_tau - 1)
        # 将一个普通的张量（Tensor）“包装”成一个“可学习的参数” (learnable parameter)。
        # nn.Parameter: 告诉 PyTorch：“这个张量不是一个固定的常量，也不是一个临时的计算结果。它是我模型的一个权重（weight），你必须在反向传播（loss.backward()） 时计算它（a） 的梯度。
        # self_a: 因为 PLIFNode 继承自 nn.Module，当您将一个 nn.Parameter 赋值给 self 的一个属性时，PyTorch 会自动将其注册到模型的“参数列表”中
        self.a = nn.Parameter(torch.tensor(init_a, dtype=torch.float))

    # EQ.6 in original paper: H_t = V_t-1 + k(a) * (- (V_t-1 - V_reset) + X_t);         k(a) = 1/(1+e^a) = sigmoid(a)
    # self.v <=> V_t-1; X_t <=> Input at step-t; self.w.sigmoid(a) <=> k(a); self.v += ... : Update membrane voltage <=> H_t (Voltage after firing a spike)
    def neuronal_charge(self, X_t: torch.Tensor):
        if self.v_reset is None:
            self.v += (X_t - self.v) * self.a.sigmoid()
        else:
            self.v += (X_t - (self.v - self.v_reset)) * self.a.sigmoid()

    def tau(self):
        return 1 / self.a.data.sigmoid().item()

    def extra_repr(self):
        return f'learnable_a={self.a}, v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, step_mode={self.step_mode}, backend={self.backend}'


# Template for processing ** static datasets **
class StaticNetBase(nn.Module):
    def __init__(self, T, init_tau, use_plif, use_max_pool, alpha_learnable, detach_reset):
        super().__init__()
        self.T = T
        self.init_tau = init_tau
        self.use_plif = use_plif
        self.use_max_pool = use_max_pool
        self.alpha_learnable = alpha_learnable
        self.detach_reset = detach_reset

        # Initialize training states
        self.train_times = 0
        self.max_test_accuracy = 0
        self.epoch = 0
        self.static_conv = None
        self.boost = nn.AvgPool1d(kernel_size=10, stride=10)
        self.fc = None
        self.conv = None

    def forward(self, x):
        x = self.static_conv
        out_spikes_counter = self.boost(self.fc(self.conv(x)).unsqueeze(1)).squeeze(1)
        for t in range(1, self.T):
            out_spikes_counter += self.boost(self.fc(self.conv(x)).unsqueeze(1)).squeeze(1)
        return out_spikes_counter




# MNIST
class MNISTNet(StaticNetBase):
    def __init__(self, T, init_tau, use_plif, use_max_pool, alpha_learnable, detach_reset):
        super().__init__(T=T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool, alpha_learnable=alpha_learnable, detach_reset=detach_reset)

        self.static_conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.conv = nn.Sequential(
            PLIFNode(init_tau=init_tau, surrogate_function=LearnableSurrogateFunction.ATan_learnable(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else
                LIFNode(tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
            nn.MaxPool2d(2,2) if use_max_pool else nn.AvgPool2d(2,2),
            nn.Conv2d(128,128,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(128),
            PLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else
                LIFNode(tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
            nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2)
        )
        self.fc = nn.Sequential(

        )

# dataset: CIFAR10, FashionMNIST, MNIST (All static graphic datasets)
def get_transforms(dataset_name):
    transforms_train = None
    transforms_test = None

    if dataset_name == 'MNIST':
        transforms_train = transforms.Compose(
            [
                transforms.RandomAffine(degrees=30, translate=(0.15,0.15), scale=(0.85,1.11)),      # Data augmentation
                transforms.ToTensor(),
                transforms.Normalize(0.1307,0.3081)     # 0.1307 & 0.3081: Global mean and std in MNIST datasets
            ]
        )
        transforms_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(0.1307,0.3081)
            ]
        )
    elif dataset_name == 'FashionMNIST':
        transforms_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(0.2860, 0.3530)
            ]
        )
        transforms_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(0.2860, 0.3530)
            ]
        )
    elif dataset_name == 'CIFAR10':
        transforms_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]
        )
        transforms_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]
        )

    return transforms_train, transforms_test
