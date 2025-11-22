# Initialize a surrogate function with learnable parameters
# ATan_learnable(): The LearnableAlpha is learnable during backward propagation

import torch
import math
import torch.nn as nn
# from spikingjelly.activation_based.auto_cuda import cfunction

# tab4_str = '\t\t\t\t'  # used for aligning code


def heaviside(x: torch.Tensor):
    return (x >= 0).to(x)


# backward 方法必须为 forward 方法的每一个输入都返回一个对应的梯度值
# g(x,a) = 1/pi * arctan(pi * a * x/2) + 1/2
# g'(x) = a / (2 * [ 1 + (pi * a * x / 2) ^ 2])
# g'(a) = x / (2 * [1 + (pi * x * a / 2) ^ 2])
# grad_output: $\frac{\partial L}{\partial y}$
@torch.jit.script
def atan_backward_learnbale(grad_output: torch.Tensor, x: torch.Trugensor, alpha: torch.Tensor):     # 传入atan_backward的LearnableAlpha是float还是tensor？
    LearnableAlpha = alpha
    grad_g_x = LearnableAlpha / 2 / (1 + (math.pi / 2 * LearnableAlpha * x).pow(2)) * grad_output
    grad_g_LearnableAlpha = torch.sum((x / (2 * (1 + (math.pi / 2 * LearnableAlpha * x).pow(2)))) * grad_output)
    return grad_g_x, grad_g_LearnableAlpha


class SurrogateFunctionBase_learnable(nn.Module):
    def __init__(self, spiking=True):
        super().__init__()
        self.spiking = spiking

    def extra_repr(self) -> str:
        return f'spiking = {self.spiking}'

    @staticmethod
    def spiking_function(x: torch.Tensor):
        raise NotImplementedError

    @staticmethod
    def primitive_function(self, x: torch.Tensor):
        raise NotImplementedError



class atan_learnable(torch.autograd.Function):
    # ctx: context;
    # ctx in forward: Save those tensors/vars used in gradient calculation during backward propagation
    @staticmethod
    def forward(ctx, x, LearnableAlpha):
        # if x.requires_grad:
        #     ctx.save_for_backward(x)
        #     ctx.LearnableAlpha = LearnableAlpha
        ctx.save_for_backward(x, LearnableAlpha)
        return heaviside(x)

    @staticmethod
    # backward 方法必须为 forward 方法的每一个输入都返回一个对应的梯度值
    # grad_output: 从上一层传来的梯度
    def backward(ctx, grad_output):
        # 取出保存的张量
        return atan_backward_learnbale(grad_output=grad_output,x=ctx.saved_tensors[0], alpha=ctx.saved_tensors[1])


# nn.Parameter: 当你调用 model.parameters() 时，这个 LearnableAlpha 会被包含在内。
# 当你创建优化器（如 optim.Adam(model.parameters())）时，优化器会“知道”它需要更新这个 LearnableAlpha
class ATan_learnable(SurrogateFunctionBase_learnable):
    def __init__(self, init_alpha=2.0, spiking=True, learnable=True):
        super().__init__(spiking=spiking)
        # 关键：创建并 *拥有* 可学习的 alpha 参数
        self.LearnableAlpha = nn.Parameter(torch.tensor((init_alpha)))
        self.learnable = learnable


    if self.learnable:
        # 让ATan_learnable 中的alpha可学习

    else:
        # ATan_learnable 中的alpha直接设为init_alpha并保持不变



    def forward(self, x):
        if self.spiking:
            return self.spiking_function(x)
        else:
            return self.primitive_function(x)

    # .apply(x, alpha) 是调用和执行一个自定义的 torch.autograd.Function
    # （比如你写的 atan 或 atan_learnable 类）的标准方法
    def spiking_function(self, x: torch.Tensor):
        return atan_learnable.apply(x, self.LearnableAlpha)

    # g(x,a) = 1/pi * arctan(pi * a * x/2) + 1/2
    # @torch.jit.script
    def primitive_function(self, x: torch.Tensor):
        return (math.pi / 2 * self.LearnableAlpha * x).atan() / math.pi + 0.5

    # def cuda_code(self, x: str, y: str, dtype='fp32'):
    #     sg_name = 'sg_' + self._get_name()
    #     alpha = str(self.alpha) + 'f'
    #     code = f'''
    #         {tab4_str}{self.cuda_code_start_comments()}
    #     '''
    #     if dtype == 'fp32':
    #         code += f'''
    #         {tab4_str}const float {sg_name}_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * {alpha} * {x};
    #         {tab4_str}const float {y} = {alpha} / 2.0f / (1.0f + {sg_name}_M_PI_2__alpha__x * {sg_name}_M_PI_2__alpha__x);
    #         '''
    #     elif dtype == 'fp16':
    #         code += f'''
    #         {tab4_str}const half2 {sg_name}_alpha =  __float2half2_rn({alpha});
    #         {tab4_str}const half2 {sg_name}_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), {sg_name}_alpha), {x});
    #         {tab4_str}const half2 {y} = __h2div(__h2div({sg_name}_alpha, __float2half2_rn(2.0f)), __hfma2({sg_name}_M_PI_2__alpha__x, {sg_name}_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
    #         '''
    #     else:
    #         raise NotImplementedError
    #     code += f'''
    #         {tab4_str}{self.cuda_code_end_comments()}
    #     '''
    #     return code
    #
    # def cuda_codes(self, y: str, x: str, dtype: str):
    #     return cfunction.atan_backward(y=y, x=x, alpha=self.alpha, dtype=dtype)