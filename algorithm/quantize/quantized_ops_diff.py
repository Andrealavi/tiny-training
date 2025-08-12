"""
In this file, we implement quantized ops to simulate the device-side training of quantized models
"""

import torch
import torch.nn.functional as F
from .quantized_ops import to_pt, QuantizedAvgPool, QuantizedConv2d, QuantizedElementwise, QuantizedMbBlock

QUANTIZED_GRADIENT = False
ROUNDING = 'round'
CONV_W_GRAD = True

# sanity check
if QUANTIZED_GRADIENT:
    raise NotImplementedError


def round_tensor(x):
    if ROUNDING == 'round':
        return x.round()
    elif ROUNDING == 'floor':
        return x.int().float() # int() truncates the values
    elif ROUNDING == 'debug':
        return x
    else:
        raise NotImplementedError

"""
This class implements a custom autograd Function. An autograd function
is used to compute the partial derivative using chain rule.
It can compute the derivative forward or backward.
In order to implement a custom function using pytorch you need to
extend torch.autograd.Function and implement forward and backward methods
"""
class _QuantizedAvgPoolFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor): # Computes average pooling
        ctx.input_shape = x.shape # ctx stores information for the backward pass
        x= x.to(torch.float32)
        assert x.dtype == torch.float32
        x = x.mean([-1, -2], keepdim=True) # Computes the mean of the last two dimensions
        return round_tensor(x) # rounded to quantize it

    @staticmethod
    def backward(ctx, grad_output): # Computes gradients during backpropagation
        input_shape = ctx.input_shape
        # Gradient is repeated and normalized
        grad_input = grad_output.repeat(1, 1, *input_shape[-2:]) / (input_shape[-1] * input_shape[-2])
        return grad_input


"""
This class is likely used to not call forward method directly
"""
class QuantizedAvgPoolDiff(QuantizedAvgPool):
    def forward(self, x):
        x = _QuantizedAvgPoolFunc.apply(x)
        return x


class _QuantizedElementwiseAddFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1: torch.Tensor, x2: torch.Tensor, zero_x1, zero_x2, zero_y, scale_x1, scale_x2, scale_y):
        # ensure x1 and x2 are int
        # Quantize input tensors
        x1 = x1.round()  
        x2 = x2.round()
        assert x1.shape == x2.shape
        # Saves tensors intended to be used in the backward pass
        ctx.save_for_backward(scale_x1, scale_x2, scale_y)

        # Approximates the real value input tensor using scale factor
        # x = (x_int - zero) * scaling_factor
        x1 = (x1 - zero_x1) * scale_x1
        x2 = (x2 - zero_x2) * scale_x2

        # The sum output is requantized
        out = x1 + x2
        out = round_tensor(out / scale_y)
        out = out + zero_y
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # only return gradient of zero_y, zero_x1, zero_x2, x1, x2
        scale_x1, scale_x2, scale_y = ctx.saved_tensors

        # See notes on why the gradient of zero_y is computed in this way
        grad_zero_y = grad_output.sum([0, 2, 3])

        # See notes on why the gradient is rescaled
        grad_sum = grad_output / scale_y.item()

        # Quantize the gradients since the input was quantized
        grad_x1 = grad_sum * scale_x1.item()
        grad_x2 = grad_sum * scale_x2.item()

        # Negative sign is due to the fact that zero is subtracted
        grad_zero_x1 = - grad_x1.sum([0, 2, 3])
        grad_zero_x2 = - grad_x2.sum([0, 2, 3])
        
        return grad_x1, grad_x2, grad_zero_x1, grad_zero_x2, grad_zero_y, None, None, None


class QuantizedElementwiseDiff(QuantizedElementwise):
    def __init__(self, operator, zero_x1, zero_x2, zero_y, scale_x1, scale_x2, scale_y):
        super().__init__(operator, zero_x1, zero_x2, zero_y, scale_x1, scale_x2, scale_y)
        assert self.operator == 'add'  # for mult, we do not support bias-only update

    def forward(self, x1, x2):
        return _QuantizedElementwiseAddFunc.apply(x1, x2,
                                                  self.zero_x1, self.zero_x2, self.zero_y,
                                                  self.scale_x1, self.scale_x2, self.scale_y)


class _TruncateActivationRange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a_bit):
        ctx.a_bit = a_bit
        # Checks whether tensor values are in the allowed interval
        binary_mask = (- 2 ** (a_bit - 1) <= x) & (x <= 2 ** (a_bit - 1) - 1)

        ctx.save_for_backward(binary_mask)
        return x.clamp(- 2 ** (a_bit - 1), 2 ** (a_bit - 1) - 1)

    @staticmethod
    def backward(ctx, grad_output):
        binary_mask, = ctx.saved_tensors
        
        # If an element is outside the interval its gradient is zero
        grad_x = grad_output * binary_mask
        return grad_x, None


class _QuantizedConv2dFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, zero_x, zero_y, effective_scale, stride, padding, dilation, groups):
        x = x.round()  # ensure x is int
        weight = weight.round()  # ensure weight is int

        ctx.stride = stride # Define how many pixel are shifted during convolution
        ctx.padding = padding
        ctx.dilation = dilation # Define the space between kernels
        ctx.groups = groups
        ctx.input_size = x.shape
        ctx.weight_size = weight.shape

        # weight = weight.int()  # - self.zero_w
        x = x - zero_x

        if CONV_W_GRAD:
            ctx.save_for_backward(weight, effective_scale, x)
        else:
            ctx.save_for_backward(weight, effective_scale)

        out = F.conv2d(x.float(), weight.float(), None, stride, padding, dilation, groups)
        out = round_tensor(out)  # ensure output is still int
        # here we allow bias saved as fp32, and round to int during inference (keep fp32 copy in memory)
        # Bias are reshaped in order to fit the number of channels
        out = out + bias.view(1, -1, 1, 1)  # Confirmed: we don't need to cast bias
        out = round_tensor(out * effective_scale.view(1, -1, 1, 1))
        out = out + zero_y
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # effective_scale = scale_x * scale_w / scale_y
        # b_quantized = b / (w_scales * x_scale), so we may wanna compute grad_b / (w_scale * x_scale)
        # which is grad_b / (effective_scale * scale_y)
        if CONV_W_GRAD:
            weight, effective_scale, _x = ctx.saved_tensors
        else:
            weight, effective_scale = ctx.saved_tensors

        grad_zero_y = grad_output.sum([0, 2, 3])
        _grad_conv_out = grad_output * effective_scale.view(1, -1, 1, 1)
        grad_bias = _grad_conv_out.sum([0, 2, 3])

        # Computes the gradient with respect to the input of the convolution
        _grad_conv_in = torch.nn.grad.conv2d_input(ctx.input_size, weight.float(), _grad_conv_out.float(),
                                                   stride=ctx.stride, padding=ctx.padding,
                                                   dilation=ctx.dilation, groups=ctx.groups)
        grad_zero_x = - _grad_conv_in.sum([0, 2, 3])
        grad_x = _grad_conv_in

        if CONV_W_GRAD:
            # Computes the gradient with respect to the weights of the convolution
            grad_w = torch.nn.grad.conv2d_weight(_x.float(), ctx.weight_size, _grad_conv_out.float(),
                                                 stride=ctx.stride, padding=ctx.padding,
                                                 dilation=ctx.dilation, groups=ctx.groups)
        else:
            grad_w = None

        from core.utils.config import configs
        if configs.backward_config.quantize_gradient:  # perform per-channel quantization
            # quantize grad_x and grad_w
            from .quantize_helper import get_weight_scales
            w_scales = get_weight_scales(grad_w, n_bit=8)
            grad_w = (grad_w / w_scales.view(-1, 1, 1, 1)).round() * w_scales.view(-1, 1, 1, 1)
            x_scales = get_weight_scales(grad_x.transpose(0, 1))
            grad_x = (grad_x / x_scales.view(1, -1, 1, 1)).round() * x_scales.view(1, -1, 1, 1)

        return grad_x, grad_w, grad_bias, grad_zero_x, grad_zero_y, None, None, None, None, None


class QuantizedConv2dDiff(QuantizedConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 zero_x=0, zero_w=0, zero_y=0,  # keep same args
                 effective_scale=None,
                 w_bit=8, a_bit=None,
                 ):
        super(QuantizedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                              padding, dilation, groups, bias, padding_mode)
        self.register_buffer('zero_x', to_pt(zero_x))
        # self.register_buffer('zero_w', to_pt(zero_w))
        self.register_buffer('zero_y', to_pt(zero_y))
        from core.utils.config import configs
        if configs.backward_config.train_scale:
            print('Note: the scale is also trained...')
            self.register_parameter('effective_scale', torch.nn.Parameter(effective_scale))
        else:
            self.register_buffer('effective_scale', effective_scale)

        self.w_bit = w_bit
        self.a_bit = a_bit if a_bit is not None else w_bit

    def forward(self, x):
        out = _QuantizedConv2dFunc.apply(x, self.weight, self.bias, self.zero_x, self.zero_y, self.effective_scale,
                                         self.stride, self.padding, self.dilation, self.groups)
        return _TruncateActivationRange.apply(out, self.a_bit)


class QuantizedMbBlockDiff(QuantizedMbBlock):
    def forward(self, x):
        out = self.conv(x)
        if self.q_add is not None:
            if self.residual_conv is not None:
                x = self.residual_conv(x)
            out = self.q_add(x, out)
            return _TruncateActivationRange.apply(out, self.a_bit)
        else:
            return out


class ScaledLinear(torch.nn.Linear):
    # a fp version of fc used for training
    def __init__(self, in_features: int, out_features: int, scale_x, zero_x, bias: bool = True,
                 device=None, dtype=None, norm_feat=False):
        super().__init__(in_features, out_features, bias, device, dtype)
        
        # Stores tensor as buffers in order to not update them during training
        self.register_buffer('scale_x', to_pt(scale_x))
        self.register_buffer('zero_x', to_pt(zero_x))

        self.norm_feat = norm_feat # True if normalization is applied
        if norm_feat:
            self.bias.data.fill_(2.)
            self.eps = 1e-5

    def forward(self, x):
        # Dequantize the input
        x = (x.squeeze(-1).squeeze(-1) - self.zero_x.detach().view(1, -1)) * self.scale_x.detach().view(1, -1)

        if self.norm_feat:
            # tensors are normalized, i.e. their norm becomes 1
            # div performs element-wise division on the tensor
            x_norm = x.div(torch.norm(x, p=2, dim=1).view(-1, 1) + self.eps)
            weight_norm = self.weight.div(torch.norm(self.weight, p=2, dim=1).view(-1, 1) + self.eps)

            # Computes cosine similarity
            cos_dist = (x_norm @ weight_norm.T) * self.bias.view(1, -1) # @ performs matrix multiplication
            
            return cos_dist
        else:
            return super().forward(x.float())

