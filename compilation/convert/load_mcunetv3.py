import os, sys, os.path as osp
import functools

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import tvm
from tvm import relay

from .mcunetv3_wrapper import (
    build_mcu_model,
    configs,
    load_config_from_file,
    update_config_from_args,
    update_config_from_unknown_args,
    QuantizedConv2dDiff,
    QuantizedMbBlockDiff,
    ScaledLinear,
    QuantizedAvgPoolDiff,
)


# Builds the quantized model
def build_quantized_model(net_name="mbv2-w0.35", num_classes=10):

    # Takes network data from config file
    load_config_from_file("/home/andrealavi/tirocinio/tiny-training/configs/transfer.yaml")
    configs["net_config"]["net_name"] = net_name # "mbv2-w0.35"
    configs["net_config"]["mcu_head_type"] = "quantized" # This option is used in build_mcu_model()

    subnet = build_mcu_model()

    subnet = nn.Sequential(*subnet[:5]) # The final network just has the first five layers
    resolution = 128 # Resolution of input images

    # Substitutes the last layer with a quantized 2d conv layer
    last = subnet[-1]

    subnet[-1] = QuantizedConv2dDiff(
        last.in_channels,
        num_classes,
        kernel_size=last.kernel_size,
        stride=last.stride,
        # Quantization parameters
        zero_x=last.zero_x,
        zero_y=last.zero_y,
        effective_scale=last.effective_scale[:num_classes],
    )
    subnet[-1].y_scale = last.y_scale
    subnet[-1].x_scale = last.x_scale

    # The number of weights are reduced to match the number of classes
    # in the classification task
    # The weights tensor dimensions are [out_ch, in_ch, ker_hei, ker_wid]
    subnet[-1].weight.data = last.weight.data[:num_classes, :, :, :]
    return subnet, resolution


build_quantized_mcunet = functools.partial(
    build_quantized_model, net_name="mcunet-5fps"
)
build_quantized_mbv2 = functools.partial(build_quantized_model, net_name="mbv2-w0.35")
build_quantized_proxyless = functools.partial(
    build_quantized_model, net_name="proxyless-w0.3"
)

if __name__ == "__main__":
    net, rs = build_quantized_mbv2(num_classes=10)
    d = torch.randn(1, 3, rs, rs)
    net(d)
