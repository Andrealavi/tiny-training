import copy
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

__all__ = ['get_module_device', 'remove_bn', 'trainable_param_num', 'replace_layers']

# Returns the device which is running the code (CPU or GPU)
def get_module_device(module: nn.Module) -> torch.device:
    return module.parameters().__next__().device

# Removes the batch norm layer
def remove_bn(model: nn.Module):
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            m.weight = m.bias = None
            m.forward = lambda x: x

# Returns the number of parameters that are trainable
def trainable_param_num(network: nn.Module, unit=1e6) -> float:
    # numel returns the number of elements in a tensor
    return sum(p.numel() for p in network.parameters() if p.requires_grad) / unit # division by unit is used to convert the count

# replaces one of the network layers
def replace_layers(model, old_class, new_class):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_layers(module, old_class, new_class)

        if isinstance(module, old_class):
            print('replace...')
            setattr(model, n, new_class())
