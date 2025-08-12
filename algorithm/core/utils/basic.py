import torch
from typing import Union, List, Any
import torch.distributed
import numpy as np
from core.utils import dist

__all__ = ['ddp_reduce_tensor', 'DistributedMetric', 'accuracy', 'AverageMeter']

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def list_sum(x: List) -> Any:
    r"""
    return the sum of a list of objects (can be int, float, torch.Tensor, np.ndarray, etc)
    can be used for adding losses
    """
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])


def list_mean(x: List) -> Any:
    r"""
    return the mean of a list of objects (can be int, float, torch.Tensor, np.ndarray, etc)
    """
    return list_sum(x) / len(x)


def ddp_reduce_tensor(tensor: torch.Tensor, reduce='mean') -> Union[torch.Tensor, List[torch.Tensor]]:
    if dist.size() == 1: # dist.size() returns the number of processes involved in computation
        return tensor


    tensor_list = [
        torch.empty_like(tensor) for _ in range(dist.size())
    ]

    # Gathers tensors from each process in the group and place them in tensor_list
    torch.distributed.all_gather(tensor_list, tensor.contiguous(), async_op=False)

    if reduce == 'mean':
        return list_mean(tensor_list)
    elif reduce == 'sum':
        return list_sum(tensor_list)
    elif reduce == 'cat':
        return torch.cat(tensor_list, dim=0)
    else:
        return tensor_list


# Maintains track of the average across all processes
class DistributedMetric(object):
    r"""
    average metrics for distributed training.
    """

    def __init__(self, name: str, backend='ddp'):
        self.name = name
        self.sum = 0
        self.count = 0
        self.backend = backend

    def update(self, val: Union[torch.Tensor, int, float], delta_n=1):
        # delta_n represents the number of samples
        # This product makes val reflect the total contribution
        val *= delta_n
        if type(val) in [int, float]:
            val = torch.Tensor(1).fill_(val).to(device)
        if self.backend == 'ddp':
            self.count += ddp_reduce_tensor(torch.Tensor(1).fill_(delta_n).to(device), reduce='sum')
            self.sum += ddp_reduce_tensor(val.detach(), reduce='sum')
        else:
            raise NotImplementedError

    @property
    def avg(self):
        if self.count == 0:
            return torch.Tensor(1).fill_(-1)
        else:
            return self.sum / self.count


class AverageMeter(object):
    r"""
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: Union[torch.Tensor, np.ndarray, float, int], n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Computes accuracy using top-k prediction and evaluation
# top-k refers to the k classes with the highest score
# top-k evalutaion is a metric of accuracy based on
# how many times the correct classification is within the
# top-k predictions
def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[torch.Tensor]:
    r"""
    Computes the precision@k for the specified values of k
    """

    maxk = min(max(topk), output.shape[1])
    batch_size = target.shape[0]

    # Gets predictions and reshapes them
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    # Checks which predictions were correct
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    # Checks the accuracy of predictions
    res = []
    for k in topk:
        if k <= output.shape[1]:
            # Computes accuracy of the top-k predictions
            # takes the first k predictions which are the top ones
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        else:
            res.append(torch.zeros(1).to(device) - 1.)
    return res
