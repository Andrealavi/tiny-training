from .vision import *
from ..utils.config import configs
from .vision.transform import *
import torchvision

__all__ = ['build_dataset']

# Returns a dict containing train and val folders of the selected dataset
def build_dataset():
    if configs.data_provider.dataset == 'image_folder':
        dataset = ImageFolder(
            root=f"{configs.data_provider.root}/flowers102", # TODO
            transforms=ImageTransform(),
        )
    elif configs.data_provider.dataset == 'new_gestures':
        dataset = ImageFolder(
            root=f"{configs.data_provider.root}/new_gestures",
            transforms=ImageTransform(),
        )
    elif configs.data_provider.dataset == 'pets':
        dataset = ImageFolder(
            root=f"{configs.data_provider.root}/dataset/pets",
            transforms=ImageTransform(),
        )
    elif configs.data_provider.dataset == "cub":
        dataset = ImageFolder(
            root=f"{configs.data_provider.root}/cub",
            transforms=ImageTransform(),
        )
    elif configs.data_provider.dataset == 'imagenet':
        # I don't find any imagenet function
        dataset = ImageNet(root=configs.data_provider.root,
                       transforms=ImageTransform(), )
    elif configs.data_provider.dataset == 'cifar10':
        # Downloads the dataset and saves it to root
        # before returing the dataset dict
        dataset = {
            'train': torchvision.datasets.CIFAR10(configs.data_provider.root, train=True,
                                                  transform=ImageTransform()['train'], download=True),
            'val': torchvision.datasets.CIFAR10(configs.data_provider.root, train=False,
                                                transform=ImageTransform()['val'], download=True),
        }
    elif configs.data_provider.dataset == 'cifar100':
        # Downloads the dataset and saves it to root
        # before returing the dataset dict
        dataset = {
            'train': torchvision.datasets.CIFAR100(configs.data_provider.root, train=True,
                                                   transform=ImageTransform()['train'], download=True),
            'val': torchvision.datasets.CIFAR100(configs.data_provider.root, train=False,
                                                 transform=ImageTransform()['val'], download=True),
        }
    elif configs.data_provider.dataset == 'imagehog':
        # I don't find any imagehog function
        dataset = ImageHog(
            root=configs.data_provider.root,
            transforms=ImageTransform(),
        )
    else:
        raise NotImplementedError(configs.data_provider.dataset)

    return dataset
