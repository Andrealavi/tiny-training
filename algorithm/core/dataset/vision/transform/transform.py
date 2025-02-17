import torch
import math
from core.utils.config import load_config_from_file, configs
from torchvision import transforms
import random

__all__ = ['ImageTransform']

load_config_from_file("../configs/transfer.yaml")

class ImageTransform(dict):
    def __init__(self):
        super().__init__({
            'train': self.build_train_transform(),
            'val': self.build_val_transform()
        })

    # Returns the transformation for the training images
    def build_train_transform(self):
        if 'vww' in configs.data_provider.root:
            t = transforms.Compose([
                transforms.Resize((configs.data_provider.image_size, configs.data_provider.image_size)),
                transforms.RandomHorizontalFlip(), # Randomly flips the image
                transforms.ToTensor(),
                transforms.Normalize(**self.mean_std),
            ])
        else:
            # Timm is the pytorch library containing image models
            from timm.data import create_transform
            t = create_transform(
                input_size=configs.data_provider.image_size,
                is_training=True,
                color_jitter=configs.data_provider.color_aug, # Modifies color information to augment data
                mean=self.mean_std['mean'],
                std=self.mean_std['std'],
            )

        return t
    
    # Returns the transformation for the val images
    def build_val_transform(self):
        if 'vww' in configs.data_provider.root:
            return transforms.Compose([
                transforms.Resize((configs.data_provider.image_size, configs.data_provider.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(**self.mean_std),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(int(math.ceil(configs.data_provider.image_size / 0.875))),
                transforms.CenterCrop(configs.data_provider.image_size),
                transforms.ToTensor(),
                transforms.Normalize(**self.mean_std)
            ])

    @property
    def mean_std(self):
        if True:  # MCU side model
            # This normalization is used to scale image values from the range [0,1] to [-128,127]
            print('Using MCU transform (leading to range -128, 127)')
            return {'mean': [0.5, 0.5, 0.5], 'std': [1 / 255, 1 / 255, 1 / 255]}
        else:
            # This normalization is used to scale image values to [-1,1]
            return configs.data_provider.get('mean_std',
                                             {'mean': [0.5, 0.5, 0.5],
                                              'std': [0.5, 0.5, 0.5]})
