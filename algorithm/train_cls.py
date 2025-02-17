# an image classification trainer
import os
import sys
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed

from core.utils import dist
from core.model import build_mcu_model
from core.utils.config import configs, load_config_from_file, update_config_from_args, update_config_from_unknown_args
from core.utils.logging import logger
from core.dataset import build_dataset
from core.optimizer import build_optimizer
from core.trainer.cls_trainer import ClassificationTrainer
from core.builder.lr_scheduler import build_lr_scheduler

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('config', metavar='FILE', help='config file')
parser.add_argument('--run_dir', type=str, metavar='DIR', help='run directory')
parser.add_argument('--evaluate', action='store_true')

# Builds configuration dictionary
def build_config():  # separate this config requirement so that we can call main() in ray tune
    # support extra args here without setting in args
    args, unknown = parser.parse_known_args()

    load_config_from_file(args.config)
    update_config_from_args(args)
    update_config_from_unknown_args(unknown)


def main():
    dist.init() # Initializes a process
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    assert configs.run_dir is not None
    os.makedirs(configs.run_dir, exist_ok=True)

    # Initializes logger and log some messages
    logger.init()  # dump exp config
    logger.info(' '.join([sys.executable] + sys.argv))
    logger.info(f'Experiment started: "{configs.run_dir}".')

    # set random seed
    torch.manual_seed(configs.manual_seed)
    torch.cuda.manual_seed_all(configs.manual_seed)

    # create dataset
    dataset = build_dataset()
    data_loader = dict()
    for split in dataset:
        sampler = torch.utils.data.DistributedSampler( # Sampler is used to take random samples from the dataset
            dataset[split],
            num_replicas=dist.size(),
            rank=dist.rank(),
            seed=configs.manual_seed,
            shuffle=(split == 'train')) # Shuffles only if split is true

        data_loader[split] = torch.utils.data.DataLoader( # Loads data based on sampler
            dataset[split],
            batch_size=configs.data_provider.base_batch_size,
            sampler=sampler,
            num_workers=configs.data_provider.n_worker,
            pin_memory=True,
            drop_last=(split == 'train'),
        )

    # create model
    model = build_mcu_model().cuda()

    if dist.size() > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.local_rank()])  # , find_unused_parameters=True)

    criterion = torch.nn.CrossEntropyLoss() # Loss function criterion
    optimizer = build_optimizer(model)
    lr_scheduler = build_lr_scheduler(optimizer, len(data_loader['train']))

    trainer = ClassificationTrainer(model, data_loader, criterion, optimizer, lr_scheduler)

    # kick start training
    if configs.resume:
        trainer.resume()  # trying to resume

    saved_for_backward = 0

    # Looks for sparce update configurations
    if configs.backward_config.enable_backward_config:
        from core.utils.partial_backward import parsed_backward_config, prepare_model_for_backward_config, \
            get_all_conv_ops, nelem_saved_for_backward, compute_macs
        configs.backward_config = parsed_backward_config(configs.backward_config, model)
        prepare_model_for_backward_config(model, configs.backward_config)
        logger.info(f'Getting backward config: {configs.backward_config} \n'
                    f'Total convs {len(get_all_conv_ops(model))}')

        for _, (images, labels) in enumerate(data_loader['train']):
            saved_for_backward = nelem_saved_for_backward(model, images.cuda(), configs.backward_config)

            break


    if configs.evaluate:
        val_info_dict = trainer.validate()
        print(val_info_dict)
        return val_info_dict  # for ray tune
    else:
        val_info_dict = trainer.run_training()

        from core.utils.partial_backward import compute_macs

        with open(f"./tests/test_e_{str(configs.run_config.warmup_epochs)}_d_{str(configs.data_provider.dataset)}b_{str(configs.backward_config.n_bias_update)}_w_{str(configs.backward_config.manual_weight_idx)}_r_{str(configs.backward_config.weight_update_ratio)}.txt", "w") as f:
            f.write(str(configs.data_provider.dataset))
            f.write("\n")
            f.write("warmup epochs: ")
            f.write(str(configs.run_config.warmup_epochs))
            f.write("\n")
            f.write("epochs: ")
            f.write(str(configs.run_config.n_epochs))
            f.write("\n")
            f.write(str(configs.backward_config.manual_weight_idx))
            f.write("\n")
            f.write(str(configs.backward_config.n_bias_update))
            f.write("\n")
            f.write(str(configs.backward_config.weight_update_ratio))
            f.write("\n")
            f.write(str(val_info_dict))
            f.write("\n")
            f.write('total: {:.0f}kB'.format(saved_for_backward / 1024 / 8))
            f.write("\n")
            f.write(str(compute_macs(
                model,
                configs.backward_config,
                torch.rand(
                    (configs.data_provider.base_batch_size,
                     3,
                     configs.data_provider.image_size,
                     configs.data_provider.image_size
                     )
                ).cuda()
            )))

        return (val_info_dict, saved_for_backward)  # for ray tune


if __name__ == '__main__':
    build_config()
    main()
