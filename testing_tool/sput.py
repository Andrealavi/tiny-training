# Imports needed to make the tool work.
import os
import sys
import json
import csv
from pydantic import ValidationError # Pydantic has been used for data validation
import typer # Typer has been used to set up a fully-fledged cli tool
from typing import Optional, List
from time import time
from config import TrainingConfig

# The following line is used to add the algorithm folder within the PATH
# environment variable. This way it is possible to import algorithm folder
# modules and packages.
sys.path.insert(0, "../algorithm")

# Import needed to make the training work.
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed

from core.utils import dist
from core.model import build_mcu_model
from core.utils.config import configs, update_config_from_args
from core.utils.logging import logger
from core.dataset import build_dataset
from core.optimizer import build_optimizer
from core.trainer.cls_trainer import ClassificationTrainer
from core.builder.lr_scheduler import build_lr_scheduler

from core.utils.partial_backward import parsed_backward_config, prepare_model_for_backward_config, \
    get_all_conv_ops, nelem_saved_for_backward, compute_macs

# Python set for containing all allowed output formats.
# This way it will be simple to add a new output format for the data.
ALLOWED_FORMATS = {"csv", "json"}

# Here we set the device variable depending on the platform.
# Specifically, we consider only CUDA (NVIDIA) and mps (Apple Silicon).
# If neither of the two is available, we just use the CPU.
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def perform_training(training_config: TrainingConfig):
    """
    Performs a complete training using a specific training configuration.

    This function primarily uses the code from the train_cls.py file within the
    algorithm folder. It handles the entire training pipeline including model
    creation, data loading, optimizer setup, and training execution with
    optional sparse update configurations.

    Args:
        training_config (TrainingConfig): A Pydantic-validated training configuration
            object containing all necessary parameters for the training process.

    Returns:
        tuple: If evaluating, returns validation info dictionary. Otherwise returns
            a tuple containing (training_validations, saved_for_backward, backward_macs)
            where:
            - training_validations: Dictionary of validation results per epoch
            - saved_for_backward: Number of elements saved for backward pass
            - backward_macs: String representation of MACs computation for backward pass

    Note:
        The function converts the Pydantic TrainingConfig to a dictionary since the
        original MIT code uses an easydict for managing configurations. Weight indices
        and update ratios are converted from arrays to string format using "-" as separator.
    """

    training_config_dict = training_config.model_dump()

    if training_config_dict["backward_config"]["manual_weight_idx"]:
        training_config_dict["backward_config"]["manual_weight_idx"] = "-".join(map(str, training_config_dict["backward_config"]["manual_weight_idx"]))
    if training_config_dict["backward_config"]["weight_update_ratio"]:
        training_config_dict["backward_config"]["weight_update_ratio"] = "-".join(map(str, training_config_dict["backward_config"]["weight_update_ratio"]))

    # This line of code updates the configuration easydict using the dictionary
    # obtained from the training_config.
    update_config_from_args(training_config_dict)

    dist.init() # Initializes a process
    torch.backends.cudnn.benchmark = True

    assert configs.run_dir is not None
    os.makedirs(configs.run_dir, exist_ok=True)

    # Initializes logger and log some messages
    logger.init()  # dump exp config
    logger.info(' '.join([sys.executable] + sys.argv))
    logger.info(f'Experiment started: "{configs.run_dir}".')

    # set random seed
    torch.manual_seed(configs.manual_seed)

    if device == torch.device("cuda"):
        torch.cuda.set_device(dist.local_rank())
        torch.cuda.manual_seed_all(configs.manual_seed)

    dataset = build_dataset()
    data_loader = dict()
    for split in dataset:
        # Sampler is used to take random samples from the dataset
        sampler = torch.utils.data.DistributedSampler(
            dataset[split],
            num_replicas=dist.size(),
            rank=dist.rank(),
            seed=configs.manual_seed,
            shuffle=(split == 'train')) # Shuffles only if split is true

        # Loads data based on sampler
        data_loader[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=configs.data_provider.base_batch_size,
            sampler=sampler,
            num_workers=configs.data_provider.n_worker,
            pin_memory=True,
            drop_last=(split == 'train'),
        )

    model = build_mcu_model().to(device)

    # Here we check if it is possible to parallelize training. If so we do that.
    if dist.size() > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.local_rank()])  # , find_unused_parameters=True)

    criterion = torch.nn.CrossEntropyLoss() # Loss function criterion
    optimizer = build_optimizer(model)
    lr_scheduler = build_lr_scheduler(optimizer, len(data_loader['train']))

    trainer = ClassificationTrainer(model, data_loader, criterion, optimizer, lr_scheduler)

    if configs.resume:
        trainer.resume()  # trying to resume

    saved_for_backward = 0

    # Looks for sparse update configurations
    if configs.backward_config.enable_backward_config:
        # Here we parse the sparse update configuration and prepare the model.
        configs.backward_config = parsed_backward_config(configs.backward_config, model)

        # This function simply create bitmasks for each convolutional layer in
        # case we want to perform sparse channel update.
        prepare_model_for_backward_config(model, configs.backward_config)

        logger.info(f'Getting backward config: {configs.backward_config} \n'
                    f'Total convs {len(get_all_conv_ops(model))}')

        # Here we get a sample image to measure the number of elements that need
        # to be saved for backward pass. This value will be then used to compute
        # the amount of memory needed to perform backward.
        images, _ = next(iter(data_loader['train']))
        saved_for_backward = nelem_saved_for_backward(model, images.to(device), configs.backward_config)

    if configs.evaluate:
        val_info_dict = trainer.validate()
        return val_info_dict  # for ray tune
    else:
        val_info_dict = trainer.run_training()

        backward_macs = str(compute_macs(
            model,
            configs.backward_config,
            torch.rand(
                (configs.data_provider.base_batch_size,
                 3,
                 configs.data_provider.image_size,
                 configs.data_provider.image_size
                 )
            ).to(device))
        )

        return (trainer.training_validations, saved_for_backward, backward_macs)  # for ray tune


def write_data(filename: str, out_format: str, data: List[dict], allowed_formats: set[str]) -> None:
    """
    Writes data into the selected output format (CSV or JSON).

    All data will be placed within a 'tests' folder that will be created
    if it does not exist. The function supports both CSV and JSON output formats.

    Args:
        filename (str): Name of the output file (without path, will be saved in ./tests/)
        out_format (str): Format of the output file, either "csv" or "json"
        data (List[dict]): List of dictionaries containing the data to be written

    Raises:
        ValueError: If out_format is not "csv" or "json"

    Note:
        For CSV format, the function uses the keys from the first dictionary as headers.
        For JSON format, the data is written with 4-space indentation for readability.
    """

    if out_format not in allowed_formats:
        raise ValueError(f"Unsupported format: {out_format}. Use: {allowed_formats}")

    os.makedirs("./tests", exist_ok=True)

    try:
        with open(f"./tests/{filename}", "w") as f:
            if out_format == "csv":
                csv_header = list(data[0].keys())
                writer = csv.DictWriter(f, fieldnames=csv_header)

                writer.writeheader()
                writer.writerows(data)
            elif out_format == "json":
                json_data = json.dumps(data, indent=4)
                f.write(json_data)
    except IOError as e:
        raise IOError(f"Failed to write to {filename}: {e}")


def load_configs_from_file(filename: str) -> List[TrainingConfig]:
    """
    Loads training configurations from a JSON batch file.

    A batch file is simply a list of different training configurations that will
    be processed sequentially by the program. Each configuration in the file
    is validated using Pydantic before being returned.

    Args:
        filename (str): Path to the JSON file containing the batch configurations

    Returns:
        List[TrainingConfig]: List of validated TrainingConfig objects loaded from the file

    Raises:
        FileNotFoundError: If the specified filename does not exist
        json.JSONDecodeError: If the JSON file cannot be decoded
        ValidationError: If any configuration in the file fails Pydantic validation
        Exception: Re-raises the last caught exception after printing error messages

    Note:
        The function prints user-friendly error messages for common failure cases
        before re-raising the exception.
    """

    try:
        with open(filename, "r") as file:
            batches_data = json.load(file)
            validated_training_configs = [
                TrainingConfig.model_validate(config) for config in batches_data
            ]

            return validated_training_configs
    except FileNotFoundError:
        logger.error(f"{filename} does not exists.")
    except json.JSONDecodeError:
        logger.error(f"It wasn't possible to decode JSON file {filename}.")
    except ValidationError as e:
        logger.error("Error: The configuration file is invalid.")
        print(e) # Pydantic validation error output

    raise


def main(
    # Top-level TrainingConfig attributes
    run_dir: Optional[str] = typer.Option("./runs", help="Run directory path"),

    # DataProviderConfig attributes
    dataset: str = typer.Option("image_folder", help="Dataset type"),
    root: str = typer.Option("/home/alavino/dataset/", help="Dataset root path"),
    image_size: int = typer.Option(128, help="Image size"),
    num_classes: int = typer.Option(102, help="Number of classes"),

    # RunConfig attributes
    n_epochs: int = typer.Option(50, help="Number of epochs"),
    base_lr: float = typer.Option(0.025, help="Base learning rate"),
    warmup_epochs: int = typer.Option(5, help="Warmup epochs"),
    eval_per_epochs: int = typer.Option(10, help="Evaluation frequency in epochs"),

    # NetConfig attributes
    net_name: str = typer.Option("mbv2-w0.35", help="Network name"),

    # BackwardConfig attributes
    enable_backward_config: bool = typer.Option(False, help="Enable backward configuration"),
    n_bias_update: Optional[int] = typer.Option(None, help="Number of bias updates"),
    n_weight_update: Optional[int] = typer.Option(None, help="Number of weight updates"),
    weight_update_ratio: Optional[float] = typer.Option(None, help="Weight update ratio"),
    manual_weight_idx: Optional[int] = typer.Option(None, help="Manual weight index"),
    manual_bias_idx: Optional[int] = typer.Option(None, help="Manual bias index"),
    quantize_gradient: bool = typer.Option(False, help="Quantize gradient"),

    batch: str = typer.Option("", help="File containing batches of training configs"),
    out_format: str = typer.Option("csv", help="Format of the output file (csv or json)")
):
    """
    CLI tool for neural network training configuration with sparse update support.

    This tool provides a command-line interface for training neural
    networks with configurable parameters. It supports both single training runs
    with command-line parameters and batch processing from JSON configuration files.

    The tool processes training configurations sequentially and outputs results
    including validation accuracy, loss, memory usage, and computational metrics
    for each training run. Results are saved in the specified format (CSV or JSON)
    with comprehensive information about the training process.

    Args:
        run_dir: Directory path where training runs will be stored
        dataset: Type of dataset to use for training
        root: Root path to the dataset directory
        image_size: Input image size for the model
        num_classes: Number of output classes for classification
        n_epochs: Total number of training epochs
        base_lr: Base learning rate for the optimizer
        warmup_epochs: Number of epochs for learning rate warmup
        eval_per_epochs: Frequency of validation evaluation in epochs
        net_name: Name/identifier of the neural network architecture
        enable_backward_config: Whether to enable sparse update configurations
        n_bias_update: Number of bias parameters to update (for sparse training)
        n_weight_update: Number of weight parameters to update (for sparse training)
        weight_update_ratio: Ratio of weights to update during sparse training
        manual_weight_idx: Manual specification of weight indices for sparse updates
        manual_bias_idx: Manual specification of bias indices for sparse updates
        quantize_gradient: Whether to apply gradient quantization
        batch: Path to JSON file containing multiple training configurations
        out_format: Output format for results ("csv" or "json")

    Note:
        If a batch file is provided, individual CLI parameters are ignored and
        the tool processes all configurations from the batch file sequentially.

        The output includes base information (model name, dataset, epochs, etc.)
        plus checkpoint information about accuracy results at different epoch
        checkpoints. Checkpoint evaluation frequency depends on the eval_per_epochs
        parameter in the training configuration.

        Memory usage is calculated as: (memory_elements / 1024 / 8) KB
        where memory_elements represents the number of elements saved for backward pass.
    """

    print("Training started with the provided configuration...")

    training_configs = []
    data = []
    if batch != "":
        training_configs = load_configs_from_file(batch)
    else:
        config_values = {
            "run_dir": run_dir,
            "data_provider": {
                "dataset": dataset,
                "root": root,
                "image_size": image_size,
                "num_classes": num_classes,
            },
            "run_config": {
                "n_epochs": n_epochs,
                "base_lr": base_lr,
                "warmup_epochs": warmup_epochs,
                "eval_per_epochs": eval_per_epochs,
            },
            "net_config": {
                "net_name": net_name,
            },
            "backward_config": {
                "enable_backward_config": enable_backward_config,
                "n_bias_update": n_bias_update,
                "n_weight_update": n_weight_update,
                "weight_update_ratio": weight_update_ratio,
                "manual_weight_idx": manual_weight_idx,
                "manual_bias_idx": manual_bias_idx,
                "quantize_gradient": quantize_gradient
            }
        }

        training_config = TrainingConfig(**config_values)
        training_configs.append(training_config)

    count = 1
    total_trainings = len(training_configs)
    for config in training_configs:
        print(f"Performing {count}/{total_trainings} training\n\n")
        count += 1
        val_dicts, memory, macs = perform_training(config)

        # These are the base information that will be printed in the csv or JSON.
        # Other that these, we have checkpoints information about accuracy results
        # at different epochs checkpoint. How frequent checkpoint evaluations
        # are depend on the particular training configuration.
        base_info = {
            "timestamp": time(),
            "model_name": config.net_config.net_name,
            "dataset": config.data_provider.dataset,
            "n_epochs": config.run_config.n_epochs,
            "warmup_epochs": config.run_config.warmup_epochs,
            "n_bias_update": config.backward_config.n_bias_update,
            "n_weights_update": config.backward_config.n_weight_update,
            "manual_weight_idx": config.backward_config.manual_weight_idx,
            "weight_update_ratio": config.backward_config.weight_update_ratio,
            "backward_memory_kb": f"{int(memory) / 1024 / 8}kB",
            "backward_macs": str(macs)
        }

        for epoch, val in val_dicts.items():
            row_dict = base_info.copy()

            row_dict["epoch"] = str(epoch)
            row_dict["top1_accuracy"] = str(val["val/top1"])
            row_dict["loss"] = str(val["val/loss"])

            data.append(row_dict)

    filename = f"training_{time()}.{out_format}"
    write_data(filename, out_format, data, ALLOWED_FORMATS)

if __name__ == "__main__":
    typer.run(main)
