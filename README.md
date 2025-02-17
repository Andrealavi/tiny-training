# Neural Network Training Automation Framework

This repository is a fork of the [Tiny-Training](https://github.com/mit-han-lab/tiny-training) project by MIT Han Lab, extended with automated testing capabilities for systematic neural network training experiments.

## Overview

This extension adds automated testing functionality to the original Tiny-Training framework, allowing for systematic evaluation of different training configurations. The automation scripts enable running multiple training iterations with varying parameters, specifically focusing on bias updates, layer-wise testing, and random parameter combinations testing. All automation scripts are located in the `algorithm` subfolder of the project, maintaining a clean separation between the original codebase and the testing extensions.

## Prerequisites

- Python 3.8
- Conda package manager
- CUDA-capable GPU (recommended)

## Environment Setup

The project requires specific package dependencies that can be installed using Conda. Follow these steps to set up your environment:

1. Clone the repository and switch to the test branch:
```bash
git clone [your-repository-url]
cd [repository-name]
git checkout tests
```

2. Create a new Conda environment:
```bash
conda create -n tiny-training-test python=3.8
conda activate tiny-training-test
```

3. Install dependencies using the package-list.txt file:
```bash
conda create --name tiny-training-test --file package-list.txt
```

**Note**: The package list was generated on a Linux-64 platform. For macOS or Windows users, some packages might need to be installed with different versions or through alternative channels. You may need to manually resolve platform-specific dependencies.

## Configuration

The framework uses YAML configuration files located in the `configs` directory to manage training parameters and dataset settings. There are three main configuration files:

### Default Configuration (`configs/default.yaml`)
This file contains the base configuration settings. The most important settings to modify are:

```yaml
data_provider:
  root: /path/to/your/dataset/  # Set this to your dataset directory path
  dataset: image_folder         # Dataset type to use
  base_batch_size: 64          # Batch size for training
  image_size: 128              # Input image size
  num_classes: null            # Number of classes (set in specific configs)

run_config:
  n_epochs: 3                  # Number of training epochs
  warmup_epochs: 3             # Number of warmup epochs
  base_lr: 0.025              # Base learning rate
```

### Transfer Learning Configuration (`configs/transfer.yaml`)
This configuration extends the default settings for transfer learning scenarios:

```yaml
data_provider:
  root: /path/to/your/dataset/  # Update this path
  num_classes: 102              # Set this to your dataset's class count

run_config:
  bs256_lr: 0.01               # Learning rate for batch size 256
  optimizer_name: sgd           # Optimizer selection

net_config:
  net_name: mbv2-w0.35         # Neural network architecture
```

## Dataset Setup

### Dataset Directory Structure
Create a `dataset` directory in your project root and organize it as follows:

```
dataset/
├── flowers102/
│   ├── train/
│   │   ├── class1/
│   │   ├── class2/
│   │   └── ...
│   └── val/
│       ├── class1/
│       ├── class2/
│       └── ...
├── gestures/
└── pets/
```

### Adding Custom Datasets

To add a custom dataset, you'll need to:

1. Create your dataset directory following the structure above
2. Modify `algorithm/core/dataset/dataset_entry.py` to include your dataset:

```python
def build_dataset():
    if configs.data_provider.dataset == 'your_dataset':
        dataset = ImageFolder(
            root="/path/to/your/dataset",
            transforms=ImageTransform(),
        )
```

3. Update your configuration YAML file with the appropriate settings:

```yaml
data_provider:
  dataset: your_dataset
  root: /path/to/your/dataset
  num_classes: your_number_of_classes
```

The framework currently supports several dataset types:
- `image_folder`: Generic image folder dataset
- `cifar10`: CIFAR-10 dataset
- `cifar100`: CIFAR-100 dataset
- `gestures`: Custom gestures dataset
- `pets`: Oxford-IIIT Pet Dataset
- Custom datasets (by following the steps above)

### Dataset Transforms

The framework applies standard image transforms defined in the `ImageTransform` class. These include:
- Training: Random resizing, cropping, and color augmentation
- Validation: Center crop and normalization

## Testing Framework

The testing framework consists of three main scripts, all located in the `algorithm` directory:

### 1. Bias Update Testing (`algorithm/automate_each_bias_tests.py`)
This script systematically tests different bias update values from 1 to 51, allowing you to understand how different bias update frequencies affect model performance.

Usage:
```bash
python algorithm/automate_each_bias_tests.py [--start_from START] [--end_at END]
```

Parameters:
- `--start_from`: Starting bias value (default: 1)
- `--end_at`: Ending bias value (default: 51)

### 2. Layer Testing (`algorithm/automate_each_layer_tests.py`)
This script enables individual testing of each weight layer, helping you understand the impact of updates at different network depths.

Usage:
```bash
python algorithm/automate_each_layer_tests.py [--start_from START] [--end_at END]
```

Parameters:
- `--start_from`: Starting layer index (default: 0)
- `--end_at`: Ending layer index (default: 50)

### 3. Random Training (`algorithm/automate_training.py`)
This script executes multiple training runs with randomized parameters, allowing for broader exploration of the parameter space. It includes intelligent parameter selection with weighted probabilities for certain preferred values.

Usage:
```bash
python algorithm/automate_training.py --num_runs RUNS --num_indices INDICES [--seed SEED]
```

Parameters:
- `--num_runs`: Number of training runs to execute
- `--num_indices`: Number of weight indices to generate for each run
- `--seed`: Random seed for reproducibility (optional)

## Results

Test results are automatically saved in the `./tests/` directory with filenames following the pattern:
```
test_e_{epochs}_d_{dataset}b_{bias_update}_w_{weight_idx}_r_{update_ratio}.txt
```

Each result file contains comprehensive information about the training run:
- Dataset information
- Warmup and total epoch counts
- Training parameters including:
  - Bias update values
  - Weight indices used
  - Update ratios
- Validation metrics
- Memory usage statistics
- Computational complexity metrics (MACs)

## Attribution

This work builds upon the Tiny-Training framework developed by MIT Han Lab. Please refer to the original repository for the base implementation:
[https://github.com/mit-han-lab/tiny-training](https://github.com/mit-han-lab/tiny-training)
