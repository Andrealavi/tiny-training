# Sparse Update Training CLI (`sput.py`)

## Overview

This command-line tool provides a polished and user-friendly interface for the MIT research project on sparse neural network updates. It simplifies the process of running and managing training experiments by replacing complex configuration scripts with a straightforward CLI.

The tool supports two primary modes of operation:
1.  **Single Run Mode:** Quickly run a single training experiment using default parameters, with the option to override specific settings via command-line flags.
2.  **Batch Mode:** Automatically run a series of predefined experiments from a JSON configuration file, ideal for systematic testing and hyperparameter sweeps.

All results are logged in a structured, analysis-friendly CSV format.

## Installation

Follow these steps to set up your environment and install the necessary dependencies.

### Setup Instructions

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Andrealavi/tiny-training
    cd tiny-training

    git checkout remotes/origin/dev
    ```

2.  **Create and Activate a Virtual Environment**
    It is highly recommended to use a virtual environment to manage project dependencies.

    ```bash
    # Create the virtual environment
    python -m venv .venv

    # Activate it (on macOS/Linux)
    source .venv/bin/activate

    # On Windows, use:
    # .venv\Scripts\activate
    ```

3.  **Install Dependencies**
    The required Python packages are listed in `requirements.txt`. Install them using `pip`.

    ```bash
    pip install -r requirements.txt
    ```
    This will install `typer`, `pydantic`, `torch`, and other necessary libraries.

## Usage

The main entry point for the tool is `sput.py`. All commands are run from your terminal.

After all the trainings have been completed the program will create a CSV file with the following structure:

```csv
timestamp,model_name,dataset,n_epochs,warmup_epochs,epoch,top1_accuracy,loss,backward_memory_kb,backward_macs,n_bias_update,n_weights_update,manual_weight_idx,weight_update_ratio
1754991677.0803347,mbv2-w0.35,image_folder,50,5,55,73.91445922851562,1.2294960021972656,78.953125kB,5361222,20,2,,
```

In order to allow for a higher flexibility, each row of the resulting CSV file represent a snapshot at a specific epoch of a certain configuration. Each configuration can be identified by the unique timestamp that is shared across all the snapshot. This way it is possible to have in the same batch of tests different number of epochs and evaluation checkpoints.

### Single Run Mode

This mode is used for running a single experiment. You can run it with default settings or override any parameter using command-line flags.

**Basic Usage (with defaults):**
This will run one training session using the default parameters defined in the configuration schema.
```bash
python sput.py
```

### Batch Mode

This mode is used for running multiple experiments one after the other to reduce all the hassle of performing different tests manually.
When using batch mode you pass a json file with multiple configurations (examples are in `batches_configs` folder) and the program will run each
of them sequentially.

**Batch Usage:**
This will run one training session using the default parameters defined in the configuration schema.
```bash
python sput.py --batch config.json
```

In `batches_configs` are provided different examples. Specifically, there are JSON files for executing tests for each weight and for an incremental number of biases for different datasets.

## Datasets

Please refer to the `README.md` file that is the in the `algorithm` directory to understand how to install different datasets.
If you want to add new datasets you need to modify the file `algorithm/core/dataset/dataset_entry.py` and to add a new `if` statement for your specific dataset.
