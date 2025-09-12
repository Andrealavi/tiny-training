# Sparse Update Training CLI (`sput.py`)

## Overview

This command-line tool provides an interface for the MIT research project on sparse neural network updates. It simplifies the process of running and managing training experiments by replacing configuration scripts with a CLI.

The tool supports two primary modes of operation:
1.  **Single Run Mode:** Run a single training experiment using default parameters, with the option to override specific settings via command-line flags.
2.  **Batch Mode:** Automatically run a series of predefined experiments from a JSON configuration file, ideal for systematic testing and hyperparameter sweeps.

All results are logged in CSV or JSON format.

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

    I have decided to use a simple python virtualenv because this will be available by default with python and can be used even on machines where the user does not have root access/privileges, such as a remote servers. To allow for this the code in the `algorithm` folder has been modified as there were some incompatibilities with the latest libraries versions (e.g. numpy).

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


**Important Note**: Please make sure to execute the script from `testing_tool` folder, otherwise there could be an error when building the model. In that case you will need to modify the path that is written in `tiny-training/core/model/model_entry.py` to be the absolute path to the `tiny-training` folder.

After all the trainings have been completed the program will create a CSV/JSON file with the following structure:

```csv
timestamp,model_name,dataset,n_epochs,warmup_epochs,epoch,top1_accuracy,loss,backward_memory_kb,backward_macs,n_bias_update,n_weights_update,manual_weight_idx,weight_update_ratio
1754991677.0803347,mbv2-w0.35,image_folder,50,5,55,73.91445922851562,1.2294960021972656,78.953125kB,5361222,20,2,,
```

```json
[
    {
        "timestamp": 1757614689.7122371,
        "model_name": "mbv2-w0.35",
        "dataset": "image_folder",
        "n_epochs": 4,
        "warmup_epochs": 0,
        "n_bias_update": 1,
        "n_weights_update": 0,
        "manual_weight_idx": null,
        "weight_update_ratio": null,
        "backward_memory_kb": "0.765625kB",
        "backward_macs": "18054",
        "epoch": "2",
        "top1_accuracy": "9.139697074890137",
        "loss": "4.145083427429199"
    },
    {
        "timestamp": 1757614689.7122371,
        "model_name": "mbv2-w0.35",
        "dataset": "image_folder",
        "n_epochs": 4,
        "warmup_epochs": 0,
        "n_bias_update": 1,
        "n_weights_update": 0,
        "manual_weight_idx": null,
        "weight_update_ratio": null,
        "backward_memory_kb": "0.765625kB",
        "backward_macs": "18054",
        "epoch": "4",
        "top1_accuracy": "12.148316383361816",
        "loss": "3.9741737842559814"
    }
]
```

In order to allow for a higher flexibility, each row of the resulting CSV file and each object in the list of the JSON file represent a snapshot at a specific epoch of a certain configuration. Each configuration can be identified by the unique timestamp that is shared across all the snapshot. This way it is possible to have in the same batch of tests different number of epochs and evaluation checkpoints.


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

**Important Note**: Please make sure to modify the root folder for the datasets within the batch configuration files, as it set up to `~/dataset`.
