import os
import sys

# Append the path to the folder containing the package
package_path = os.path.abspath("/home/andrealavi/tirocinio/tiny-training/algorithm")  # Update this to the actual path
sys.path.append(package_path)

# Imports function to build the microcontroller model
from core.model import build_mcu_model

# Imports utils for managing network configs
from core.utils.config import (
    configs,
    load_config_from_file,
    update_config_from_args,
    update_config_from_unknown_args,
)

# Imports quantized layers for backprop
from quantize.quantized_ops_diff import (
    QuantizedConv2dDiff,
    QuantizedMbBlockDiff,
    ScaledLinear,
    QuantizedAvgPoolDiff,
)
