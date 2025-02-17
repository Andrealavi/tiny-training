#!/usr/bin/env python3
import subprocess
import argparse
from typing import Optional


def run_single_layer_tests(start = 0, end = 50) -> None:
    """
    Execute the training script 51 times, testing each weight layer individually
    """
    base_command = """python train_cls.py configs/transfer.yaml \
        --run_dir runs/flowers/mcunet-5fps/sparse_124kb/sgd_qas_nomom \
        --net_name mbv2-w0.35 \
        --bs256_lr 0.1 \
        --optimizer_name sgd_scale_nomom \
        --enable_backward_config 1 \
        --n_bias_update 51"""

    for layer_idx in range(start, end, 1):
        print(f"\nTest {layer_idx + 1}/51")
        print(f"Testing weight layer: {layer_idx}")
        
        # Construct the full command with single layer index
        full_command = f"{base_command} --manual_weight_idx {layer_idx}"
        
        try:
            # Execute the command
            subprocess.run(full_command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during test of layer {layer_idx}: {e}")
            continue
        except KeyboardInterrupt:
            print("\nProcess interrupted by user")
            break

def main():
    parser = argparse.ArgumentParser(
        description='Run systematic tests of each weight layer individually'
    )
    parser.add_argument('--start_from', type=int, default=0,
                      help='Start testing from this layer index (default: 0)')
    parser.add_argument('--end_at', type=int, default=50,
                      help='End testing at this layer index (default: 50)')
    
    args = parser.parse_args()
    
    if not (0 <= args.start_from <= 50):
        raise ValueError("start_from must be between 0 and 50")
    if not (0 <= args.end_at <= 50):
        raise ValueError("end_at must be between 0 and 50")
    if args.start_from > args.end_at:
        raise ValueError("start_from cannot be greater than end_at")
    
    run_single_layer_tests(args.start_from, args.end_at)

if __name__ == "__main__":
    main()
