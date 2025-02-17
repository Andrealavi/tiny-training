#!/usr/bin/env python3
import subprocess
import argparse
from typing import Optional

def run_bias_tests() -> None:
    """
    Execute the training script 51 times, testing each bias value from 1 to 51
    """
    base_command = """python train_cls.py configs/transfer.yaml \
        --run_dir runs/flowers/mcunet-5fps/sparse_124kb/sgd_qas_nomom \
        --net_name mbv2-w0.35 \
        --bs256_lr 0.1 \
        --optimizer_name sgd_scale_nomom \
        --enable_backward_config 1 \
        --n_weight_update 0"""

    for bias_value in range(1, 52):  # Testing from 1 to 51
        print(f"\nTest {bias_value}/51")
        print(f"Testing with n_bias_update: {bias_value}")
        
        # Construct the full command with bias value
        full_command = f"{base_command} --n_bias_update {bias_value}"
        
        try:
            # Execute the command
            subprocess.run(full_command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during test with bias {bias_value}: {e}")
            continue
        except KeyboardInterrupt:
            print("\nProcess interrupted by user")
            break

def main():
    parser = argparse.ArgumentParser(
        description='Run systematic tests of different bias update values'
    )
    parser.add_argument('--start_from', type=int, default=1,
                      help='Start testing from this bias value (default: 1)')
    parser.add_argument('--end_at', type=int, default=51,
                      help='End testing at this bias value (default: 51)')
    
    args = parser.parse_args()
    
    if not (1 <= args.start_from <= 51):
        raise ValueError("start_from must be between 1 and 51")
    if not (1 <= args.end_at <= 51):
        raise ValueError("end_at must be between 1 and 51")
    if args.start_from > args.end_at:
        raise ValueError("start_from cannot be greater than end_at")
    
    run_bias_tests()

if __name__ == "__main__":
    main()
