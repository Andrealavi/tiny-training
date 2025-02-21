#!/usr/bin/env python3
import subprocess
import argparse
from datetime import datetime
from typing import Optional

def run_manual_bias_tests(start_from: int = 0, end_at: int = 50) -> None:
    """
    Execute the training script for each manual_bias_idx value from start_from to end_at

    Args:
        start_from (int): Starting index for testing (default: 0)
        end_at (int): Ending index for testing (default: 50)
    """
    base_command = """python train_cls.py configs/transfer.yaml \
        --run_dir runs/flowers/mcunet-5fps/sparse_124kb/sgd_qas_nomom \
        --net_name mbv2-w0.35 \
        --bs256_lr 0.1 \
        --optimizer_name sgd_scale_nomom \
        --enable_backward_config 1 \
        --n_weight_update 0"""

    total_tests = end_at - start_from + 1

    for idx, bias_idx in enumerate(range(start_from, end_at + 1)):
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{current_time}] Test {idx + 1}/{total_tests}")
        print(f"Testing with manual_bias_idx: {bias_idx}")

        # Construct the full command with manual bias index
        full_command = f"{base_command} --manual_bias_idx {bias_idx}"

        try:
            # Execute the command
            subprocess.run(full_command, shell=True, check=True)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Completed test with manual_bias_idx={bias_idx}")
        except subprocess.CalledProcessError as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Error during test with manual_bias_idx={bias_idx}: {e}")
            continue
        except KeyboardInterrupt:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Process interrupted by user")
            break

def main():
    parser = argparse.ArgumentParser(
        description='Run systematic tests of different manual bias index values'
    )
    parser.add_argument('--start_from', type=int, default=0,
                      help='Start testing from this bias index (default: 0)')
    parser.add_argument('--end_at', type=int, default=50,
                      help='End testing at this bias index (default: 50)')

    args = parser.parse_args()

    if not (0 <= args.start_from <= 50):
        raise ValueError("start_from must be between 0 and 50")
    if not (0 <= args.end_at <= 50):
        raise ValueError("end_at must be between 0 and 50")
    if args.start_from > args.end_at:
        raise ValueError("start_from cannot be greater than end_at")

    print(f"\nStarting test suite at {datetime.now().strftime('%H:%M:%S')}")
    print(f"Will run {args.end_at - args.start_from + 1} tests")
    print(f"Testing manual_bias_idx values from {args.start_from} to {args.end_at}")

    run_manual_bias_tests(args.start_from, args.end_at)

    print(f"\nAll tests completed at {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
