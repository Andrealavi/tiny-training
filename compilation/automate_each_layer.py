import argparse
import subprocess
import sys
import concurrent.futures
from datetime import datetime

def run_single_test(weight_idx: int, n_bias_update: int, num_classes: int, total_tests: int, current_test: int) -> None:
    """
    Run a single test with specified parameters

    Args:
        weight_idx: Index of the weight layer to update
        n_bias_update: Number of bias layers to update
        num_classes: Number of classes for classification
        total_tests: Total number of tests to run
        current_test: Current test number
    """
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}] Starting test {current_test}/{total_tests} (weight_idx={weight_idx})")

    cmd = [
        sys.executable,
        "mod.py",
        "--n_bias_update", str(n_bias_update),
        "--weight_idx", str(weight_idx),
        "--num_classes", str(num_classes)
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Completed test {current_test}/{total_tests} (weight_idx={weight_idx})")
    except subprocess.CalledProcessError as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Error in test {current_test}/{total_tests} (weight_idx={weight_idx}): {e}")

def main():
    parser = argparse.ArgumentParser(description='Automate execution of mod.py with different parameters')
    parser.add_argument('--num_classes', type=int, required=True,
                      help='Number of classes for classification')
    parser.add_argument('--parallel', action='store_true',
                      help='Run tests in parallel using ProcessPoolExecutor')
    parser.add_argument('--max_workers', type=int, default=4,
                      help='Maximum number of parallel workers when using --parallel')
    args = parser.parse_args()

    # Fixed parameters
    n_bias_update = 51
    weight_indices = list(range(51))  # 0 to 50 inclusive
    total_tests = len(weight_indices)

    print(f"\nStarting test suite at {datetime.now().strftime('%H:%M:%S')}")
    print(f"Will run {total_tests} tests with weight_idx values: {weight_indices}")
    print(f"Using fixed n_bias_update={n_bias_update}")

    if args.parallel:
        print(f"\nRunning tests in parallel with {args.max_workers} workers...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(run_single_test, idx, n_bias_update, args.num_classes, total_tests, idx + 1)
                for idx in weight_indices
            ]
            concurrent.futures.wait(futures)
    else:
        print("\nRunning tests sequentially...")
        for idx in weight_indices:
            run_single_test(idx, n_bias_update, args.num_classes, total_tests, idx + 1)

    print(f"\nAll tests completed at {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
