import argparse
import subprocess
import sys
import concurrent.futures
from datetime import datetime

def run_single_test(n_bias_update: int, num_classes: int, total_tests: int, current_test: int) -> None:
    """
    Run a single test with specified parameters

    Args:
        n_bias_update: Number of bias layers to update
        num_classes: Number of classes for classification
        total_tests: Total number of tests to run
        current_test: Current test number
    """
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}] Starting test {current_test}/{total_tests} (n_bias_update={n_bias_update})")

    cmd = [
        sys.executable,
        "mod.py",
        "--n_bias_update", str(n_bias_update),
        "--weight_idx", "0",  # Not used when bias_only is True
        "--num_classes", str(num_classes),
        "--bias_only"
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Completed test {current_test}/{total_tests} (n_bias_update={n_bias_update})")
    except subprocess.CalledProcessError as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Error in test {current_test}/{total_tests} (n_bias_update={n_bias_update}): {e}")

def main():
    parser = argparse.ArgumentParser(description='Automate execution of mod.py with different bias updates')
    parser.add_argument('--num_classes', type=int, required=True,
                      help='Number of classes for classification')
    parser.add_argument('--start', type=int, default=1,
                      help='Starting number of bias updates (default: 1)')
    parser.add_argument('--end', type=int, default=51,
                      help='Ending number of bias updates (default: 51)')
    parser.add_argument('--step', type=int, default=5,
                      help='Step size between bias update values (default: 5)')
    parser.add_argument('--parallel', action='store_true',
                      help='Run tests in parallel using ProcessPoolExecutor')
    parser.add_argument('--max_workers', type=int, default=4,
                      help='Maximum number of parallel workers when using --parallel')
    args = parser.parse_args()

    # Generate range of n_bias_update values
    bias_updates = list(range(args.start, args.end + 1, args.step))
    total_tests = len(bias_updates)

    print(f"\nStarting test suite at {datetime.now().strftime('%H:%M:%S')}")
    print(f"Will run {total_tests} tests with n_bias_update values: {bias_updates}")

    if args.parallel:
        print(f"\nRunning tests in parallel with {args.max_workers} workers...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(run_single_test, n_bias, args.num_classes, total_tests, idx + 1)
                for idx, n_bias in enumerate(bias_updates)
            ]
            concurrent.futures.wait(futures)
    else:
        print("\nRunning tests sequentially...")
        for idx, n_bias in enumerate(bias_updates, 1):
            run_single_test(n_bias, args.num_classes, total_tests, idx)

    print(f"\nAll tests completed at {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
