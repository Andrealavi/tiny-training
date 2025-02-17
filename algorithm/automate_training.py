#!/usr/bin/env python3
import subprocess
import random
import argparse
from typing import List, Set, Tuple


def generate_weight_indices(num_indices: int, n_bias_update: int, max_value: int = 50) -> str:
	"""
	Generate random sorted weight indices with higher probability for specific values,
	ensuring the smallest index is not less than 51 - n_bias_update
	"""
	# Preferred values with higher weights
	preferred_values = [48, 45, 42, 39, 36, 33]
	min_allowed_value = 51 - n_bias_update

	# Filter preferred values based on constraint
	valid_preferred = [v for v in preferred_values if v >= min_allowed_value]

	# Create weighted population with valid values only
	weighted_population = [i for i in range(max_value + 1) if i >= min_allowed_value]
	for value in valid_preferred:
		weighted_population.extend([value] * 4)  # Add 4 more occurrences (5x total)

	if not weighted_population:
		raise ValueError(f"No valid indices available with n_bias_update={n_bias_update}")

	# Generate random indices and ensure they're unique
	indices: Set[int] = set()
	while len(indices) < num_indices:
		if len(weighted_population) < num_indices - len(indices):
			raise ValueError(f"Not enough valid indices available for selection")
		indices.add(random.choice(weighted_population))

	# Convert to list, sort, and join with hyphens
	return '-'.join(map(str, sorted(indices)))


def generate_update_ratios(num_ratios: int) -> str:
	"""
	Generate random update ratios from the allowed values
	"""
	allowed_ratios = [0, 0.125, 0.25, 0.50, 0.75, 1.0]
	ratios = [random.choice(allowed_ratios) for _ in range(num_ratios)]
	return '-'.join(map(str, ratios))


def run_training(num_runs: int, num_indices: int) -> None:
	"""
	Execute the training script multiple times with different parameters
	"""
	base_command = """python train_cls.py configs/transfer.yaml \
        --run_dir runs/flowers/mcunet-5fps/sparse_124kb/sgd_qas_nomom \
        --net_name mbv2-w0.35 \
        --bs256_lr 0.1 \
        --optimizer_name sgd_scale_nomom \
        --enable_backward_config 1"""

	for run in range(num_runs):
		try:
			# Generate random n_bias_update
			n_bias_update = random.randint(1, 51)

			# Generate new random weight indices and update ratios
			weight_indices = generate_weight_indices(num_indices, n_bias_update)
			update_ratios = generate_update_ratios(num_indices)

			# Count how many preferred values were selected
			preferred_values = set([48, 45, 42, 39, 36, 33])
			selected_preferred = len([x for x in weight_indices.split('-') if int(x) in preferred_values])

			# Construct the full command
			full_command = f"{base_command} \
                --n_bias_update {n_bias_update} \
                --manual_weight_idx {weight_indices} \
                --weight_update_ratio {update_ratios}"

			print(f"\nRun {run + 1}/{num_runs}")
			print(f"Using n_bias_update: {n_bias_update}")
			print(f"Using weight indices: {weight_indices}")
			print(f"Using update ratios: {update_ratios}")
			print(f"Number of preferred values selected: {selected_preferred}")
			print(f"Minimum allowed weight index: {51 - n_bias_update}")

			# Execute the command
			subprocess.run(full_command, shell=True, check=True)

		except ValueError as e:
			print(f"Error during run {run + 1}: {e}")
			print("Retrying with different random values...")
			run -= 1  # Retry this run
			continue
		except subprocess.CalledProcessError as e:
			print(f"Error during run {run + 1}: {e}")
			continue
		except KeyboardInterrupt:
			print("\nProcess interrupted by user")
			break


def main():
	parser = argparse.ArgumentParser(description='Automate training runs with random parameters')
	parser.add_argument('--num_runs', type=int, required=True,
						help='Number of training runs to execute')
	parser.add_argument('--num_indices', type=int, required=True,
						help='Number of weight indices to generate for each run')
	parser.add_argument('--seed', type=int, default=None,
						help='Random seed for reproducibility')

	args = parser.parse_args()

	if args.seed is not None:
		random.seed(args.seed)

	run_training(args.num_runs, args.num_indices)


if __name__ == "__main__":
	main()