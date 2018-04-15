import math
import numpy as np
import algorithms as algs
from dpcomp_core import workload

def run_experiment(datasets, alg_engine, epsilons, seed, num_bins):
	total_runs = len(epsilons)*len(datasets)
	print("total runs: ", total_runs)
	num_done = 0
	experiment_results = []
	for i in  in range(len(datasets)):
		dataset = datasets[i]
		if sum(dataset) == 0 or len(dataset) <= 2:
			print("bad dataset")
			continue # for some reason there are '0' data vectors
			# also, for branching, length should be at least 3

		dataset = np.array(dataset)
		scale = sum(dataset)
		domain_size = len(dataset)
		data_range = max(dataset) - min(dataset)
		std_dev = math.sqrt(np.var(dataset))
		uniform_distance = algs.uniform_distance(dataset)

		for epsilon in epsilons:
			w = workload.Prefix1D(domain_shape_int=len(dataset))
			dataset_hat = alg_engine.Run(w, dataset, epsilon, seed)

			histogram, bin_size = algs.get_histogram(dataset, num_bins)
			private_hist, bin_size = algs.get_histogram(dataset_hat, num_bins)
			error = algs.get_scaled_error(histogram, private_hist)

			experiment_results.append((scale, domain_size, error, data_range, std_dev, uniform_distance, epsilon, data_set_index, i))
			num_done +=1
			if num_done % 50 ==0 :
				print("num done: ", num_done)
	return experiment_results