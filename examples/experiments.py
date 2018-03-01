def run_experiment(datasets, alg_engine, epsilons, seed, num_bins):
	experiment_results = []

	print("datasets: ", datasets)
	for dataset in datasets:
		scale = sum(dataset)
		domain_size = len(dataset)
		data_range = max(dataset) - min(dataset)
		std_dev = math.sqrt(np.var(data_vector))
		uniform_distance = algs.uniform_distance(data_vector)

		for epsilon in epsilons:
			w = workload.Prefix1D(domain_shape_int=len(dataset))
			dataset_hat = alg_engine.Run(dataset, x, epsilon, seed)

			histogram, bin_size = algs.get_histogram(x, num_bins)
			private_hist, bin_size = algs.get_histogram(x_hat, num_bins)
			error = algs.get_scaled_error(histogram, private_hist)

			experiment_results.append((scale, domain_size, error, data_range. std_dev, uniform_distance, epsilon))

	return experiment_results