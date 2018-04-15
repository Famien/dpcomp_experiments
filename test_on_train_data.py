import os
from os import listdir
from os.path import isfile, join
import numpy as np
import math
from dpcomp_core.algorithm import *
from dpcomp_core import dataset
from dpcomp_core import util
from dpcomp_core import workload
import algorithms as algs
import pickle
from sklearn.externals import joblib
import numpy as np

models = joblib.load("models_6.pkl")
test_data = np.load("/home/famien/Code/pipe/AHP_data_6_test.npy")
datasets = np.load("/home/famien/Code/MENG/regression/DATA6.npy")

alg_engines = [DPcube1D.DPcube1D_engine(),dawa.dawa_engine(),ahp.ahpND_engine(),HB.HB_engine()]
# run four algs on each files
seed = 2

all_results = []
num_done = 0

for alg_engine in alg_engines:
	model = models[alg_engine.short_name]
	for dataset_stat in test_data:
		scale = dataset_stat[0]
		domain_size = dataset_stat[1]
		actual_error = dataset_stat[2]
		data_range = dataset_stat[3]
		std_dev = dataset_stat[4]
		uniform_distance = dataset_stat[5]
		actual_epsilon = dataset_stat[6]
		datset = datasets[dataset_stat[7]]

		w = workload.Prefix1D(domain_shape_int=len(domain_size))

		dataset_hat = alg_engine.Run(w, dataset, predicted_epsilon, seed)
		histogram, bin_size = algs.get_histogram(dataset, num_bins)
		private_hist, bin_size = algs.get_histogram(dataset_hat, num_bins)
		actual_error = algs.get_scaled_error(histogram, private_hist)
		alg_epsilon_info[alg_engine.short_name] = (predicted_epsilon, actual_error)
		error_pairs.append((error, actual_error))

		#results["runs"].append((alg_engine.short_name, epsilon, error))
		num_done +=1
		if num_done % 50 ==0 :
			print("num done: ", num_done)

pickle.dump(error_pairs, open("/home/famien/Code/pipe/error_pairs_real_data.p", "wb"))

#pickle.dump(all_results, open("mini_experiment_results2.p", "wb"))