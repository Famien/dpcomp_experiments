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

models = joblib.load("models_final.pkl")


def uniform_distance(A):
	total = sum(A)
	avg = total/len(A)	
	distance = 0
	for element in A:
		distance += abs(avg - element)

	scaled_distance = (distance/total)/2.0

	return scaled_distance

dataset_vectors = []
mypath = join(os.getcwd(), "dpcomp_core/datafiles/1D")
data_files = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
for data_file in data_files:
	data_from_file = np.load(data_file)
	dataset_vectors.append(data_from_file)

alg_engines = [DPcube1D.DPcube1D_engine(),dawa.dawa_engine(),ahp.ahpND_engine(),HB.HB_engine()]
# run four algs on each files
seed = 2
num_bins = 50
epsilons = [float(i)/100 for i in range(1,100)]

print("total : ", len(dataset_vectors)*len(epsilons))
all_results = []
num_done = 0
errors = [x/float(100) for x in range(200)]
error_pairs = []
print "total runs : ", len(epsilons)*len(dataset_vectors)
for i in range(len(dataset_vectors)):
	# dataset = np.load(data_file)

	dataset = np.array(dataset_vectors[i])
	results = {}
	results["dataset"] = dataset
	scale = sum(dataset)
	domain_size = len(dataset)
	scale = sum(dataset)
	data_range = max(dataset) - min(dataset)
	std_dev = math.sqrt(np.var(dataset))
	for error in errors:
		dataset_stat = [scale, domain_size, error, data_range, std_dev, uniform_distance(dataset)]
		alg_epsilon_info = {}

		w = workload.Prefix1D(domain_shape_int=len(dataset))
		for alg_engine in alg_engines:
			model = models[alg_engine.short_name]
			predicted_epsilon =  model.predict([dataset_stat])[0]
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
		
		# best_alg = min(alg_epsilon_info, key = lambda x : x[1])
		# predicted_alg = min(alg_epsilon_info, key = lambda x : x[1])
		#print "results: ", results
		#break
		all_results.append(results)

print len(all_results)

pickle.dump(error_pairs, open("/home/famien/Code/pipe/error_pairs_real_data.p", "wb"))

#pickle.dump(all_results, open("mini_experiment_results2.p", "wb"))