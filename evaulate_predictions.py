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


mypath = join(os.getcwd(), "dpcomp_core/datafiles/1D")
data_files = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

alg_engines = [DPcube1D.DPcube1D_engine(),dawa.dawa_engine(),ahp.ahpND_engine(),HB.HB_engine()]

predictions = pickle.load( open( "/home/famien/Code/MENG/regression/model_predictions.p", "rb" ) )

# run four algs on each files
seed = 2
num_bins = 50

error_errors = []
all_results = []
num_correct = 0

for i in range(len(data_files)):
	data_file = data_files[i]
	dataset = np.load(data_file)
	epsilon = .01
	w = workload.Prefix1D(domain_shape_int=len(dataset))
	results = {}
	predicted_error = predictions[i]['dataset_stat'][2]

	for alg_engine in alg_engines:
		predicted_epsilon = predictions[i][alg_engine.short_name][0]
		dataset_hat = alg_engine.Run(w, dataset, predicted_epsilon, seed)
		histogram, bin_size = algs.get_histogram(dataset, num_bins)
		private_hist, bin_size = algs.get_histogram(dataset_hat, num_bins)
		error = algs.get_scaled_error(histogram, private_hist)
		error_errors.append(abs(predicted_error - error))
		results[alg_engine.short_name] = error

	actual_best = min(results, key =results.get)
	predictions_algs = {}
	for key in predictions[i].keys():
		if key == 'dataset_stat':
			continue
		predictions_algs[key] = predictions[i][key][0] #isolate epsilons

	predicted_best = min(predictions_algs, key = predictions_algs.get)
	if actual_best == predicted_best:
		num_correct += 1
	#all_results.append(results)

#pickle.dump(all_results, open("experiment_results.p", "wb"))
print "avg error: ", sum(error_errors)/len(error_errors)
print "'%'' correct: ", (num_correct/float(len(error_errors))*len(alg_engines))*100