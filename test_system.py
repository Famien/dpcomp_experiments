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

# run four algs on each files
seed = 2
num_bins = 50

all_results = []
for data_file in data_files:
	dataset = np.load(data_file)
	epsilon = .01
	w = workload.Prefix1D(domain_shape_int=len(dataset))
	results = {}
	for alg_engine in alg_engines:
		dataset_hat = alg_engine.Run(w, dataset, epsilon, seed)
		histogram, bin_size = algs.get_histogram(dataset, num_bins)
		private_hist, bin_size = algs.get_histogram(dataset_hat, num_bins)
		error = algs.get_scaled_error(histogram, private_hist)
		results[alg_engine.short_name] = error

	predictions = pickle.load( open( "model_predictions.p", "rb" ) )


	print "prediction: ", predictions[i]
	predicted_error = predictions[i]['dataset_stat'][2]
	predicted_epsilon = predictions[i][alg_name][0]
	print "results: ", exp_results[i]


	
	all_results.append(results)

pickle.dump(all_results, open("experiment_results.p", "wb"))