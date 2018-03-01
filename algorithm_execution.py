from __future__ import division
from __future__ import print_function
from past.utils import old_div
from dpcomp_core.algorithm import *
from dpcomp_core import dataset
from dpcomp_core import util
from dpcomp_core import workload
import numpy as np
import algorithms as algs
from experiments import *
'''
example of running an algo on synthetic data
'''

# get data
x = np.load("/home/famien/Code/MENG/regression/DATA2.npy")

# Instantiate algorithm
algs = [ DPcube1D.DPcube1D_engine(), dawa.dawa_engine(), ahp.ahpND_engine(), HB.HB_engine()]

# Setup parameters
num_bins = 50
seed = 1
epsilons1= [float(i)/10000 for i in range(1,1000)]
epsilons2 = [float(i)/1000 for i in range(10, 10000)]
epsilons = epsilons1 + epsilons2

for alg_engine in algs:
	results = run_experiment(x, alg_engine, epsilons, seed, num_bins)
	np.save(alg_engine.short_name+"_results.npy", np.array(results))
