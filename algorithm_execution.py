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
from os import listdir
from os.path import isfile, join
'''
example of running an algo on synthetic data
'''

# get data
x = np.load("/home/famien/Code/MENG/regression/DATA6.npy")

mypath = "/home/famien/Code/pipe/1D"
data_files = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

# 2. get data vectors from files
# x = []

# for data_file in data_files:
# 	data_from_file = np.load(data_file)
# 	x.append(data_from_file)


# Instantiate algorithm
algs = [ DPcube1D.DPcube1D_engine(), dawa.dawa_engine(), ahp.ahpND_engine(), HB.HB_engine()]
algs = [algs[2]]
# Setup parameters
num_bins = 50
seed = 1
epsilons1= [float(i)/10000 for i in range(1,1000)]
epsilons2 = [float(i)/1000 for i in range(10, 10000)]
epsilons = epsilons1
epsilons = [float(i)/26 for i in range(1,50)]

print("len datasets: ", len(x))
print("Num epsilons: ", len(epsilons))
print("total : ", len(x)*len(epsilons))

for alg_engine in algs:
	results = run_experiment(x, alg_engine, epsilons, seed, num_bins)
	np.save("/home/famien/Code/pipe/" +alg_engine.short_name+"_data_6.npy", np.array(results))

# 1 = 