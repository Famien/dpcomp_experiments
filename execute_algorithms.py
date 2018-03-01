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

# get data
datasets = np.load("DATA1.npy")

# Instantiate algorithm
# a = DPcube1D.DPcube1D_engine()
a = mwemND.mwemND_engine()
# Instantiate workload

# Calculate noisy estimate for x
seed = 1
epsilons1= [float(x)/100 for x in range(1,10)]
epsilons2 = [float(x)/10 for x in range(10, 20)]
epsilons = epsilons1 + epsilons2
num_bins = 50

run_experiment(datasets, a, epsilons, seed, num_bins)