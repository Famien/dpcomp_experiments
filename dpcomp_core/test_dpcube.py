import experiment
import dataset
import metric
import workload
from algorithm import *	

def test_DPcube1D():
        A = DPcube1D.DPcube1D_engine()
        E = experiment.Single([1,2,3,4,5], 
                              self.W1, 
                              A, 
                              epsilon=self.expr_eps, 
                              seed=self.expr_seed).run()

        E_dict = E.asDict()
        self.assertEqual(self.expr_seed, E_dict['seed'])
        self.assertEqual(self.expr_eps, E_dict['epsilon'])

test_DPcube1D()