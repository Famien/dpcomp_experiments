from sklearn import linear_model
from sklearn import datasets
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import numpy as np
import csv
import matplotlib.pyplot as plt
import pylab
import random
import sys
import math
import pickle
from sklearn.externals import joblib

'''
generate model from synthetic data
'''

# fields: scale, domain_size, error, data_range, std_dev, uniform_distance, epsilon
#           0         1         2       3           4           5              6
algs = ["HB", "AHP", "DPCube", "DAWA"]
features = ["scale", "domain_size", "error", "data_range", "std_dev", "uniform_distance"]

for alg in algs:

	data = np.load("/home/famien/Code/pipe/"+alg+"_data_5.npy")
	'''
	split into train and test data
	'''
	train = []
	test = []
	for i in range(len(data)):
		if random.random() >= .8:
			train.append(i)
		else:
			test.append(i)

	train_X = []
	train_y = []
	test_X = []
	test_y = []

	for index in train:
		train_X.append(data[index][0:6])
		train_y.append(data[index][6])

	for index in test:
		test_X.append(data[index][0:6])
		test_y.append(data[index][6])

	print "train len: ", len(train_X)

	X_ = train_X
	y = train_y

	regr = RandomForestRegressor()
	#regr = DecisionTreeRegressor(random_state=0)
	regr.fit(X_,y)
	try:
		models = joblib.load("models_final.pkl")
	except IOError:
		models = {}
	models[alg] = regr
	joblib.dump(models, "models_final.pkl")

	epsilon_predict = regr.predict(test_X)

	#print "accuracy: ", regr.score(test_X,test_y)
	print("Alg: ", alg)
	print("\tvar explained: ", r2_score(test_y, epsilon_predict))
	print(sorted(zip(map(lambda x: float("{0:.2f}".format(round(x, 4))), regr.feature_importances_), features),
	             reverse=True))