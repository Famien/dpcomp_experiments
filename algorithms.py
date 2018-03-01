import numpy as np
from random import random
import math

'''
	Takes a dataset B, a set of queries, num_iterations and epsilon
	
	returns a private version of B using Multiplicative Weights and Exponential Mechansim Algorithm
'''
def mwem(B, queries, num_iterations, epsilon):
	n = sum(B)
	A = get_uniform(B)
	A = np.array(A)
	measurements = dict()
	num_iterations = min(num_iterations, len(queries)) # can't iterator for more queries than we have	
	for i in range(num_iterations):
		query_i = exponential_mechanism(B, A, queries, epsilon)
		times = 1
		while query_i in measurements.keys():
			query_i = exponential_mechanism(B, A, queries, epsilon, measurements)
			times +=1
		measurements[query_i] = get_private_measure(queries[query_i], B, num_iterations, epsilon) 
		A = multiplicative_weights(A, measurements, queries)
		
	return A
def exponential_mechanism(B, A, queries, epsilon, measurements = {}):
	errors = []
	for i in range(len(queries)):
		q_result_B = evaluate_query(B, queries[i])
		q_result_A = evaluate_query(A, queries[i])
		errors.append(epsilon*abs(q_result_B - q_result_A)/2.0)	
	
	for measure in measurements.keys():
		errors[measure] = 0 # don't use previously selected keys ???

	max_error = max(errors)
	for i in range(len(errors)):
		errors[i] = math.e**(errors[i] - max_error)

	uniform = sum(errors)*random()
	for i in range(len(errors)):
		uniform -= errors[i]
		if (uniform <= 0):
			return i
	
	return len(errors) - 1 

def get_queries(num_bins, domain_size):
	queries = []
	bin_size = domain_size/ num_bins
	index = 0
	bin_num = 1
	while index < domain_size - 1:
		left_query = index
		right_query = index + bin_size
		if bin_num == num_bins: # last bin, so collect the rest of the domain
			right_query = domain_size - 1
		
		query = [left_query, right_query]
		queries.append(query)
		index = right_query 
		bin_num += 1
	return queries
	
def multiplicative_weights(A, measurements, queries):
	total = sum(A)
	for i in range(50):
		for query_i in measurements.keys():	
			error = measurements[query_i] - evaluate_query(A, queries[query_i])
			query_ranges = queries[query_i]
			for i in range(query_ranges[0], query_ranges[1] + 1):
				A[i] *= math.e**(query_component(queries[query_i], i)*error/(2.0*total))
			count = np.sum(A)
			A *= total/float(count)	
	return A

def get_private_measure(query, B, T, epsilon):
	measurement = evaluate_query(B, query)
	laplace_noise = get_laplace_noise(epsilon/float(2*T))
	return measurement + laplace_noise

def get_uniform(A):
	total = sum(A)
	size = len(A)
	uniform = []
	dist = int(total/float(size))
	for i in range(size):
		uniform.append(dist)
	
	return uniform

def query_component(q_range, i):
	if i <= q_range[1] and i >= q_range[0]:
		return 1
	return 0

def evaluate_query_slow(A, q_range):
	total = 0
	for i in range(len(A)):
		total += query_component(q_range, i) * A[i]
	return total

def evaluate_query(A, q_range):
	return np.sum(A[q_range[0]:q_range[1]+1])

def get_histogram(counts, num_bins):
    '''
        creates a histogram of array counts with num_bins bins
        returns histogram and bin_size
    '''
    hist = []
    domain = len(counts)
    bin_size = max(1, domain/ num_bins) # make sure we have at least 1 bin
    bin_sum = 0
    for i in range(len(counts)):
        bin_sum += counts[i]
        if i % bin_size == 0:
            hist.append(bin_sum)
            bin_sum =0
    hist.append(bin_sum)
    ''' 
    for i in range(num_bins):
        sum_i = i*bin_size
        bucket_sum = 0
        while sum_i < (i*bin_size + bin_size) and sum_i < len(counts):
            bucket_sum += counts[sum_i]
            sum_i += 1  
        hist.append(bucket_sum)
    '''

    return hist, bin_size

def get_scaled_error(actual_counts, expected_counts):
    total_error = 0
    for i in range(len(actual_counts)):
        total_error += abs(actual_counts[i] - expected_counts[i])
    return total_error / sum(actual_counts)


def merge_data_vector(data_vector, new_size):
    new_num_bins = new_size * len(data_vector)
    merge_size = len(data_vector)/new_num_bins
    new_data_vector = []

    bin_sum = 0
    for i in range(len(data_vector)):
        bin_sum += data_vector[i]
        if i % merge_size == 0:
            new_data_vector.append(bin_sum)
            bin_sum = 0
    new_data_vector.append(bin_sum)
    return new_data_vector

def get_laplace_noise(e):
        loc, scale = 0., 1/float(e)
        return np.random.laplace(loc, scale, 1)[0] #return first and only result

def get_laplace_hist(counts, e):
    '''
        takes a histogram, counts, and epsilon e
        returns private histogram
    '''
    output = []
    for i in range(len(counts)):
        new_count = counts[i] + get_laplace_noise(e)[0]
        output.append(new_count)
    return output

def uniform_distance(A):
	total = sum(A)
	avg = total/len(A)
	
	distance = 0
	for element in A:
		distance += abs(avg - element)
		
	scaled_distance = (distance/total)/2
	
	return scaled_distance