# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 00:45:21 2019

@author: raona
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import _pickle as Pickle
import gzip
from numpy import genfromtxt


def load_data():
    print("loading data")
    f = gzip.open('C:/Mnist Data/data/mnist.pkl.gz', 'rb')
    print("loaded")
    training_data, validation_data, test_data = Pickle.load(f,encoding='latin1')
    f.close()
#    print(np.shape(training_data))
#    print(np.shape(training_data))
#    print(np.shape(training_data))
#    print(training_data[0])
    return(training_data, validation_data , test_data)
    	
     
def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    
#    print('gzipped pickled data loaded successfully')
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    print(np.shape(training_inputs))
#    print("my training inputs %f", training_inputs )
    training_results = [vectorized_result(y) for y in tr_d[1]]
#    print("my training result %f", training_results)	
    training_data = list(zip(training_inputs, training_results))
    print(np.shape(training_data))
#    print(training_data.__sizeof__)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
#    print(test_data)
    print("got my training, test and validation data")
    return(training_data, validation_data, test_data)
    
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))

    e[j] = 1.0
    print(e)
    return e

def sigmoid(z):

    """The sigmoid function."""
    print(np.shape(z))
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
