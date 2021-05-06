# -*- coding: utf-8 -*-
"""
Created on wed April 28 01:45:30 2021

@author: raonav
"""

#standard Library

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import _pickle as cPickle
import gzip
from numpy import genfromtxt
from Mnist_loader import load_data
from Mnist_loader import load_data_wrapper
import json
from Mnist_loader import vectorized_result, sigmoid, sigmoid_prime
import sys


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        return (a-y)
class Network(object):
    def __init__(self, sizes, cost=CrossEntropyCost):
#        print("Initialized Network")
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                    for x, y in zip(sizes[:-1], sizes[1:])]
        self.cost = cost
        
#        print(np.size(self.weights))
#        print(np.shape(self.weights[0]))
#        print(np.shape(self.weights[1]))
#        print(np.shape(self.biases[0]))
#        print(np.shape(self.biases[1]))
        load_data_wrapper()
#        print(m)
        
#        r = self.training_data()
#        print(r)
#        training_data, test_data, validation_data = m
#        self.training_data = training_data
#        self.test_data = test_data
#        self.validation_data = test_data
        
#        m.load_data()
        
#        r = load_data()
#        print("loading data completed")
#        print(r)
#        ti = load_data_wrapper()
#        print("Dividing data")
#        print(ti)
    def feedforward(self, a):
#        print("feed forward started")
        """it returns the output of the network if ''a''is input"""
        for b, w in zip(self.biases, self.weights):
#            print(np.size(self.weights))
            a = sigmoid(np.dot(w, a)+b)
           # print("feed forward successfully completed")
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda = 0.0, validation_data = None, monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        
#        print("SGD function started")
        if validation_data: n_test = len(validation_data)
#        print(test_data, "len of test data")
        n= len(training_data)
#        print(np.shape(training_data[0][1]))
        train_cost = []
#        train_acc  = []
        Graph_test = []
        Graph_test1 = []
        accu_cal = []
        acc_cal1 = []
        validation_cost = []
        val_acc = []
        train_cal = []
        training_accuracy = []

        for j in range(epochs):
#            print("entered into for loop")
#            random.shuffle(training_data, random )
            random.shuffle(training_data)
#            print("shuffle training data")
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            print("lets divide mini batches")
            for mini_batch in mini_batches:
               # print("dividing mini batches")
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
                
            if monitor_training_cost:
                cost = self.cost_fun(training_data, lmbda)
                train_cost.append(cost)
                print ("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.evaluate(training_data, convert=True)
                training_accuracy.append(accuracy)
                train_cal = float((self.evaluate(training_data, convert=True))/n)
#                acc_cal1.append(train_cal)
                print ("training Accuracy on training data {}: {} / {} = {}".format(j,
                    accuracy, n, train_cal ))
            if monitor_evaluation_cost:
                cost = self.cost_fun(validation_data, lmbda, convert=True)
                validation_cost.append(cost)
                print ("Cost on validation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.evaluate(validation_data)
                val_acc.append(accuracy)
                Cal = float((self.evaluate(validation_data))/n_test)
#                accu_cal.append(Cal)
                print("validation accuracy on evaluation of Epoch: {} : {} / {} = {}".format(j, (self.evaluate(validation_data)), n_test,Cal))
        
            
            
            
#       plotting the graph for accurancy 
            Graph_test.append(j)
#            print(np.shape(Graph_test))
            accu_cal.append(Cal)
#            print(np.shape(accu_cal))
            Graph_test1.append(j)
#            print(np.shape(Graph_test1))
            acc_cal1.append(train_cal)
#            print(np.shape(acc_cal1))
        
        plt.plot(Graph_test, accu_cal, label = "valAcc")
#        
        plt.plot(Graph_test1, acc_cal1, label = "trainAcc")
        plt.xlabel('Epochs')
        plt.ylabel('accuracy of both val&train data')
        plt.title(' Accurancy graph')
        plt.legend() 
        plt.show()

# plotting the graph for cost funtion 
        plt.plot(Graph_test, validation_cost, label = "valcost")
        
        plt.plot(Graph_test1, train_cost, label = "traincost")
        plt.xlabel('Epochs')
        plt.ylabel('Cost of both val & training data')
        plt.title('Cost funtion graph')
        plt.legend() 
        plt.show()
#        
        print()
        return train_cost, training_accuracy, \
            validation_cost, val_acc 
                
    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
#        print("my both weights and biases are enabled")
        for x, y in mini_batch:
#            print("print entered into update_mini_batch funtion")
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
#            print("back to update mini batch")
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
#        print("done with update mini batch funtion")
                      
    def backprop(self, x, y):
#        print("entered into backprop")
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
#        print("both my nabla_b and nabla_w")
#        feedforward
        activation = x  
#        list to store all activations layer by layer
        activations = [x]
#        list to store all z vectores layer by layer
        zs = []
        for b, w in zip(self.biases, self.weights):
#            print(np.shape(w))
#            print(np.shape(activation))
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
#       backword pass
        delta = self.cost_der(activations[-1], y) * sigmoid_prime(zs[-1])
           # sigmoid(z)*(1-sigmoid(z))

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
#        note : variable l in below layer is very little usefull.   
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
        
    def evaluate(self, validation_data, convert=False):
        """The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data."""
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in validation_data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in validation_data]
        return sum(int(x == y) for (x, y) in results)


   
    
    def cost_fun(self, validation_data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data."""
        cost = 0.0
        for x, y in validation_data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(validation_data)
        cost += 0.5*(lmbda/len(validation_data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost


    def cost_der(self, output_activations, y):
    
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        
        return (output_activations-y)


if __name__ == "__main__":
    print("Main")

#    Network([74, 30, 10])
    s = Network([784, 30, 10], cost=CrossEntropyCost)
    training_data, validation_data, test_data = load_data_wrapper()
#    print(training_data, "training data")
#    n = s.feedforward(30)
    s.SGD(training_data, 30, 10, 3.0, validation_data = validation_data, monitor_evaluation_cost=True, monitor_evaluation_accuracy=True,
          monitor_training_cost=True, monitor_training_accuracy=True)
    
#    n = s.feedforward(30)
#    print(n)
#    r = s.SGD([12, 12], 20, 30, 40)
#    print(r)
#    s.update_mini_batch(n[0], 20)

#### Miscellaneous functions

