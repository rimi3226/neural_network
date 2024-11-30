import replaceNumpy as rnp
import math
import numpy as np

def sigmoid(matrix):
    # sigmoid(x) = 1 / (1 + exp(-x))
    exp_neg_matrix = rnp.exp([[-x for x in row] for row in matrix])
    return [[1 / (1 + exp_neg) for exp_neg in row] for row in exp_neg_matrix]

def tanh(matrix):
    return rnp.tanh(matrix)  

def relu(matrix):
    return [[rnp.maximum(x, 0) for x in row] for row in matrix] 

def leaky_relu(matrix, alpha=0.01):
    return [[x if x > 0 else alpha * x for x in row] for row in matrix]  # Leaky ReLU 구현

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()