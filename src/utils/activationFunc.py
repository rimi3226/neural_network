# coding: utf-8
import numpy as np

def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))    


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
    

def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    # 입력이 스칼라인 경우 배열로 변환
    x = np.array(x, ndmin=1)
    grad = np.zeros_like(x)
    grad[x >= 0] = 1
    return grad
    
    
def leaky_relu(x):
    return np.where(x > 0, x, 0.01 * x)


def leaky_relu_grad(x):
    grad = np.where(x > 0, 1, 0.01)
    return grad


def tanh(x):
    return np.tanh(x)


def tanh_grad(x):
    return 1 - np.tanh(x) ** 2

def softmax(x):
    # 입력에서 최대값을 빼 오버플로우 방지
    x = x - np.max(x, axis=-1, keepdims=True)  
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


def cross_entropy_loss(y_pred, y_true):
    log_y_pred = np.log(y_pred + 1e-9)  # log(0) 방지
    return -np.sum(y_true * log_y_pred) / y_pred.shape[0]



def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)