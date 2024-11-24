import os
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils")))

import activationFunc as af
import numpy as np
from Node import Node 

class Layer:
    def __init__(self,layer_size,activation):
        self.layer_size=layer_size                      # 레이어 크기
        self.nodes=[Node(activation) for _ in range(layer_size)]  # 레이어에 할당된 노드 리스트
        self.activation=activation                      # 해당 레이어의 활성화 함수
        
    # 노드 값 할당 -입력층
    def setLayer(self,input_data):
        for i in range(self.layer_size):
            self.nodes[i].val = input_data[i]
            
            
    def setWeight(self,weight_init="He"):
        for i in range(self.layer_size):
            self.nodes[i].init_weight(2,weight_init)
            print("weight: ",self.nodes[i].weight)
        
    
    # 순전파 계산
    def forward(self):
        output_val = []
        output_gradient = []
    
        # 노드의 가중치와 값의 행렬곱 계산 및 편향 추가
        prev_val = np.array([node.val for node in self.nodes])
        weights = np.array([node.weight for node in self.nodes])
        multiplication_val = np.round(np.dot(prev_val,weights),8)

        print("multi : ",multiplication_val)

        # 활성화 함수 적용 및 그레디언트 계산
        if self.activation == "sigmoid":
            val = af.sigmoid(multiplication_val)
            gradient = af.sigmoid_grad(multiplication_val)
        elif self.activation == "relu":
            val = af.relu(multiplication_val)
            gradient = af.relu_grad(multiplication_val)
        elif self.activation == "leaky_relu":
            val = af.leaky_relu(multiplication_val)
            gradient = af.leaky_relu_grad(multiplication_val) 
        elif self.activation == "tanh":
            val = af.tanh(multiplication_val)
            gradient = af.tanh_grad(multiplication_val)  # tanh의 도함수
            
        # 다음 레이어로 전달될 값 리스트에 추가
        output_val.append(val)
        output_gradient.append(gradient)
        return output_val, output_gradient
        
    # 역전파 계산
    def backward(self):
        print()
        