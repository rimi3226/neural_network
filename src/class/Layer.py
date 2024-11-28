import os
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils")))

import activationFunc as af
import numpy as np
from Node import Node 

class Layer:
    def __init__(self,layer_size,activation="relu",next_layer_size=0,learning_rate=0.1,dropout_rate=0.5):
        self.layer_size=layer_size                      # 레이어 크기
        self.nodes=[Node(activation) for _ in range(layer_size)]  # 레이어에 할당된 노드 리스트
        self.activation=activation                      # 해당 레이어의 활성화 함수
        self.next_layer_size=next_layer_size;
        self.learning_rate=learning_rate
        self.dropout_rate = dropout_rate  # Dropout 비율 추가
        self.dropout_mask = np.ones(layer_size)  # Dropout 마스크 초기화

    # Dropout 마스크 생성
    def apply_dropout(self):
        if self.dropout_rate > 0:
            self.dropout_mask = np.random.rand(self.layer_size) > self.dropout_rate
        else:
            self.dropout_mask = np.ones(self.layer_size)

    # 노드 값 할당 
    def set_layer(self,data):
        for i in range(self.layer_size):
            self.nodes[i].val = data[i]
            
    # 가중치 할당 
    def set_weight(self,weight_init="He"):
        # print("=========init weight=========")
        for i in range(self.layer_size):
            self.nodes[i].init_weight(self.next_layer_size,weight_init)
            # print("weight: ",self.nodes[i].weight)
        
    # 활성화 그래디언트 설정
    def set_activation_gradient(self, gradient):
        for i in range(self.layer_size):
            self.nodes[i].activation_gradient = gradient[i]
            
    # 순전파 계산
    def forward(self,what_layer,activation=None):    
        # print(f"================{what_layer}=============")
        if activation == None:
            activation = self.activation
        
        # 1. 노드의 가중치와 값의 행렬곱 계산 및 편향 추가
        val = np.array([node.val for node in self.nodes])
        weights = np.array([node.weight for node in self.nodes])
        multiplication_val = np.dot(val,weights)
        # print("multiplication_val: ",multiplication_val)
        # 2. 활성화 함수 적용 및 그레디언트 계산
        if activation == "sigmoid":
            activation_val = af.sigmoid(multiplication_val)
            gradient = af.sigmoid_grad(multiplication_val)
        elif activation == "relu":
            activation_val = np.maximum(0,multiplication_val)
            gradient = af.relu_grad(multiplication_val)
        elif activation == "leaky_relu":
            activation_val = np.where(multiplication_val > 0, multiplication_val, 0.01 * multiplication_val)
            gradient = af.leaky_relu_grad(multiplication_val) 
        elif activation == "tanh":
            activation_val = af.tanh(multiplication_val)
            gradient = af.tanh_grad(multiplication_val)  
        elif activation == "softmax":
            activation_val = af.softmax(multiplication_val)
            gradient = activation_val * (1-activation_val)
            
        # print("activation_val: ",activation_val)
        return activation_val, gradient
        
    # 출력층 역전파 계산
    def get_gradient_output_MSE(self,target):
        # 1. MSE 사용해서 손실함수 구하기
        output=np.array([node.val for node in self.nodes])
        gradient=-(output-target)

        # 2. 각 노드의 gradient값 구하기
        for i,node in enumerate(self.nodes):
            self.nodes[i].gradient=gradient[i]*node.activation_gradient
            # print("output_gradient : ",self.nodes[i].gradient);
            # print(i," gradient: ",node.gradient)
            # print(i," activation_gradient: ",node.activation_gradient)
            
    # 출력층 역전파 계산
    def get_gradient_output(self, target):
        # 교차 엔트로피 손실 함수 기반 그래디언트 계산
        output = np.array([node.val for node in self.nodes])
        gradient = output - target  # Cross-Entropy Loss의 Gradient 계산

        # 각 노드의 gradient값 설정
        for i, node in enumerate(self.nodes):
            self.nodes[i].gradient = gradient[i]  # 활성화 함수의 미분은 softmax와 합쳐짐

    # 역전파 계산
    def get_gradient(self,next_layer_nodes):
        weights=np.transpose(np.array([node.weight for node in self.nodes]))
        res_multiplication=np.dot(next_layer_nodes,weights)
        # print("weights: ",weights)
        # print("next_layer_node * weight transpose : ",res_multiplication)
        
        for i,node in enumerate(self.nodes):
            node.gradient=np.round(res_multiplication[i]*node.activation_gradient,10)
            # print("grad : ",node.gradient)
            # print("activation_grad: ",node.activation_gradient)
            
    # 가중치 갱신
    def update_weight(self,prev_layer_val,what_layer):
        for i,node in enumerate(self.nodes):
            for j,weight in enumerate(node.weight):
                # print("------------------------------------")
                # print(weight)
                weight-=self.learning_rate*node.gradient*prev_layer_val[j]
                self.nodes[i].weight[j]=weight
                # print(f"* {what_layer} result : ",self.learning_rate*node.gradient*prev_layer_val[j])
                # print(f"* {what_layer} : ", node.gradient, " ", prev_layer_val[j])
                # print(weight)
   