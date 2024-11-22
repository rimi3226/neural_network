import math
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils")))

import replaceNumpy as rnp
import activationFunc as af 

# 레이어
class Layer:
    # 레이어 초기화
    def __init__(self, layer_size, weight_init_method="None"):
        self.layer_size = layer_size
        self.nodes = None 
        self.weights = None 
        self.next_layer = None 

    # 가중치 초기화
    def initialize_weights(self, next_layer_size, method="Xavier"):
        if method == "std":
            self.weights = rnp.random_matrix(self.layer_size, next_layer_size)
        elif method == "Xavier":
            limit = math.sqrt(1.0 / self.layer_size)
            self.weights = [[rnp.random_matrix(1, 1)[0][0] * limit for _ in range(next_layer_size)] for _ in range(self.layer_size)]
        elif method == "He":
            limit = math.sqrt(2.0 / self.layer_size)
            self.weights = [[rnp.random_matrix(1, 1)[0][0] * limit for _ in range(next_layer_size)] for _ in range(self.layer_size)]
        else:
            raise ValueError(f"Invalid weight initialization method: {method}")

    # 레이어간 연결
    def set_next_layer(self, next_layer):
        self.next_layer = next_layer

    # 순전파 계산
    def forward(self, input_data, activation="sigmoid"):
        # 1. 입력 데이터가 1차원 리스트일 경우 2차원으로 변환
        if isinstance(input_data[0], float): 
            input_data = [input_data]
            
        # print(f"input_data : {len(input_data[0])}")
        # print(f"Layer size : {self.layer_size}")
        # print(f"length of weights : {len(self.weights)}")
        
        # 2. 레이어의 노드와 가중치의 행렬 곱
        z = rnp.multiply_matrix(input_data, self.weights)
      
        # print(f"행렬곱 결과 : {z}")
        
        # 3. 활성화 함수 적용
        if activation == "sigmoid":
            return af.sigmoid(z)
        elif activation == "ReLU":
            return af.relu(z)
        elif activation == "tanh":
            return af.tanh(z)
        elif activation == "leaky_relu":
            return af.leaky_relu(z)
        elif activation == "softmax":
            return af.softmax(z)
        else:
            raise ValueError(f"Invalid activation function: {activation}")