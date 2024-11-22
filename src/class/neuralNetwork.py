import sys
import os
import matplotlib.pyplot as plt  # 그래프 시각화를 위해 Matplotlib 사용
import math

# 직접 만든 라이브러리 사용
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils")))

import replaceNumpy as rnp
import activationFunc as af 
import visualization as viz
from Layer import Layer

# --- 신경망 클래스 --- 
# 1. 초기화
# 2. 순전파 (정확도 테스트)
# 3. 역전파 (훈련)
class neuralNetwork:
    
    # 신경망 초기화 
    def __init__(self, input_nodes, input_data, hidden_layers, output_nodes, learning_rate, 
                 weight_init="He", activation="ReLU", output_activation="softmax"):

        # 1. 하이퍼 파라미터 설정
        self.learning_rate = learning_rate
        self.activation = activation
        self.output_activation = output_activation

        # 2. 각 레이어 생성
        self.layers = []
        self._create_layers(input_nodes, hidden_layers, output_nodes, weight_init)
        
        # 2-1. 입력 데이터를 입력 레이어에 할당
        if len(input_data) != input_nodes:
            raise ValueError(f"Input data length ({len(input_data)}) does not match input nodes ({input_nodes}).")
        self.layers[0].nodes = input_data

    # 신경망에 사용될 레이어 생성
    def _create_layers(self, input_nodes, hidden_layers, output_nodes, weight_init):
        # 1. 입력층 생성
        input_layer = Layer(input_nodes)
        self.layers.append(input_layer)

        # 2. 은닉층 생성
        prev_layer = input_layer
        for size in hidden_layers:
            hidden_layer = Layer(size)
            prev_layer.set_next_layer(hidden_layer)
            prev_layer.initialize_weights(size, method=weight_init)
            self.layers.append(hidden_layer)
            prev_layer = hidden_layer

        # 3. 출력층 생성
        output_layer = Layer(output_nodes)
        prev_layer.set_next_layer(output_layer)
        prev_layer.initialize_weights(output_nodes, method=weight_init)
        self.layers.append(output_layer)

    # 순전파
    def forward_propagation(self):
        # 1. current_data에 입력층 할당
        #    current_data : 현재 계산하고 있는 레이어
        current_data = self.layers[0].nodes
    
        # 2. 입력층과 은닉층 계산 : 행렬곱 & 활성화함수 
        for i, layer in enumerate(self.layers[:-2]):  # 출력층 제외
            print(f"Layer {i}: Forward Propagation with activation {self.activation}")
            next_data = layer.forward(current_data, activation=self.activation)
            self.layers[i + 1].nodes = next_data  
            current_data = next_data
            
        # 3. 출력층 계산
        print(f"Layer Hidden -> Output : Forward Propagation with activation Softmax")
        self.layers[-1].nodes = self.layers[-2].forward(self.layers[-2].nodes, activation="softmax")
        

    def calculate_loss(self, outputs, targets, loss_function):
        """손실 계산"""
        if loss_function == "MSE":
            return sum((o - t) ** 2 for o, t in zip(outputs, targets)) / len(targets)
        elif loss_function == "cross_entropy":
            return -sum(t * math.log(o) + (1 - t) * math.log(1 - o) for o, t in zip(outputs, targets))
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")