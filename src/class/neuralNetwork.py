import sys
import os
import matplotlib.pyplot as plt  # 그래프 시각화를 위해 Matplotlib 사용
import math
import csv
import numpy as np

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
    def __init__(self, input_nodes, hidden_layers, output_nodes, learning_rate, 
                 weight_init="He", activation="ReLU", output_activation="softmax"):

        # 1. 하이퍼 파라미터 설정
        self.learning_rate = learning_rate
        self.activation = activation
        self.output_activation = output_activation

        # 2. 각 레이어 생성
        self.layers = []
        self._create_layers(input_nodes, hidden_layers, output_nodes, weight_init)
        self.target_data=None

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

    # 입력 데이터와 타겟값 지정하고 다른 레이어 노드 초기화하는 함수
    def store_training_data(self, input_data, target_data):
        if len(input_data) != 784:
            raise ValueError("입력 데이터의 크기가 잘못되었습니다. 입력 데이터는 784개의 값을 가져야 합니다.")
        if not isinstance(target_data, (int, float)) or not (0 <= target_data < 154):
            raise ValueError("타겟 데이터는 0 이상 154 미만의 인덱스를 나타내는 float 값이어야 합니다.")

        # 입력 데이터와 타겟 데이터를 각 층의 노드에 저장
        self.layers[0].nodes = [input_data]
        self.target_data = [0] * 154
        self.target_data[int(target_data)] = 1  # 원핫 인코딩 방식으로 타겟값 설정
        
        # 은닉층 노드 초기화
        for layer in self.layers[1:-1]:
            layer.nodes = [0] * len(layer.nodes)
            
        print("===============재설정===========")
        for layer in self.layers:
            print(layer.nodes)
        print("===============재설정===========")

        
    # 순전파
    def forward_propagation(self):
        print("==============순전파=============")
        # 1. 입력층과 은닉층 계산 : 행렬곱 & 활성화함수 
        for i, layer in enumerate(self.layers[:-2]):  # 출력층 제외
            # print(f"Layer {i}: Forward Propagation with activation {self.activation}")
            self.layers[i + 1].nodes = layer.forward(layer.nodes, activation=self.activation)
            print(np.round(layer.nodes,4))
            
        # 2. 출력층 계산
        # print(f"Layer Hidden -> Output : Forward Propagation with activation Softmax")
        self.layers[-1].nodes = self.layers[-2].forward(self.layers[-2].nodes, activation="softmax")
        print(np.round(self.layers[-2].nodes,4))
        
        # 출력층 결과와 타겟값 비교하여 정확도 계산
        predicted_index = np.argmax(self.layers[-1].nodes[0])
        target_index = np.argmax(self.target_data)
        result = True if predicted_index == target_index else False
        
        print("==========정확도 측정용==========")
        print(self.layers[-1].nodes[0])
        print(self.target_data)
        print(predicted_index)
        print(target_index)
        
        return result
        
    # 역전파
    def back_propagation(self, loss_function="MSE"):
        # 1. [ 출력층 - 은닉층 계산 ]
        # 1-1. 출력층의 오차 계산
        output_layer = self.layers[-1]
        output_errors = []

        # self.target_data를 기반으로 오차 계산
        if loss_function == "MSE":
            output_errors = [[(output - target) for output, target in zip(output_layer.nodes[0], self.target_data)]]
        elif loss_function == "cross_entropy":
            output_errors = [[-(target / output) for output, target in zip(output_layer.nodes[0], self.target_data)]]
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")
        
        # 1-2. 출력층 활성화 함수의 미분 적용
        if self.output_activation == "sigmoid":
            output_derivative = [[output * (1 - output) for output in output_layer.nodes[0]]]
        elif self.output_activation == "tanh":
            output_derivative = [[1 - output ** 2 for output in output_layer.nodes[0]]]
        elif self.output_activation == "ReLU":
            output_derivative = [[1 if output > 0 else 0 for output in output_layer.nodes[0]]]
        else:
            output_derivative = [[1 for _ in output_layer.nodes[0]]]  # softmax의 경우에는 따로 미분 적용하지 않음

        # 1-3. 출력층의 그라디언트 계산
        next_errors = [[output_errors[0][j] * output_derivative[0][j] for j in range(len(output_layer.nodes[0]))]]

        # 2. [ 은닉층 - 은닉층 계산 ]
        for layer_idx in reversed(range(1, len(self.layers) - 1)):
            layer = self.layers[layer_idx]
            
            # 2-1. 가중치 업데이트
            for i in range(len(layer.weights)):
                for j in range(len(layer.weights[i])):
                    layer.weights[i][j] -= self.learning_rate * next_errors[0][j] * layer.nodes[0][j]
                    
            # 2-2. 이전 레이어의 오차 계산
            if self.activation == "sigmoid":
                activation_derivative = [[node * (1 - node) for node in layer.nodes[0]]]
            elif self.activation == "tanh":
                activation_derivative = [[1 - node ** 2 for node in layer.nodes[0]]]
            elif self.activation == "ReLU":
                activation_derivative = [[1 if node > 0 else 0 for node in layer.nodes[0]]]
            else:
                activation_derivative = [[1 for _ in layer.nodes[0]]]

            # 2-3. 은닉층의 그라디언트 계산
            next_errors = [[sum(next_errors[0][j] * layer.weights[i][j] * activation_derivative[0][i] for j in range(len(layer.weights[0]))) for i in range(len(layer.weights))]]
            
        # [ 은닉층 - 입력층 계산 ]
        input_layer = self.layers[0] 
        first_hidden_layer = self.layers[1]  

        # 1. 가중치 업데이트
        for i in range(len(input_layer.weights)):
            for j in range(len(input_layer.weights[i])):
                input_layer.weights[i][j] -= self.learning_rate * next_errors[0][j] * first_hidden_layer.nodes[0][j]


    def train(self, inputs, targets, epochs, batch_size, loss_function="MSE"):
        for epoch in range(epochs):
            for i in range(0, len(inputs), batch_size):
                batch_inputs = inputs[i:i + batch_size]
                batch_targets = targets[i:i + batch_size]

                # 순전파 및 역전파
                self.layers[0].nodes = batch_inputs[0]
                outputs = self.forward_propagation()
                self.back_propagation(batch_targets[0], loss_function=loss_function)

    def save_weights_to_csv(self, directory="weights"):
        """
        각 레이어의 가중치를 CSV 파일로 저장

        Args:
            directory (str): 저장할 디렉토리 경로 (기본값은 "weights")
        """
        import os

        # 디렉토리 생성 (없으면 생성)
        if not os.path.exists(directory):
            os.makedirs(directory)

        for i, layer in enumerate(self.layers):
            # 입력층은 가중치가 없으므로 스킵
            if not layer.weights:
                continue

            # 파일 이름: 레이어 번호에 따라 지정
            filename = os.path.join(directory, f"layer_{i}_weights.csv")
            
            # 가중치를 CSV 파일로 저장
            with open(filename, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(layer.weights)

            print(f"Layer {i} weights saved to {filename}")