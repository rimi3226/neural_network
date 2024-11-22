import sys
import os
import matplotlib.pyplot as plt  # 그래프 시각화를 위해 Matplotlib 사용
import math


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils")))

import replaceNumpy as rnp
import activationFunc as af 
from Layer import Layer

class neuralNetwork:
    def __init__(self, input_nodes, input_data, hidden_layers, output_nodes, learning_rate, 
                 weight_init="He", activation="ReLU", output_activation="softmax"):
        """
        신경망 초기화

        Args:
            input_nodes (int): 입력층 크기
            hidden_layers (list): 은닉층 크기 ex) [256, 200, 160]
            output_nodes (int): 출력층 크기
            learning_rate (float): 학습률
            weight_init (str): 가중치 초기화 방법 ('Xavier', 'He')
            activation (str): 은닉층 활성화 함수 ('sigmoid', 'tanh', 'ReLU')
            output_activation (str): 출력층 활성화 함수 ('sigmoid', 'softmax')
        """
        self.learning_rate = learning_rate
        self.activation = activation
        self.output_activation = output_activation

        # 레이어 생성
        self.layers = []
        self._create_layers(input_nodes, hidden_layers, output_nodes, weight_init)
        
        # 입력 데이터 설정
        if len(input_data) != input_nodes:
            raise ValueError(f"Input data length ({len(input_data)}) does not match input nodes ({input_nodes}).")
        self.layers[0].nodes = input_data


    def _create_layers(self, input_nodes, hidden_layers, output_nodes, weight_init):
        """신경망 레이어 생성 및 초기화"""
        # 입력층
        input_layer = Layer(input_nodes)
        self.layers.append(input_layer)

        # 은닉층
        prev_layer = input_layer
        for size in hidden_layers:
            hidden_layer = Layer(size)
            prev_layer.set_next_layer(hidden_layer)
            prev_layer.initialize_weights(size, method=weight_init)
            self.layers.append(hidden_layer)
            prev_layer = hidden_layer

        # 출력층
        output_layer = Layer(output_nodes)
        prev_layer.set_next_layer(output_layer)
        prev_layer.initialize_weights(output_nodes, method=weight_init)
        self.layers.append(output_layer)

    def forward_propagation(self):
        """순전파 실행

        Args:
            input_data (list): 입력 데이터
        """
        current_data = self.layers[0].nodes 
    
        for i, layer in enumerate(self.layers[:-1]):  # 출력층 제외
            print(f"Layer {i + 1}: Forward Propagation with activation {self.activation}")
            next_data = layer.forward(current_data, activation=self.activation)
            self.layers[i + 1].nodes = next_data  # 다음 레이어에 전달
            current_data = next_data
            
        # 출력층 처리 전 노드 값 출력
        output_layer = self.layers[-1]
        print(f"Layer {len(self.layers)}: Node values before applying {self.output_activation}")
        print(output_layer.nodes)

        # 선 그래프 그리기
        self.plot_layer_nodes(output_layer.nodes, len(self.layers))
        
        # 출력층 처리
        output_layer = self.layers[-1]
        print(f"Layer {len(self.layers)}: Forward Propagation with activation {self.output_activation}")
        output_layer.nodes=af.softmax(output_layer.nodes)
        
        
        # 인덱스 생성
        x = range(len(output_layer.nodes[0]))
        print(len(output_layer.nodes[0]))
        print(output_layer.nodes)

        self.plot_layer_nodes(output_layer.nodes[0],len(output_layer.nodes[0]))

    def train(self, inputs, targets, epochs, batch_size, loss_function="MSE"):
        """신경망 학습"""
        for epoch in range(epochs):
            # 배치 처리
            for i in range(0, len(inputs), batch_size):
                batch_inputs = inputs[i:i + batch_size]
                batch_targets = targets[i:i + batch_size]

                # 순전파 및 역전파 (구현 필요)
                outputs = self.forward_propagation(batch_inputs)
                loss = self.calculate_loss(outputs, batch_targets, loss_function)
                # 역전파 구현 생략

    def calculate_loss(self, outputs, targets, loss_function):
        """손실 계산"""
        if loss_function == "MSE":
            return sum((o - t) ** 2 for o, t in zip(outputs, targets)) / len(targets)
        elif loss_function == "cross_entropy":
            return -sum(t * math.log(o) + (1 - t) * math.log(1 - o) for o, t in zip(outputs, targets))
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")
    
    def print_network(self):
        """네트워크 출력"""
        for i, layer in enumerate(self.layers):
            print(f"Layer {i + 1}:")
            print(f"  Nodes: {layer.nodes}")
            if layer.weights:
                print(f"  Weights: {layer.weights}")
    
    def print_layer_nodes(self):
        """네트워크의 각 레이어 노드 출력"""
        print("=" * 50)
        print("Neural Network Layer Nodes:")
        for i, layer in enumerate(self.layers):
            print(f"Layer {i + 1}:")
            print(f"  Nodes: {layer.nodes}")
        print("=" * 50)

                 
    def plot_weights(self):
        """각 은닉층의 가중치를 그래프로 시각화"""
        for i, layer in enumerate(self.layers[1:], start=1):  # 입력층 제외
            if layer.weights:
                flattened_weights = [w for row in layer.weights for w in row]  # 2차원 가중치를 1차원으로 변환
                plt.figure(figsize=(8, 4))
                plt.hist(flattened_weights, bins=50, alpha=0.75, color='blue', edgecolor='black')
                plt.title(f"Layer {i}: Weight Distribution")
                plt.xlabel("Weight Value")
                plt.ylabel("Frequency")
                plt.grid(True)
                plt.show()
                
    def plot_layer_nodes(self, nodes, layer_number):
        """
        특정 Layer의 노드 값을 선 그래프로 시각화

        Args:
            nodes (list): 레이어 노드 값
            layer_number (int): 레이어 번호
        """
        # Flatten if 2D array
        if isinstance(nodes[0], list):  # 2D array
            flat_nodes = [node for row in nodes for node in row]
        else:  # 1D array
            flat_nodes = nodes

        # 선 그래프 시각화
        plt.figure(figsize=(10, 6))
        plt.plot(flat_nodes, marker='o', linestyle='-', alpha=0.75)
        plt.title(f"Layer {layer_number}: Node Value Distribution")
        plt.xlabel("Node Index")
        plt.ylabel("Node Value")
        plt.grid(True)
        plt.show()
