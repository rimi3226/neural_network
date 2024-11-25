import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils")))

import activationFunc as af
from Node import Node
from Layer import Layer

class NeuralNetwork:
    def __init__(self, input_layer_size, hidden_layers_size, output_layer_size, learning_rate=0.01, activation="relu", output_activation="softmax", weight_init="He"):
        self.layers = []
        self.learning_rate = learning_rate
        self.activation = activation
        self.output_activation = output_activation
        self.target_layer = []
        self.weight_init = weight_init
        self.output_layer_size=output_layer_size
        # 레이어 생성
        # (1) 입력층 생성
        input_layer = Layer(input_layer_size, activation, hidden_layers_size[0], learning_rate)
        input_layer.set_weight(weight_init)
        self.layers.append(input_layer)

        # (2) 은닉층 생성
        for i in range(len(hidden_layers_size)):
            next_layer_size = hidden_layers_size[i + 1] if i + 1 < len(hidden_layers_size) else output_layer_size
            hidden_layer = Layer(hidden_layers_size[i], activation, next_layer_size, learning_rate)
            hidden_layer.set_weight(weight_init)
            self.layers.append(hidden_layer)

        # (3) 출력층 생성
        output_layer = Layer(output_layer_size, output_activation, 0, learning_rate)
        self.layers.append(output_layer)


    # 입력층 노드에 값 설정
    def set_input_layer(self, data):
        self.layers[0].set_layer(data)

    def set_target_layer(self, data):           
        target = np.zeros(self.output_layer_size)
        target[int(data)] = 1
        self.target_layer = target

    # 순전파
    def forward_propagation(self):
        # (1) 입력층 -> 은닉층
        for i in range(0, len(self.layers)-2):
            activation_val, gradient = self.layers[i].forward(what_layer=i)
            self.layers[i+1].set_layer(activation_val)
            self.layers[i+1].set_activation_gradient(gradient)
            
        # (2) 은닉층 -> 출력층
        activation_val, gradient = self.layers[i].forward(5,activation="softmax")
        self.layers[-1].set_layer(activation_val)
        self.layers[-1].set_activation_gradient(gradient)

    # 역전파 
    def backward_propagation(self):
         # 1. 출력층에서의 역전파 계산
        self.layers[-1].get_gradient_output(self.target_layer)
        # 2. 출력층 이전의 모든 레이어에 대해 역전파 계산 및 가중치 업데이트
        for i in range(len(self.layers) - 2, 0, -1):
            # 다음 레이어 노드 값 가져오기
            next_layer_grads = [node.gradient for node in self.layers[i + 1].nodes]
            self.layers[i].get_gradient(next_layer_grads)
            
        # self.print_layer_details()
        # 3. 가중치 업데이트하기
        for i in range(1,len(self.layers)):
            prev_layer_vals = [node.val for node in self.layers[i-1].nodes]
            self.layers[i].update_weight(prev_layer_vals,i)
        # self.print_layer_details()

                
    # 가중치 저장
    def store_weight(self, file_name="model_weights.csv"):
        weights_data = []
        for i, layer in enumerate(self.layers):
            for node in layer.nodes:
                weights_data.append({
                    "Layer": i,
                    "Node": layer.nodes.index(node),
                    "Weights": node.weight
                })
        df = pd.DataFrame(weights_data)
        df.to_csv(file_name, index=False)

    # 레이어 정보 출력
    def print_layer_details(self):
        print(f"\n========= PRINT =========")

        for layer_idx, layer in enumerate(self.layers):
            print(f"\n=== Layer {layer_idx + 1} ===")
            for node_idx, node in enumerate(layer.nodes):
                print(f"Node {node_idx + 1}:")
                print(f"  Value: {node.val}")
                print(f"  Weights: {node.weight}")
                # print(f"  Gradient: {node.gradient}")
                # print(f"  Activation Gradient: {node.activation_gradient}")
        print(f"\n==========================")
        
    # NeuralNetwork 클래스에 추가할 plot_output 메서드
    def plot_output(self):
        # 출력층 노드 값 가져오기
        output = np.array([node.val for node in self.layers[-1].nodes])
        
        # 그래프 그리기
        plt.figure(figsize=(10, 6))
        plt.plot(output, marker='o', linestyle='-', label="Output Values")
        plt.title("Output Layer Node Values")
        plt.xlabel("Node Index")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend()
        plt.show()
 