import math
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils")))

import replaceNumpy as rnp
import activationFunc as af 

class Layer:
    def __init__(self, layer_size, weight_init_method="None"):
        """신경망 레이어 초기화

        Args:
            layer_size (int): 레이어 노드 수
            weight_init_method (str): 가중치 초기화 방법 ("Xavier", "He", "std", "None")
        """
        self.layer_size = layer_size
        self.nodes = rnp.zeros(layer_size)  # 0으로 초기화된 노드 생성
        self.biases = rnp.zeros(layer_size)  # 0으로 초기화된 편향 생성
        self.weights = None  # 가중치 초기화
        self.next_layer = None  # 다음 레이어 연결

    def initialize_weights(self, next_layer_size, method="Xavier"):
        """가중치 초기화

        Args:
            next_layer_size (int): 다음 레이어의 노드 개수
            method (str): 초기화 방법 ("Xavier", "He", "std")
        """
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

    def set_next_layer(self, next_layer):
        """다음 레이어 할당

        Args:
            next_layer (Layer): 연결할 다음 레이어
        """
        self.next_layer = next_layer

    def forward(self, input_data, activation="sigmoid"):
        """활성화 값을 계산하여 저장

        Args:
            input_data (list): 입력 데이터
            activation (str): 활성화 함수 ("ReLU", "sigmoid", "tanh")
        """
        # 입력 데이터가 1차원 리스트일 경우 2차원으로 변환
        if isinstance(input_data[0], float):  # 단일 샘플일 경우
            input_data = [input_data]

        # 입력 데이터와 가중치 값 강제 변환
        input_data = [[float(x) for x in row] for row in input_data]
        self.weights = [[float(w) for w in row] for row in self.weights]

        # 가중치와 입력 데이터의 행렬 곱
        z = rnp.multiply_matrix(input_data, self.weights)

        # 편향 추가
        for i in range(len(z)):
            z[i] = [zi + self.biases[j] for j, zi in enumerate(z[i])]
        print("SS")
        af.sigmoid(z)
        print("SS")

        # 활성화 함수 적용
        if activation == "sigmoid":
            return af.sigmoid(z)
        elif activation == "ReLU":
            return af.relu(z)
        elif activation == "tanh":
            return af.tanh(z)
        elif activation == "leaky_relu":
            return af.leaky_relu(z)
        else:
            raise ValueError(f"Invalid activation function: {activation}")