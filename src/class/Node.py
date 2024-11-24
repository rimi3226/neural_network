import numpy as np

class Node:
    def __init__(self,activation):
        self.val=0                  # 노드 값
        self.gradient=0             # 해당 노드의 그레디언트값 
        self.weight=[]              # 다음 레이어에 할당된 가중치 (1*n, n=다음 레이어 노드 수)          
        self.activation_gradient=0  # 활성화함수 미분값
        self.activation=activation  # 활성화함수 종류
        self.bias=[]                # 편향
        
    # 가중치 초기화
    def init_weight(self,next_layer_size,weight_init="He"):
        # 1. 초기값 설정
        if weight_init.lower() in ('he'):
            scale = np.sqrt(2.0 / next_layer_size) 
        elif weight_init.lower() in ('xavier'):
            scale = np.sqrt(1.0 / next_layer_size) 
        else:
            scale = float(weight_init) 
        
        # 2. 가중치 및 편향 초기화
        self.weight = scale * np.random.randn(next_layer_size)
        self.bias = 0        
