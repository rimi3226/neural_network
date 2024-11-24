import os
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils")))

import activationFunc as af
import Node 

class Layer:
    def __init__(self,layer_size,activation):
        self.layer_size=layer_size                      # 레이어 크기
        self.nodes=[Node() for _ in range(layer_size)]   # 레이어에 할당된 노드 리스트
        
    # 노드 값 할당 -입력층
    def setLayer(self):
        print()
    
    # 순전파 계산
    def forward(self):
        print()
        
    # 역전파 계산
    def backward(self):
        print()
        