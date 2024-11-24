class Node:
    def __init__(self,activation):
        self.val=0                  # 노드 값
        self.gradient=0             # 해당 노드의 그레디언트값 
        self.weight=[]              # 다음 레이어에 할당된 가중치 (1*n, n=다음 레이어 노드 수)          
        self.activation_gradient=0  # 활성화함수 미분값
        self.activation=activation  # 활성화함수 종류
        self.bias=[]                # 편향
        
        
