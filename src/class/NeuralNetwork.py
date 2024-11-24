import os
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils")))

import activationFunc as af
import Node 
import Layer

class NeuralNetwork:
    def __init__(self,learning_rate=0.01,activation="relu",output_activation="softmax",weight_init="He"):
        self.layers=[]
        self.learning_rate=learning_rate
        self.activation=activation
        self.output_activation=output_activation
        self.target_nodes=[]
        self.weight_init=weight_init
        
    def setIntputLayer(self,data):
        print()
        
    def setTargetNodes(self,data):
        print()
        
    def forward_propagation(self):
        print()
    
    def backward_propagation(self):
        print()
        
    def store_weight(self):
        print()
        
        