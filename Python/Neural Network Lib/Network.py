import numpy as np
import json
from Layers import Dense
import logging


def mse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.mean(np.power(y_true-y_pred,2))

def mse_prime(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return 2*(y_pred-y_true)/np.size(y_true)




class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.1):
        self.layers = layers
        self.learning_rate = learning_rate

    def feedforword(self,inputs):
        for layer in self.layers:
           inputs = layer.forword(inputs)
        return inputs

    def backpropagate(self, predicted, expected):
        grad = mse_prime(expected, predicted)
        for layer in reversed(self.layers):
            grad = layer.backword(grad, self.learning_rate)

    def save(self,path="./network.json"):
        dic = {"Layer Count":len(self.layers),
               "Learning Rate":self.learning_rate}
        layer_counter=0
        for layer in self.layers:
            d = layer.to_json()
            dic.update({f"Layer {layer_counter+1}":d})
            layer_counter+=1
        
        op_file=open("network.json","w")
        json.dump(dic,op_file,indent=3)
        op_file.close()
    
    def load(self,path="./network.json"):
        with open(path,'r') as target:
            data= json.load(target)
        layer_counter = data["Layer Count"]
        self.layers = []
        for i in range(layer_counter):
            layer_data = data[f"Layer {i+1}"]
            if layer_data["type"]=="Dense":
                layer = Dense(1,2)
                layer.from_json(layer_data)
            self.layers.append(layer)
        logging.debug("Network Loaded")
        
