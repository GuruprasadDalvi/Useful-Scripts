import numpy as np

#Constants 
TANH="tanh"
SIGMOID = "sigmoid"



#Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanH(x):
    return np.tanh(x)


def d_tanH(x):
    return 1 - (np.tanh(x) ** 2)

def activate(inputs,name):
    if name == SIGMOID:
        return sigmoid(inputs)
    elif name == TANH:
        return tanH(inputs)

def d_actvate(inputs,name):
    if name == SIGMOID:
        return d_sigmoid(inputs)
    elif name == TANH:
        return d_tanH(inputs)



class Layer:
    def __init__(self):
        self.input_nodes = None
        self.output_nodes = None
        self.inputs = None
        self.outputs = None

    def forword(self, inputs):
        pass

    def backword(self, output_gradient, learning_rate):
        pass


class Dense(Layer):
    def __init__(self, input_nodes, output_nodes, activation_function=SIGMOID):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.bias = np.random.randn(output_nodes, 1)
        self.weights = np.random.randn(output_nodes, input_nodes)
        self.activation_function = activation_function

    def forword(self, inputs):
        self.inputs = inputs
        self.activations = activate(np.dot(self.weights, self.inputs) + self.bias,self.activation_function)
        return self.activations

    def backword(self, output_gradient, learning_rate):
        output_gradient = np.multiply(output_gradient, d_actvate(self.activations,self.activation_function))
        weights_gradient = np.dot(output_gradient, self.inputs.T)

        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        
        return np.dot(self.weights.T, output_gradient)

    def to_json(self):
        dic = {"activationFunction":self.activation_function,
               "output_nodes":self.output_nodes,
               "input_nodes":self.input_nodes,
               "weights":self.weights.tolist(),
               "bias":self.bias.tolist()}
        return dic 

d = Dense(4,5)
print(d.to_json())