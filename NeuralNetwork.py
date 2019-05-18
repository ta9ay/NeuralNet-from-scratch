import numpy as np
from itertools import zip_longest

class Neuron:
    def __init__(self):
        self.input = []
        self.weights = ''
        self.bias = ''
        self.output = [] #intermediate value. Used in derivatives
        self.tanoutput = ''

def tanh(x):
    return np.tanh(x)

def tanhDeriv(x):
    return 1.0 - np.tanh(x)**2

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

class NeuralNetwork:
    def __init__(self,inpnum,hnum):
        # initialise the object values, weights, and create the network
        Wx = np.random.normal(size=[inpnum,hnum]) # Input layer - Hidden layer weights
        Wh = np.random.normal(size=[hnum,1]) # Hidden layer - Output Layer weights
        ##      creating the hidden layer       ##
        self.HiddenLayer = {}
        for n in range(hnum): 
            self.HiddenLayer[n] = Neuron()
            self.HiddenLayer[n].weights = Wx[:,n]
            self.HiddenLayer[n].bias = np.random.normal()

        ##       creating the output node       ##
        self.OutputNode = Neuron()
        self.OutputNode.weights = Wh
        self.OutputNode.bias = np.random.normal()
    
    
    def feedforward(self,inputs):
        self.OutputNode.input = []
        for i in range(hnum): # iterate over all hidden layer neurons
            temp = []
            self.HiddenLayer[i].input = inputs
            self.HiddenLayer[i].output = np.dot(self.HiddenLayer[i].input,self.HiddenLayer[i].weights) + self.HiddenLayer[i].bias
            self.HiddenLayer[i].tanoutput = tanh(self.HiddenLayer[i].output)
            for output in self.HiddenLayer[i].tanoutput: #append all the outputs of hidden layer to input of outputnode
                temp.append(output)
            self.OutputNode.input.append(temp)
            
        self.OutputNode.input = np.array(self.OutputNode.input)
        self.OutputNode.output = np.dot(self.OutputNode.input.T, self.OutputNode.weights) + self.OutputNode.bias
        self.OutputNode.tanoutput = tanh(self.OutputNode.output) # Final output

        # print("Final Output Ypred: \n",self.OutputNode.tanoutput)
        return self.OutputNode.tanoutput


    def train(self,X,Ytrue,LR,epochs):
        # update weights and biases
        # LR = 0.1
        # epochs = 100
        for epoch in range(epochs):
            
            for row in range(X.shape[0]):
                cost = Ytrue[row] - self.OutputNode.tanoutput[row]
                dL_dYpred = -2 * cost 
                # Update Wh
                adjWh = []
                for n in range(hnum):
                    dYpred_dWh = self.HiddenLayer[n].tanoutput[row] * tanhDeriv(self.OutputNode.output[row])
                    adjWh.append(dL_dYpred * dYpred_dWh)
                adjWh = np.array(adjWh)
                self.OutputNode.weights -= LR * adjWh
                dYpred_dB = tanhDeriv(self.OutputNode.output[row])
                for bias in dYpred_dB:
                    dYpred_dB = bias
                # Update Bias
                self.OutputNode.bias -= LR * dL_dYpred * dYpred_dB
            
                for bias in self.OutputNode.bias:
                    self.OutputNode.bias = bias
                
                
                
                # Update Wx
                for n in range(hnum):
                    dYpred_dWh = self.HiddenLayer[n].tanoutput[row] * tanh(self.OutputNode.output[row])
                    for row in range(X.shape[0]):
                        for m in range(inpnum):
                            dH_dWx = self.HiddenLayer[n].input[row][m] * tanhDeriv(self.HiddenLayer[n].output[row]) #index 0 because numpy array of ([[x1,x2,x3]])
                            self.HiddenLayer[n].weights[m] -= LR * dL_dYpred * dYpred_dWh * dH_dWx
                            
                        dH_dB = tanhDeriv(self.HiddenLayer[n].output[row])
                        # Update Bias
                        self.HiddenLayer[n].bias -= LR * dL_dYpred * dYpred_dWh * dH_dB
                        
                        for bias in self.HiddenLayer[n].bias:
                            self.HiddenLayer[n].bias = bias
                        
                
            # feedforward again to repeat
            self.feedforward(X)

#sample data
X = np.array([
    [-2,-1,-3],
    [-3,-2,-3],
    [1,2,1],
    [-3,0,-1],
    [2,3,1],
    [3,1,2],
    [4,5,3],
    [5,2,3],
    [-3,-2,-3]

])

Y = np.array([-1,-1,1,-1,1,1,1,1,-1])


inpnum = X.shape[1]
hnum = int(input("Enter number of hidden layer neurons: ".strip()))
lr = float(input("Enter the learning rate: ").strip())
epochs = int(input("Enter number of epochs: ").strip())
print("Input matrix: \n",X)
print("Y true values: \n",Y)

network = NeuralNetwork(inpnum,hnum)
network.feedforward(X)
network.train(X,Y,lr,epochs)



#Sample prediction
output = network.feedforward(np.array([[4,2,3]]))
print("Prediction for the given sample is: ",output)
