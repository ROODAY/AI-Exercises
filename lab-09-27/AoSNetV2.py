import numpy as np
import os, sys

class NeuralNetwork:
    def __init__(self, Ws, activate, deltaActivate, Ts):
        self.Ws = Ws
        self.activate = activate
        self.deltaActivate = deltaActivate
        self.Ts = Ts
    
    def fit(self, X, Y, learningRate = 0.001):
        pass

    def predict(self, X):
        outputs = self.forward(X)
        print("outputs", outputs)
        YPredict = outputs.argmax(axis = 1)
        return YPredict

    def forward(self, X):
        print('first activate: {} x {}'.format(X, self.Ws[0]))
        outputP = self.activate(X * self.Ws[0], self.Ts[0])
        print('second activate: {} x {}'.format(outputP, self.Ws[1]))
        outputM = self.activate(outputP * self.Ws[1], self.Ts[1])
        return outputM
        
    def backpropagate(self):
        pass
        
    def getCost(self, YTrue, YPredict):
        pass
        
def threshold(S, T):
    result = np.zeros(S.shape)
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if S[i, j] >= T[j]:
                result[i, j] = 1
    return result

def getResult(result):
    if outputM[0]:
        return "Acquaintances"
    elif outputM[1]:
        return "Siblings"

def getWeightedSum(X, W):
    S = 0
    for i in range(len(X)):
        S += X[i] * W[i]
    return S

# encode each individual
M = {"Robert" : 0, "Rachel" : 1, "Romeo" : 2,      # family R
        "Joan" : 3, "James" : 4, "Juliet" : 5}     # family J
classes = ["Acquaintances", "Siblings"]

W1 = np.matrix([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]]).T
W2 = np.matrix([[1, 1], [-1, -1]]).T
Ws = [W1, W2]

print(Ws[0])
print('\n\n\n')

T1 = [0.5, 0.5]
T2 = [1.5, -1.5]
Ts = [T1, T2]

X = [[0, 1, 0, 0, 0, 1],
     [0, 1, 1, 0, 0, 0],
     [0, 0, 1, 1, 1, 1],
     [1, 1, 1, 0, 0, 0]]

NN = NeuralNetwork(Ws, threshold, None, Ts)
NN.fit(X, None)
YPredict = NN.predict(X)
results = [classes[y] for y in YPredict]
print(results)
