from sys import maxsize
import numpy as np

def threshold(S, T):
    result = np.zeros(S.shape)
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if S[i, j] >= T[j]:
                result[i, j] = 1
    return result

def argmax(L):
    index, M = 0, -maxsize
    for i in range(len(L)):
        if L[i] > M:
            M = L[i]
            index = i
    return index

def getWeightedSum(X, W):
    S = 0
    for i in range(len(X)):
        S += X[i] * W[i]
    return S

# encode each individual
M = {"Robert" : 0, "Rachel" : 1, "Romeo" : 2,      # family R
        "Joan" : 3, "James" : 4, "Juliet" : 5}     # family J
classes = ["Acquaintances", "Siblings"]
x = np.array([1, 1, 0, 0, 0, 0])

W1 = np.matrix([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]]).T
S = x * W1
print(S)
T1 = [0.5, 0.5]
outputP = threshold(S, T1)
print(outputP)

'''
output1 = threshold(getWeightedSum(outputP, [1, 1]), 1.5) # the first magenta node
output2 = threshold(getWeightedSum(outputP, [-1, -1]), -1.5) # the second magenta node
outputM = [output1, output2]

result = argmax(outputM)
print(classes[result])
'''