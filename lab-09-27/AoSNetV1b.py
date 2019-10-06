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
X = [[0, 1, 0, 0, 0, 1],
     [0, 1, 1, 0, 0, 0],
     [0, 0, 1, 1, 1, 1],
     [1, 1, 1, 0, 0, 0]]

W1 = np.matrix([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]]).T
S1 = X * W1
T1 = [0.5, 0.5]
outputP = threshold(S1, T1)
# print(outputP)

W2 = np.matrix([[1, 1], [-1, -1]]).T
S2 = outputP * W2
T2 = [1.5, -1.5]
outputM = threshold(S2, T2)
print(outputM)

result = outputM.argmax(axis = 1) # do argmax for each row!
print(result)
print([classes[y] for y in result])