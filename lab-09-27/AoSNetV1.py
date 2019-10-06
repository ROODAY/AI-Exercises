from sys import maxsize

def threshold(S, T):
    return S >= T

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
x = [1, 1, 0, 0, 0, 0]

output1 = threshold(getWeightedSum(x[:3], [1] * 3), 0.5) # the first purple node
output2 = threshold(getWeightedSum(x[3:], [1] * 3), 0.5) # the second purple node
outputP = [output1, output2]

output1 = threshold(getWeightedSum(outputP, [1, 1]), 1.5) # the first magenta node
output2 = threshold(getWeightedSum(outputP, [-1, -1]), -1.5) # the second magenta node
outputM = [output1, output2]

result = argmax(outputM)
print(classes[result])