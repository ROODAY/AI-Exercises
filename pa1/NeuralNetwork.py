import numpy as np
import os, sys
import math 
import matplotlib.pyplot as plt
from tqdm import tqdm

def softmax(X):
  e_x = np.exp(X - np.max(X))
  return e_x / e_x.sum()

def delta_softmax(X):
  f = softmax(X)
  s = f.reshape(-1,1)
  return np.diagflat(s) - np.dot(s, s.T)

def sigmoid(X):
  return 1.0/(1.0 + np.exp(-X))

def delta_sigmoid(X):
  f = sigmoid(X)
  return f * (1 - f)

def tanh(X):
  return np.tanh(X)

def delta_tanh(X):
  return 1 - np.square(tanh(X))

class NeuralNetwork:
  def __init__(self, HNodes, ONodes, activate, deltaActivate):
    self.HNodes = HNodes # the number of nodes in the hidden layer
    self.ONodes = ONodes # the number of nodes in the output layer
    self.activate = activate # a function used to activate
    self.deltaActivate = deltaActivate # the derivative of activate

  def initWeights(self, X):
    # Layer 1 Input = [a, b, ..., BIAS], shape = (X.shape[1] + 1,)
    # Layer 1 Weights should be shape (HNodes, X.shape[1] + 1) to output shape (HNodes,)
    self.W1 = np.random.rand(X.shape[1] + 1, self.HNodes)

    # Layer 2 Input = [...HNodes, BIAS], shape = (HNodes + 1,)
    # Layer 2 Weights should be shape (ONodes, HNodes + 1) to output shape (ONodes,)
    self.W2 = np.random.rand(self.HNodes + 1, self.ONodes)

  def fit(self, X, Y, learningRate, epochs, regLambda):
    """
    This function is used to train the model.
    Parameters
    ----------
    X : numpy matrix
        The matrix containing sample features for training.
    Y : numpy array
        The array containing sample labels for training.
    Returns
    -------
    None
    """

    self.initWeights(X)
    
    for e in tqdm(range(epochs), desc='Epochs'):
      for i in range(len(X)): #for f, b in zip(foo, bar):
        YPredict = self.forward(X[i])
        self.backpropagate(X[i], Y[i], YPredict, learningRate)

  def predict(self, X):
    """
    Predicts the labels for each sample in X.
    Parameters
    X : numpy matrix
        The matrix containing sample features for testing.
    Returns
    -------
    YPredict : numpy matrix
        The predictions of X.
    ----------
    """
    return np.array([self.forward(sample).argmax() for sample in tqdm(X, desc="Predict")])

  def forward(self, X):
    # Perform matrix multiplication and activation twice (one for each layer).
    # (hint: add a bias term before multiplication)
    X_biased = np.hstack((X, 1))
    self.X1 = X_biased
    self.Z1 = np.dot(self.W1.T, X_biased)
    l1_output = self.activate(self.Z1)
    self.A1 = l1_output

    l1_biased = np.hstack((l1_output, 1))
    self.X2 = l1_biased
    self.Z2 = np.dot(self.W2.T, l1_biased)
    output_layer_output = softmax(self.Z2)
    self.A2 = output_layer_output

    return output_layer_output
      
  def backpropagate(self, X, YTrue, YPredict, learningRate):
    # https://dev.to/shamdasani/build-a-flexible-neural-network-with-backpropagation-in-python

    Y = np.zeros(self.ONodes)
    Y[int(YTrue)] = 1

    # calculate change for W2
    dCost_dOutput = YPredict - Y
    dOutput_dZ2 = delta_softmax(self.Z2)
    dCost_dZ2 = np.dot(dCost_dOutput, dOutput_dZ2)[:, np.newaxis]
    dZ2_dW2 = self.X2[:, np.newaxis]
    dCost_dW2 = np.dot(dCost_dZ2, dZ2_dW2.T)

    # [a1|1] * W2 = Z2
    # cost(Z2) = ...
    # dcost / dz2 = dCost_dZ2

    # calculate change for W1
    #print('\n\nW2 shape: {}, dCost_dZ2 shape: {}'.format(self.W2.shape, dCost_dZ2.shape))
    dCost_dA1 = sum(np.dot(self.W2, dCost_dZ2))
    #print('\n\ndCost_dA1 shape: {}'.format(dCost_dA1.shape))
    dA1_dZ1 = self.deltaActivate(self.Z1)[:, np.newaxis]
    #print('dA1_dZ1 shape: {}'.format(dA1_dZ1.shape))
    dCost_dZ1 = np.dot(dCost_dA1, dA1_dZ1.T)[:, np.newaxis]
    #print('dCost_dZ1 shape: {}'.format(dCost_dZ1.shape))
    dZ1_dW1 = self.X1[:, np.newaxis]
    #print('dZ1_dW1 shape: {}'.format(dZ1_dW1.shape))
    dCost_dW1 = np.dot(dCost_dZ1, dZ1_dW1.T)
    #print('dCost_dW1 shape: {}'.format(dCost_dW1.shape))

    self.W1 = self.W1 - dCost_dW1.T * learningRate
    self.W2 = self.W2 - dCost_dW2.T * learningRate
    return
    
      
  def getCost(self, YTrue, YPredict, regLambda):
    # Compute loss / cost in terms of crossentropy.
    # (hint: your regularization term should appear here)
    # https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
    return -1 * math.log(YPredict[int(YTrue)]) + (regLambda / (2 * self.ONodes)) * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))

def getData(XPath, YPath):
  '''
  Returns
  -------
  X : numpy matrix
      Input data samples.
  Y : numpy array
      Input data labels.
  '''

  X = np.genfromtxt(XPath, delimiter=',')
  Y = np.genfromtxt(YPath, delimiter=',')
  #print('X.shape: {}'.format(X.shape))
  #print('Y.shape: {}'.format(Y.shape))
  
  return X, Y

def splitData(X, Y, K = 5):
  '''
  Returns
  -------
  result : List[[train, test]]
      "train" is a list of indices corresponding to the training samples in the data.
      "test" is a list of indices corresponding to the testing samples in the data.
      For example, if the first list in the result is [[0, 1, 2, 3], [4]], then the 4th
      sample in the data is used for testing while the 0th, 1st, 2nd, and 3rd samples
      are for training.
  '''
  # Get a list of K numpy arrays, where each array contains unique indices
  folds = np.array_split([i for i in range(len(Y))], K)

  splits = []
  for i in range(len(folds)):
    test = folds[i].tolist() # Isolate one fold as test set
    train = np.concatenate(folds[:i] + folds[i+1:]).tolist() # Concatenate other folds as train set
    np.random.shuffle(train) # Shuffle train set
    splits.append([train, test])

  return splits

def plotDecisionBoundary(model, X, Y):
  """
  Plot the decision boundary given by model.
  Parameters
  ----------
  model : model, whose parameters are used to plot the decision boundary.
  X : input data
  Y : input labels
  """
  x1_array, x2_array = np.meshgrid(np.arange(-4, 4, 0.01), np.arange(-4, 4, 0.01))
  grid_coordinates = np.c_[x1_array.ravel(), x2_array.ravel()]
  Z = model.predict(grid_coordinates)
  Z = Z.reshape(x1_array.shape)
  plt.contourf(x1_array, x2_array, Z, cmap=plt.cm.bwr)
  plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.bwr)
  plt.show()

def train(XTrain, YTrain, args):
  """
  This function is used for the training phase.
  Parameters
  ----------
  XTrain : numpy matrix
      The matrix containing samples features (not indices) for training.
  YTrain : numpy array
      The array containing labels for training.
  args : List
      The list of parameters to set up the NN model.
  Returns
  -------
  NN : NeuralNetwork object
      This should be the trained NN object.
  """

  # 1. Initializes a network object with given args.
  HNodes, ONodes, activate, deltaActivate, learningRate, epochs, regLambda = args
  model = NeuralNetwork(HNodes, ONodes, activate, deltaActivate)
  
  # 2. Train the model with the function "fit".
  print("Train Model")
  model.fit(XTrain, YTrain, learningRate, epochs, regLambda)
  #print("Plot decision boundary")
  #plotDecisionBoundary(model, XTrain, YTrain)
  
  # 3. Return the model.
  return model

def test(XTest, model):
  """
  This function is used for the testing phase.
  Parameters
  ----------
  XTest : numpy matrix
      The matrix containing samples features (not indices) for testing.
  model : NeuralNetwork object
      This should be a trained NN model.
  Returns
  -------
  YPredict : numpy array
      The predictions of X.
  """
  return model.predict(XTest)

def getConfusionMatrix(YTrue, YPredict):
  """
  Computes the confusion matrix.
  Parameters
  ----------
  YTrue : numpy array
      This array contains the ground truth.
  YPredict : numpy array
      This array contains the predictions.
  Returns
  CM : numpy matrix
      The confusion matrix.
  """
  TP, FP, FN, TN = 0, 0, 0, 0
  for i in range(len(YTrue)):
      if YTrue[i] == YPredict[i]:
          if YTrue[i] == 1:
              TP+= 1
          else:
              TN+= 1
      else:
          if YTrue[i] == 1:
              FP+= 1
          else:
              FN+= 1
  CM = np.matrix([[TN, FN], [FP, TP]])
  
  return CM
    
def getPerformanceScores(YTrue, YPredict):
  """
  Computes the accuracy, precision, recall, f1 score.
  Parameters
  ----------
  YTrue : numpy array
      This array contains the ground truth.
  YPredict : numpy array
      This array contains the predictions.
  Returns
  {"CM" : numpy matrix,
  "accuracy" : float,
  "precision" : float,
  "recall" : float,
  "f1" : float}
      This should be a dictionary.
  """
  cm = getConfusionMatrix(YTrue, YPredict) 
  d = {} 
  TP = cm[0][0]
  TN = cm[1][1]
  FP = cm[0][1]
  FN = cm[1][0] 
  Precision = float (TP /(TP+FP)) 
  Recall = float(TP/(TP+FN))
  Accuracy = float((TP +TN) / (TP + TN + FP + FN))
  F1 = (2*Recall*Precision)/ (Recall + Precision) 
  d["CM"] = cm 
  d["accuracy"] = Accuracy 
  d["precision"] = Precision 
  d["recall"] = Recall 
  d["f1"] = F1
  return d

'''print("Linear Data Tests")
X, Y = getData('Data/dataset1/LinearX.csv', 'Data/dataset1/LinearY.csv')
splits = splitData(X, Y)

HNodes = 5
ONodes = 2
activate = sigmoid
deltaActivate = delta_sigmoid
learningRate = 1
epochs = 50
regLambda = 1
args = (HNodes, ONodes, activate, deltaActivate, learningRate, epochs, regLambda)

for i, split in enumerate(splits):
  print('Beginning Split {}'.format(i+1))
  train_set = split[0]
  XTrain = np.array([X[index] for index in train_set])
  YTrain = np.array([Y[index] for index in train_set])
  model = train(XTrain, YTrain, args)

  test_set = split[1]
  XTest = np.array([X[index] for index in test_set])
  YTest = np.array([Y[index] for index in test_set])
  predicts = test(XTest, model)
  accuracy = sum([1 for i in range(len(predicts)) if predicts[i] == YTest[i]]) / len(predicts)
  print('Split {} Accuracy: {}\n'.format(i+1, accuracy))

print("Nonlinear Data Tests")
X, Y = getData('Data/dataset1/NonlinearX.csv', 'Data/dataset1/NonlinearY.csv')
splits = splitData(X, Y)

HNodes = 5
ONodes = 2
activate = sigmoid
deltaActivate = delta_sigmoid
learningRate = 1
epochs = 50
regLambda = 1
args = (HNodes, ONodes, activate, deltaActivate, learningRate, epochs, regLambda)

for i, split in enumerate(splits):
  print('Beginning Split {}'.format(i+1))
  train_set = split[0]
  XTrain = np.array([X[index] for index in train_set])
  YTrain = np.array([Y[index] for index in train_set])
  model = train(XTrain, YTrain, args)

  test_set = split[1]
  XTest = np.array([X[index] for index in test_set])
  YTest = np.array([Y[index] for index in test_set])
  predicts = test(XTest, model)
  accuracy = sum([1 for i in range(len(predicts)) if predicts[i] == YTest[i]]) / len(predicts)
  print('Split {} Accuracy: {}\n'.format(i+1, accuracy))
'''
print("Digit Data Tests")
XTrain, YTrain = getData('Data/dataset2/Digit_X_train.csv', 'Data/dataset2/Digit_y_train.csv')
XTest, YTest = getData('Data/dataset2/Digit_X_test.csv', 'Data/dataset2/Digit_y_test.csv')

HNodes = 7
ONodes = 10
activate = sigmoid
deltaActivate = delta_sigmoid
learningRate = 1.25
epochs = 50
regLambda = 1
args = (HNodes, ONodes, activate, deltaActivate, learningRate, epochs, regLambda)

model = train(XTrain, YTrain, args)
predicts = test(XTest, model)
accuracy = sum([1 for i in range(len(predicts)) if predicts[i] == YTest[i]]) / len(predicts)
print('Accuracy: {}\n'.format(accuracy))