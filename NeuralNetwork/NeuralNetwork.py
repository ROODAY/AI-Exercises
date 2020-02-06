import numpy as np
import os, sys
import math 
import matplotlib.pyplot as plt
from tqdm import tqdm

# Activation function for output layer
def softmax(X):
  e_x = np.exp(X - np.max(X))
  return e_x / e_x.sum()

# Derivative of softmax for backpropagation
def delta_softmax(X):
  f = softmax(X)
  s = f.reshape(-1,1)
  return np.diagflat(s) - np.dot(s, s.T)

# Activation function for hidden layer
def sigmoid(X):
  return 1.0/(1.0 + np.exp(-X))

# Derivative of sigmoid for backpropagation
def delta_sigmoid(X):
  f = sigmoid(X)
  return f * (1 - f)

class NeuralNetwork:
  def __init__(self, HNodes, ONodes, activate, deltaActivate):
    self.HNodes = HNodes # the number of nodes in the hidden layer
    self.ONodes = ONodes # the number of nodes in the output layer
    self.activate = activate # a function used to activate
    self.deltaActivate = deltaActivate # the derivative of activate

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
    
    # Layer 1 Input = [a, b, ..., BIAS], shape = (X.shape[1] + 1,)
    # Layer 1 Weights should be shape (HNodes, X.shape[1] + 1) to output shape (HNodes,)
    self.W1 = np.random.rand(X.shape[1] + 1, self.HNodes)

    # Layer 2 Input = [...HNodes, BIAS], shape = (HNodes + 1,)
    # Layer 2 Weights should be shape (ONodes, HNodes + 1) to output shape (ONodes,)
    self.W2 = np.random.rand(self.HNodes + 1, self.ONodes)
    
    for e in tqdm(range(epochs), desc='Epochs'):
      for x, y in zip(X, Y):
        YPredict = self.forward(x)
        self.backpropagate(x, y, YPredict, learningRate, regLambda)
        #print('Cost:', self.getCost(y, YPredict, regLambda))


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
    return np.array([self.forward(sample).argmax() for sample in tqdm(X, desc="Predicts")])

  def forward(self, X):
    # Perform matrix multiplication and activation twice (one for each layer).

    # Hidden Layer
    self.X1 = np.hstack((X, 1)) # augmentation of input to include bias term
    self.Z1 = np.dot(self.W1.T, self.X1)
    self.A1 = self.activate(self.Z1)

    # Output Layer
    self.X2 = np.hstack((self.A1, 1)) # augmentation of input to include bias term
    self.Z2 = np.dot(self.W2.T, self.X2)
    self.A2 = softmax(self.Z2)

    return self.A2
      
  def backpropagate(self, X, YTrue, YPredict, learningRate, regLambda):
    #https://en.wikipedia.org/wiki/Backpropagation#Finding_the_derivative_of_the_error
    #[:, np.newaxis] converts shape (x,) => (x, 1), used to ensure dot product doesn't error

    Y = np.zeros(self.ONodes)
    Y[int(YTrue)] = 1 # Turn YTrue into one-hot encoded vector

    # calculate change for W2
    dCost_dOutput = YPredict - Y
    dOutput_dZ2 = delta_softmax(self.Z2)
    dCost_dZ2 = np.dot(dCost_dOutput, dOutput_dZ2)[:, np.newaxis]
    dZ2_dW2 = self.X2[:, np.newaxis]
    dCost_dW2 = np.dot(dCost_dZ2, dZ2_dW2.T) + regLambda * self.W2.T # apply regularization term

    # calculate change for W1
    dCost_dA1 = sum(np.dot(self.W2, dCost_dZ2))
    dA1_dZ1 = self.deltaActivate(self.Z1)[:, np.newaxis]
    dCost_dZ1 = np.dot(dCost_dA1, dA1_dZ1.T)[:, np.newaxis]
    dZ1_dW1 = self.X1[:, np.newaxis]
    dCost_dW1 = np.dot(dCost_dZ1, dZ1_dW1.T) + regLambda * self.W1.T # apply regularization term

    # update weights
    self.W1 -= dCost_dW1.T * learningRate
    self.W2 -= dCost_dW2.T * learningRate
      
  def getCost(self, YTrue, YPredict, regLambda):
    # Compute loss / cost in terms of crossentropy.
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

def train(XTrain, YTrain, args, plot=True):
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
  if plot:
    print("Plot decision boundary")
    plotDecisionBoundary(model, XTrain, YTrain)
  
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
  # Note this is considered multi-class, not binary. TP is along diagonal (this generalizes to both datasets)
  CM = np.zeros((len(set(YTrue)), len(set(YTrue))))

  for i in range(len(YTrue)):
    predict = int(YPredict[i])
    actual = int(YTrue[i])
    CM[actual, predict] += 1

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

  diag = np.diag(cm)
  col_sum = np.sum(cm, axis = 0)
  row_sum = np.sum(cm, axis = 1)

  # Calculate Precision, Recall, and Accuracy assuming multiclass classification
  # For dataset1, binary classification results would simply be the Precision and Recall of class 1
  Precision = np.mean(np.divide(diag, col_sum, out=np.zeros_like(diag), where=col_sum!=0))
  Recall = np.mean(np.divide(diag, row_sum, out=np.zeros_like(diag), where=row_sum!=0))
  Accuracy = np.trace(cm) / np.sum(cm)
  F1 = (2*Recall*Precision)/ (Recall + Precision) 

  d["CM"] = cm 
  d["accuracy"] = Accuracy 
  d["precision"] = Precision 
  d["recall"] = Recall 
  d["f1"] = F1
  return d

'''
print("#################")
print("Linear Data Tests")
print("#################")
X, Y = getData('Data/dataset1/LinearX.csv', 'Data/dataset1/LinearY.csv')
splits = splitData(X, Y)

HNodes = 5
ONodes = 2
activate = sigmoid
deltaActivate = delta_sigmoid
learningRate = 1
epochs = 50
regLambda = 0
args = (HNodes, ONodes, activate, deltaActivate, learningRate, epochs, regLambda)

allTests = [] 
allPredicts = [] 
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
  allTests.append(YTest)
  allPredicts.append(predicts)

perform = getPerformanceScores(np.concatenate(allTests), np.concatenate(allPredicts))
print('Confusion Matrix: ')
print(perform['CM'])
print('Accuracy: {}'.format(perform['accuracy']))
print('Precision: {}'.format(perform['precision']))
print('Recall: {}'.format(perform['recall']))
print('F1 Score: {}\n'.format(perform['f1']))


print("####################")
print("Nonlinear Data Tests")
print("####################")
X, Y = getData('Data/dataset1/NonlinearX.csv', 'Data/dataset1/NonlinearY.csv')
splits = splitData(X, Y)

HNodes = 5
ONodes = 2
activate = sigmoid
deltaActivate = delta_sigmoid
learningRate = 1
epochs = 50
regLambda = 0
args = (HNodes, ONodes, activate, deltaActivate, learningRate, epochs, regLambda)

allTests = [] 
allPredicts = [] 
for i, split in enumerate(splits):
  print('Beginning Split {}'.format(i+1))
  ytest = [] 
  ypredict = [] 
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
  allTests.append(YTest)
  allPredicts.append(predicts)
  
perform = getPerformanceScores(np.concatenate(allTests), np.concatenate(allPredicts))
print('Confusion Matrix: ')
print(perform['CM'])
print('Accuracy: {}'.format(perform['accuracy']))
print('Precision: {}'.format(perform['precision']))
print('Recall: {}'.format(perform['recall']))
print('F1 Score: {}\n'.format(perform['f1']))
'''

print("################")
print("Digit Data Tests")
print("################")
XTrain, YTrain = getData('Data/dataset2/Digit_X_train.csv', 'Data/dataset2/Digit_y_train.csv')
XTest, YTest = getData('Data/dataset2/Digit_X_test.csv', 'Data/dataset2/Digit_y_test.csv')

HNodes = 37
ONodes = 10
activate = sigmoid
deltaActivate = delta_sigmoid
learningRate = 0.5
epochs = 500
regLambda = 0.005
args = (HNodes, ONodes, activate, deltaActivate, learningRate, epochs, regLambda)

model = train(XTrain, YTrain, args, False) # don't plot
predicts = test(XTest, model)
accuracy = sum([1 for i in range(len(predicts)) if predicts[i] == YTest[i]]) / len(predicts)
print('Accuracy: {}\n'.format(accuracy))

perform = getPerformanceScores(YTest, predicts)
print('Confusion Matrix: ')
print(perform['CM'])
print('Accuracy: {}'.format(perform['accuracy']))
print('Precision: {}'.format(perform['precision']))
print('Recall: {}'.format(perform['recall']))
print('F1 Score: {}\n'.format(perform['f1']))
