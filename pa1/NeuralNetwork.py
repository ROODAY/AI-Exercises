import numpy as np
import os, sys
import math 

def softmax(X):
  """Compute softmax values for each sets of scores in x."""
  e_x = np.exp(X - np.max(X))
  return e_x / e_x.sum()

def sigmoid(X):
  return 1.0/(1.0 + np.exp(-X))

class NeuralNetwork:
  def __init__(self, HNodes, ONodes, activate, deltaActivate):
    self.HNodes = HNodes # the number of nodes in the hidden layer
    self.ONodes = ONodes # the number of nodes in the output layer
    self.activate = activate # a function used to activate
    self.deltaActivate = deltaActivate # the derivative of activate

  def test(self, X):
    # Layer 1 Input = [a, b, ..., BIAS], shape = (X.shape[1] + 1,)
    # Layer 1 Weights should be shape (HNodes, X.shape[1] + 1) to output shape (HNodes,)
    self.W1 = np.random.rand(self.HNodes, X.shape[1] + 1)

    # Layer 2 Input = [...HNodes, BIAS], shape = (HNodes + 1,)
    # Layer 2 Weights should be shape (ONodes, HNodes + 1) to output shape (ONodes,)
    self.W2 = np.random.rand(self.ONodes, self.HNodes + 1)

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
    self.W1 = np.random.rand(self.HNodes, X.shape[1] + 1)

    # Layer 2 Input = [...HNodes, BIAS], shape = (HNodes + 1,)
    # Layer 2 Weights should be shape (ONodes, HNodes + 1) to output shape (ONodes,)
    self.W2 = np.random.rand(self.ONodes, self.HNodes + 1)
    
    for e in range(epochs):
      for i in range(len(X)):
        out = self.forward(X)
        self.backpropogate(X, Y, out)  
    
    # For each epoch, do
        # For each training sample (X[i], Y[i]), do
            # 1. Forward propagate once. Use the function "forward" here!
    
            
            # 2. Backward progate once. Use the function "backpropagate" here!
            
            
    pass

  def predict(self, X):
    """
    Predicts the labels for each sample in X.
    Parameters
    X : numpy matrix
        The matrix containing sample features for testing.
    Returns
    -------
    YPredict : int
        The predictions of X.
    ----------
    """
    outputs = self.forward(X)
    print("outputs", outputs)
    YPredict = outputs.argmax()
    return YPredict

  def forward(self, X):
    # Perform matrix multiplication and activation twice (one for each layer).
    # (hint: add a bias term before multiplication)
    X_biased = np.hstack((X, 1))
    l1_output = self.activate(np.dot(self.W1, X_biased))
    l1_biased = np.hstack((l1_output, 1))
    output_layer_output = softmax(np.dot(self.W2, l1_biased)) #self.activate(np.dot(self.W2, l1_biased))
    return output_layer_output
      
  def backpropagate(self):
    # Compute loss / cost using the getCost function.
    loss = YTrue - YPredict # cost 
    d_output  = loss*self.deltaActivate(YPredict)
    # Compute gradient for each layer.
    g_loss = d_output.dot(self.W2.T) 
    d_loss = g_loss*self.deltaActivate(self.activate(self.np.dot(X, self.W1)))
        
    # Update weight matrices.

    self.W1 += X.T.dot(d_loss)
    self.W2 += d_loss.T.dot(d_output) 
                 
    pass
      
  def getCost(self, YTrue, YPredict):
    # Compute loss / cost in terms of crossentropy.
    # (hint: your regularization term should appear here)
    if YPredict == 1:
        return -math.log(YTrue)
    else:
        return -math.log(1 - YPredict)

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
  print('X.shape: {}'.format(X.shape))
  print('Y.shape: {}'.format(Y.shape))
  
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
  plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr)
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
  model = NeuralNetwork(args[0], args[1], args[2])
  
  # 2. Train the model with the function "fit".
  model.fit(XTrain, YTrain, learningRate, epochs, regLambda)
  plotDecisionBoundary(XTrain, YTrain)
  
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

X, Y = getData('Data/dataset1/LinearX.csv', 'Data/dataset1/LinearY.csv')
splits = splitData(X, Y)

model = NeuralNetwork(5, 2, sigmoid, 1)
model.test(X)
pre = model.predict(X[0])
print(pre)
