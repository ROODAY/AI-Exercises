import numpy as np
import os, sys

class NeuralNetwork:
  def __init__(self, NNodes, activate, deltaActivate):
    self.NNodes = NNodes # the number of nodes in the hidden layer
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

    # Layer 1 Input = [a, b, ..., BIAS], shape = X.shape[1] + 1,)
    # Layer 1 Weights should be shape (NNodes, X.shape[1] + 1) to output shape (NNodes, 1)
    self.W1 = np.random.rand(self.NNodes, X.shape[1] + 1)

    # Layer 2 Input = [...NNodes, BIAS], shape = (NNodes + 1,)
    # Layer 2 Weights should be shape (1, NNodes + 1) to output shape (1, 1)
    self.W2 = np.random.rand(1, self.NNodes + 1)
    
    
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
    YPredict : numpy array
        The predictions of X.
    ----------
    """
    outputs = self.forward(X)
    YPredict = outputs.argmax(axis = 1)
    return YPredict

  def forward(self, X):
    # Perform matrix multiplication and activation twice (one for each layer).
    # (hint: add a bias term before multiplication)
    hidden_layer_output = self.activate(self.W1 * (X + [1]))
    output_layer_output = self.activate(self.W2 * hidden_layer_output)
    return output_layer_output
      
  def backpropagate(self):
    # Compute loss / cost using the getCost function.
            
            
            
    # Compute gradient for each layer.
    
    
    
    # Update weight matrices.
    pass
      
  def getCost(self, YTrue, YPredict):
    # Compute loss / cost in terms of crossentropy.
    # (hint: your regularization term should appear here)
    pass

def getData(dataDir):
  '''
  Returns
  -------
  X : numpy matrix
      Input data samples.
  Y : numpy array
      Input data labels.
  '''

  X = np.genfromtxt('{}/LinearX.csv'.format(dataDir), delimiter=',')
  Y = np.genfromtxt('{}/LinearY.csv'.format(dataDir), delimiter=',')
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
  
  
  # 2. Train the model with the function "fit".
  # (hint: use the plotDecisionBoundary function to visualize after training)
  
  
  # 3. Return the model.
  
  pass

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
  pass
    
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
  pass

X, Y = getData('Data/dataset1')
splits = splitData(X, Y)