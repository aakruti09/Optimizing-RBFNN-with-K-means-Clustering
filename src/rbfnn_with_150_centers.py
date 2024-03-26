from math import *
import numpy as np

class Radial_Bias_Function:
    
# Constructor used to intialize the variable sigma and number of center points selected. 
  def __init__(self, sigma, numberOfCenterPoints):
    self.sigma = sigma
    self.centerList = None
    self.numberOfCenterPoints = numberOfCenterPoints
    self.Weights = None

# Calculating the RBF Gaussian Kernel function
  def gaussianKernel(self, x, vi):
    numerator = (np.linalg.norm(x-vi)**2)
    G_i = exp((- numerator)/2 * pow(self.sigma, 2))
    return G_i

# Defining the activation function
  def activationFunction(self, inputVector):
    Vector_G = np.zeros((inputVector.shape[0], self.numberOfCenterPoints), float)
    for centerIndex, center in enumerate(self.centerList):
      for xi, x in enumerate(inputVector):
        Vector_G[xi, centerIndex] = self.gaussianKernel(x, center)
    return Vector_G

# Defining the function used for calculating the output values
  def calculateOutput(self, inputVector):
    Vector_G = self.activationFunction(inputVector)
    O = np.dot(Vector_G, self.Weights)
    return O

# Random selecting the center points from the input data.
  def randomlySelectCenterPoints(self, inputVector):
    rand_poit = np.random.choice(len(inputVector), self.numberOfCenterPoints)
    center_list = inputVector[rand_poit]
    return center_list

# Defining the RBF training function and calculating the weights
  def training(self, inputVector, O):
    self.centerList = self.randomlySelectCenterPoints(inputVector)
    Vector_G = self.activationFunction(inputVector)
    self.Weights = np.dot(np.linalg.pinv(Vector_G), O)

# Calculation of the error using mean square error method
  def errorCalculation(self, true_output, calculate_output):
    dif_vec = []
    for i in range(len(true_output)):
      diff = true_output[i] - calculate_output[i]
      diff_sq = pow(diff, 2)
      dif_vec.append(diff_sq)
    return sum(dif_vec)/len(true_output)
