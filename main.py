"""
@where:    EECS 545 - Machine Learning (at Umich)
@what:     Final Project
@title:    Replication of Conditional Neural Processes for
           Regression with Synthetic and Dow Jones data
@authors:  Mengyao Huang, Fabricio Vasselai, Xiaoxue Xin,
           Jiayi Zhang (alphabetical order by surname)
@contact:  vasselai@umich.edu
"""

###############################################################################
# Loads necessary external libraries
###############################################################################
import tensorflow as tf
import numpy as np
import numpy.random as rng
from cnp import *
from plotting import *   
from financialdata import *


###############################################################################
# Defines hyperparameters
###############################################################################
maxIterations = 100000
minContextPoints = 3
maxContextPoints = 10
nSampledFuncs = 10
testInterval = maxIterations/10

# Number of nodes in each layer of the encoder and of the decoder
encoderLayersSizes = [128, 128, 128] #paper says 3 layers, last with 128 nodes so we chose same number for others
decoderLayersSizes = [128, 128, 64, 32, 2]  #paper says 5 layers, last with 2 nodes so we chose same number for others


###############################################################################
# Initializes the Tensorflow dataflow
###############################################################################
tf.reset_default_graph()


###############################################################################
# Loads financial data
###############################################################################
fileDir = 'D:/Michigan/Classes/5th Semester/EECS 545/Final Project/code/real/data/'
fileName = 'DJIprices_std.csv'
trainXc, trainYc, trainXt, trainYt, trainNc, testXc, \
  testYc, testXt, testYt, testNc \
  = getFinancialData(fileDir + fileName, nSampledFuncs, 241, minContextPoints, maxContextPoints)


###############################################################################
# Defines the CNP regression problem
###############################################################################

# Defines the training stage:
trainMu, trainSigma = cnp(trainXc, trainYc, trainNc, trainXt, encoderLayersSizes, decoderLayersSizes)
gaussian = tf.contrib.distributions.MultivariateNormalDiag(loc=trainMu, scale_diag=trainSigma)
loss = -tf.reduce_mean(gaussian.log_prob(trainYt))
training = tf.train.AdamOptimizer().minimize(loss)

# Defines the test stage:
testMu, testSigma = cnp(testXc, testYc, testNc, testXt, encoderLayersSizes, decoderLayersSizes)


###############################################################################
# Runs and plots the CNP analysis
###############################################################################
sess = tf.Session()
sess.run(tf.initialize_all_variables()) #initializes tensorflow session

#main loop over many iterations:
for iteration in range(maxIterations):
    #train phase happens every iteration:
    sess.run(training)
    #test phase can help just in some chonse intervals:
    if iteration % testInterval == 0:
        yHat, var, currentLoss, Xc, Yc, Xt, Yt = sess.run(
          [testMu, testSigma, loss, testXc, testYc, testXt, testYt])
        print('Iteration: {}, loss: {}'.format(iteration, currentLoss))
        # Plot the prediction and the context
        reg_plotting(iteration, Xt, Yt, Xc, Yc, yHat, var)
      
