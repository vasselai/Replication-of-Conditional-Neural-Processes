"""
Helper function to load different financial data
"""
import numpy as np
import tensorflow as tf


def getFinancialData(filePath, nSampledFuncs, maxRow, \
                     minContextPoints, maxContextPoints):
    ''' Loads financial data and prepares it to be used in the
    CNP model, by generating train and test context and target
    points for both X and Y '''
    
    #load data:
    realData = np.genfromtxt(filePath, delimiter=',', dtype=np.float32)
    #number of row where 2018 data starts to be nan:
    firstNanIdx = np.where(np.isnan(realData[:,-1]))[0][0]
    #drop first row, with titles:
    realData = realData[1:maxRow,1:]
    realData = realData.T
    
    #get the Y and x data from the loaded dataset and reshape
    #to put in the appropriate format for the CNP:
    testY = realData[-1,:]
    testY = np.reshape(testY, [1, len(testY), 1])
    trainY = realData[:-1,:]
    trainY = np.reshape(trainY, [trainY.shape[0], trainY.shape[1], 1])
    testX = np.reshape(np.arange(testY.shape[1], dtype=np.float32),testY.shape)
    trainX = np.tile(testX, trainY.shape[0]).T
    
    #connects loaded train and test X and Y to tensorflow:
    bchTrainY = tf.convert_to_tensor(trainY)
    bchTrainX = tf.convert_to_tensor(trainX)
    
    #choose functions that will be sampled from:
    sampledFuncsIdx = tf.random.uniform(shape=[nSampledFuncs], minval=0, maxval=bchTrainY.shape[0], dtype=tf.int32)
    trainYt = tf.gather(bchTrainY, sampledFuncsIdx, axis=0)
    trainXt = tf.gather(bchTrainX, sampledFuncsIdx, axis=0)
    trainYc = tf.gather(bchTrainY, sampledFuncsIdx, axis=0)
    trainXc = tf.gather(bchTrainX, sampledFuncsIdx, axis=0)
    #choose context and target points that will be used for
    #training:
    numContext = tf.random.uniform(shape=[], minval=minContextPoints, maxval=maxContextPoints, dtype=tf.int32)
    numTarget = tf.random.uniform(shape=[], minval=minContextPoints, maxval=maxContextPoints, dtype=tf.int32)
    contextIdx = tf.random.uniform(shape=[numContext], minval=0, maxval=trainY.shape[1], dtype=tf.int32)
    targetIdx = tf.random.uniform(shape=[numTarget], minval=0, maxval=trainY.shape[1], dtype=tf.int32)
    trainYt = tf.gather(trainYt, targetIdx, axis=1)
    trainXt = tf.gather(trainXt, targetIdx, axis=1)
    trainYc = tf.gather(trainYc, contextIdx, axis=1)
    trainXc = tf.gather(trainXc, contextIdx, axis=1)
    trainNc = numContext
    #choose context and target points that will be used for
    #testing:    
    numContext = tf.random.uniform(shape=[], minval=minContextPoints, maxval=maxContextPoints, dtype=tf.int32)
    contextIdx = tf.random.uniform(shape=[numContext], minval=0, maxval=testY.shape[1], dtype=tf.int32)
    testYt = tf.convert_to_tensor(testY)
    testXt = tf.convert_to_tensor(testX)
    testYc = tf.convert_to_tensor(testY)
    testXc = tf.convert_to_tensor(testX)
    testYc = tf.gather(testYc, contextIdx, axis=1)
    testXc = tf.gather(testXc, contextIdx, axis=1)
    testNc = numContext
    
    return trainXc, trainYc, trainXt, trainYt, trainNc, testXc,\
           testYc, testXt, testYt, testNc

