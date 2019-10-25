# purpose: top level script to train a new model for the specified set of parameters.

import os, sys, math
import random

import torch
import torch.nn as nn
from torch.autograd import Variable

import data

import imp
modelTrainer = imp.load_source('model.trainer', 'model.trainer.py')


def getTrainParams():
  return dataPath, modelSavePath, modelSaveNamePrefix, embedDim, lossMargin, negSampleSizeRatio


###

dataPath = 'data/'
modelSavePath = 'model/'
modelSaveNamePrefix = 'model1'
embedDim = 100
lossMargin = 1.0
negSampleSizeRatio = 3

if __name__ == '__main__':

  dataPath, modelSavePath, modelSaveNamePrefix, embedDim, lossMargin, negSampleSizeRatio = getTrainParams()

  learningRate = 0.001
  nIters = 1000
  batchSize = 49336
  #batchSize = 881

  print('Training...')
  print(' ','fresh training')
  print(' ',str(nIters)+' iters to do now')

  dataObj = data.LUBM1UReasonData(dataPath, negSampleSizeRatio)
  sys.stdout.flush()

  trainer = modelTrainer.ModelTrainer(dataObj, dataObj.getEntityCount(), dataObj.getUConceptCount(), dataObj.getBConceptCount(), embedDim)
  logF = open(modelSavePath+'/'+modelSaveNamePrefix+'.'+'log', 'w')
  logF.write('Train: nIters='+str(nIters)+'\n')
  trainer.init(logF)
  trainer.trainIters(batchSize, learningRate, nIters, lossMargin, logF)
  sys.stdout.flush()
  logF.close()

  trainer.saveModel(modelSavePath, modelSaveNamePrefix, str(nIters))
  dataObj.saveEntityConceptMaps(modelSavePath, modelSaveNamePrefix)

  with open(modelSavePath+'/'+modelSaveNamePrefix+'.'+'nIters', 'w') as f:
    f.write("%s\n" % str(nIters))
    f.close()


