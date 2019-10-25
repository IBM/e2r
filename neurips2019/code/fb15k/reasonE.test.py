# purpose: top level script to evaluate an existing model on the test data and compute accuracy.

import os, sys, math
import random

import torch
import torch.nn as nn
from torch.autograd import Variable

import data

import imp
train = imp.load_source('reasonE.train', 'reasonE.train.py')
modelEval = imp.load_source('model.eval', 'model.eval.py')


###

if __name__ == '__main__':

  dataPath, modelSavePath, modelSaveNamePrefix, embedDim, lossMargin, negSampleSizeRatio = train.getTrainParams()
  modelSaveNamePrefix = 'model2'

  batchSize = 30000

  print('Loading data...')
  sys.stdout.flush()
  dataObj = data.FB15KReasonData(dataPath, negSampleSizeRatio, modelSavePath, modelSaveNamePrefix)
  entityCount = dataObj.getEntityCount()
  uConceptCount = dataObj.getUConceptCount()
  bConceptCount = dataObj.getBConceptCount()
  sys.stdout.flush()

  with open(modelSavePath+'/'+modelSaveNamePrefix+'.'+'nIters','r') as f:
    lst = f.readlines()
    nIters = int(lst[0].strip())
  nIters = 5000

  print('Evaluation...')
  print(' ','with model after '+str(nIters)+' iters of training')

  evalObj = modelEval.ModelEval()
  evalObj.setParam(dataObj, entityCount, uConceptCount, bConceptCount, embedDim, batchSize)
  evalObj.loadModel(modelSavePath, modelSaveNamePrefix, str(nIters))
  evalObj.evalModel()


