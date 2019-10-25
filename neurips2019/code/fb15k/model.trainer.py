# purpose: model trainer class - to train a new model or load and retrain an existing model on the training data for a specific number of iterations and store the resultant model.

import os, sys
import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import model


class ModelTrainer:
  def __init__(self, dataObj, entityCount, uConceptCount, bConceptCount, embedDim):
    self.dataObj = dataObj
    self.entityCount = entityCount
    self.uConceptCount = uConceptCount
    self.bConceptCount = bConceptCount
    self.embedDim = embedDim

  def init(self, logF, retrainFlag=False, modelPath=None, modelNamePrefix=None, modelNamePostfix=None):
    if retrainFlag==False:
      self.model = model.ReasonEModel(self.entityCount, self.uConceptCount, self.bConceptCount, self.embedDim)
    else:
      self.model = self.loadModel(modelPath, modelNamePrefix, modelNamePostfix)

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', self.device)
    sys.stdout.flush()

    self.model = self.model.to(self.device)

  def trainIters(self, batchSize, learningRate, nIters, lossMargin, logF):
    print('Training iters...')
    sys.stdout.flush()

    modelOpt = torch.optim.Adam(self.model.parameters(), lr = learningRate)

    for it in range(0, nIters):
      self.dataObj.updateRandomNegAboxTripleList()
      self.dataObj.updateRandomTrainIndexList()
      bE2CMemberAccLoss = 0
      bE2CDiscMemberAccLoss = 0
      uniqENormAccLoss = 0
      uniqBCBasisAlignAccLoss = 0
      uniqBCBasisCountAccLoss = 0
      accLoss = 0
      accCount = 0
      for tI in range(self.dataObj.getTrainDataLen(batchSize)):
        modelOpt.zero_grad()
        aBHET, aBTET, aBCT, nABHET, nABTET, nABCT, uniqET, uniqBCT = self.dataObj.getTrainDataTensor(tI, batchSize)
  
        aBHET = aBHET.to(self.device)
        aBTET = aBTET.to(self.device)
        aBCT = aBCT.to(self.device)
        nABHET = nABHET.to(self.device)
        nABTET = nABTET.to(self.device)
        nABCT = nABCT.to(self.device)
        uniqET = uniqET.to(self.device)
        uniqBCT = uniqBCT.to(self.device)

        bE2CMemberL, bE2CDiscMemberL, uniqENormL, uniqBCBasisAlignL, uniqBCBasisCountL = self.model(aBHET, aBTET, aBCT, nABHET, nABTET, nABCT, uniqET, uniqBCT, lossMargin, self.device)

        bE2CMemberLoss = torch.sum(bE2CMemberL)/len(aBHET)
        bE2CDiscMemberLoss = torch.sum(bE2CDiscMemberL)/len(nABHET)
        uniqENormLoss = torch.sum(uniqENormL)/len(uniqET)
        uniqBCBasisAlignLoss = torch.sum(uniqBCBasisAlignL)/len(uniqBCT)
        uniqBCBasisCountLoss = torch.sum(uniqBCBasisCountL)/len(uniqBCT)

        loss = bE2CMemberLoss + bE2CDiscMemberLoss + uniqENormLoss + uniqBCBasisAlignLoss + uniqBCBasisCountLoss

        loss.backward()
        modelOpt.step()

        bE2CMemberAccLoss += bE2CMemberLoss.item()
        bE2CDiscMemberAccLoss += bE2CDiscMemberLoss.item()
        uniqENormAccLoss += uniqENormLoss.item()
        uniqBCBasisAlignAccLoss += uniqBCBasisAlignLoss.item()
        uniqBCBasisCountAccLoss += uniqBCBasisCountLoss.item()
        accLoss += loss.item()
        accCount += 1
        c = accCount
        print('iter='+str(it)+' :', 'overall loss='+'{:.5f}'.format(accLoss/c)+',', 'bE2CMember='+'{:.5f}'.format(bE2CMemberAccLoss/c)+',', 'bE2CDiscMember='+'{:.5f}'.format(bE2CDiscMemberAccLoss/c)+',', 'uniqENorm='+'{:.5f}'.format(uniqENormAccLoss/c)+',', 'uniqBCBasisAlign='+'{:.5f}'.format(uniqBCBasisAlignAccLoss/c)+',', 'uniqBCBasisCount='+'{:.5f}'.format(uniqBCBasisCountAccLoss/c))

      accLoss /= accCount
      bE2CMemberAccLoss /= accCount
      bE2CDiscMemberAccLoss /= accCount
      uniqENormAccLoss /= accCount
      uniqBCBasisAlignAccLoss /= accCount
      uniqBCBasisCountAccLoss /= accCount
 
  def loadModel(self, modelPath, modelNamePrefix, modelNamePostfix):
    model = torch.load(modelPath+'/'+modelNamePrefix+'.reasonEModel.'+modelNamePostfix)
    print('Loaded model '+modelPath+'/'+modelNamePrefix+'.reasonEModel.'+modelNamePostfix)
    return model

  def saveModel(self, modelPath, modelNamePrefix, modelNamePostfix):
    torch.save(self.model, modelPath+'/'+modelNamePrefix+'.reasonEModel.'+modelNamePostfix)
    print('Saved model '+modelPath+'/'+modelNamePrefix+'.reasonEModel.'+modelNamePostfix)


