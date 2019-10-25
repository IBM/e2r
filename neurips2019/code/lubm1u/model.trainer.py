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
      uE2CMemberAccLoss = 0
      bE2CMemberAccLoss = 0
      uE2CDiscMemberAccLoss = 0
      bE2CDiscMemberAccLoss = 0
      uC2CHierBasisAlignAccLoss = 0
      bC2CHierBasisAlignAccLoss = 0
      uC2CHierBasisCountAccLoss = 0
      bC2CHierBasisCountAccLoss = 0
      uniqENormAccLoss = 0
      uniqUCBasisAlignAccLoss = 0
      uniqBCBasisAlignAccLoss = 0
      uniqUCBasisCountAccLoss = 0
      uniqBCBasisCountAccLoss = 0
      accLoss = 0
      accCount = 0
      for tI in range(self.dataObj.getTrainDataLen(batchSize)):
        modelOpt.zero_grad()
        aUET, aUCT, nAUET, nAUCT, aBHET, aBTET, aBCT, nABHET, nABTET, nABCT, tUCCT, tUPCT, tBCCT, tBPCT, uniqET, uniqUCT, uniqBCT, rdHUCT, rdTUCT, rdBCT, nRdHUCT, nRdTUCT = self.dataObj.getTrainDataTensor(tI, batchSize)
  
        aUET = aUET.to(self.device)
        aUCT = aUCT.to(self.device)
        nAUET = nAUET.to(self.device)
        nAUCT = nAUCT.to(self.device)
        aBHET = aBHET.to(self.device)
        aBTET = aBTET.to(self.device)
        aBCT = aBCT.to(self.device)
        nABHET = nABHET.to(self.device)
        nABTET = nABTET.to(self.device)
        nABCT = nABCT.to(self.device)
        tUCCT = tUCCT.to(self.device)
        tUPCT = tUPCT.to(self.device)
        tBCCT = tBCCT.to(self.device)
        tBPCT = tBPCT.to(self.device)
        uniqET = uniqET.to(self.device)
        uniqUCT = uniqUCT.to(self.device)
        uniqBCT = uniqBCT.to(self.device)

        uE2CMemberL, bE2CMemberL, uE2CDiscMemberL, bE2CDiscMemberL, uC2CHierBasisAlignL, bC2CHierBasisAlignL, uC2CHierBasisCountL, bC2CHierBasisCountL, uniqENormL, uniqUCBasisAlignL, uniqBCBasisAlignL, uniqUCBasisCountL, uniqBCBasisCountL = self.model(aUET, aUCT, nAUET, nAUCT, aBHET, aBTET, aBCT, nABHET, nABTET, nABCT, tUCCT, tUPCT, tBCCT, tBPCT, uniqET, uniqUCT, uniqBCT, rdHUCT, rdTUCT, rdBCT, nRdHUCT, nRdTUCT, lossMargin, self.device)

        uE2CMemberLoss = torch.sum(uE2CMemberL)/len(aUET)
        bE2CMemberLoss = torch.sum(bE2CMemberL)/len(aBHET)
        uE2CDiscMemberLoss = torch.sum(uE2CDiscMemberL)/len(nAUET)
        bE2CDiscMemberLoss = torch.sum(bE2CDiscMemberL)/len(nABHET)
        uC2CHierBasisAlignLoss = torch.sum(uC2CHierBasisAlignL)/len(tUCCT)
        bC2CHierBasisAlignLoss = torch.sum(bC2CHierBasisAlignL)/len(tBCCT)
        uC2CHierBasisCountLoss = torch.sum(uC2CHierBasisCountL)/len(tUCCT)
        bC2CHierBasisCountLoss = torch.sum(bC2CHierBasisCountL)/len(tBCCT)
        uniqENormLoss = torch.sum(uniqENormL)/len(uniqET)
        uniqUCBasisAlignLoss = torch.sum(uniqUCBasisAlignL)/len(uniqUCT)
        uniqBCBasisAlignLoss = torch.sum(uniqBCBasisAlignL)/len(uniqBCT)
        uniqUCBasisCountLoss = torch.sum(uniqUCBasisCountL)/len(uniqUCT)
        uniqBCBasisCountLoss = torch.sum(uniqBCBasisCountL)/len(uniqBCT)

        loss = uE2CMemberLoss + bE2CMemberLoss + uE2CDiscMemberLoss + bE2CDiscMemberLoss + uC2CHierBasisAlignLoss + bC2CHierBasisAlignLoss + uC2CHierBasisCountLoss + bC2CHierBasisCountLoss + uniqENormLoss + uniqUCBasisAlignLoss + uniqBCBasisAlignLoss + uniqUCBasisCountLoss + uniqBCBasisCountLoss

        loss.backward()
        modelOpt.step()

        uE2CMemberAccLoss += uE2CMemberLoss.item()
        bE2CMemberAccLoss += bE2CMemberLoss.item()
        uE2CDiscMemberAccLoss += uE2CDiscMemberLoss.item()
        bE2CDiscMemberAccLoss += bE2CDiscMemberLoss.item()
        uC2CHierBasisAlignAccLoss += uC2CHierBasisAlignLoss.item()
        bC2CHierBasisAlignAccLoss += bC2CHierBasisAlignLoss.item()
        uC2CHierBasisCountAccLoss += uC2CHierBasisCountLoss.item()
        bC2CHierBasisCountAccLoss += bC2CHierBasisCountLoss.item()
        uniqENormAccLoss += uniqENormLoss.item()
        uniqUCBasisAlignAccLoss += uniqUCBasisAlignLoss.item()
        uniqBCBasisAlignAccLoss += uniqBCBasisAlignLoss.item()
        uniqUCBasisCountAccLoss += uniqUCBasisCountLoss.item()
        uniqBCBasisCountAccLoss += uniqBCBasisCountLoss.item()
        rdNegUCAlignAccLoss = 0 
        accLoss += loss.item()
        accCount += 1
        c = accCount
        print('iter='+str(it)+' :', 'overall loss='+'{:.5f}'.format(accLoss/c)+',', 'uE2CMember='+'{:.5f}'.format(uE2CMemberAccLoss/c)+',', 'bE2CMember='+'{:.5f}'.format(bE2CMemberAccLoss/c)+',', 'uE2CDiscMember='+'{:.5f}'.format(uE2CDiscMemberAccLoss/c)+',', 'bE2CDiscMember='+'{:.5f}'.format(bE2CDiscMemberAccLoss/c)+',', 'uC2CHierBasisAlign='+'{:.5f}'.format(uC2CHierBasisAlignAccLoss/c)+',', 'bC2CHierBasisAlign='+'{:.5f}'.format(bC2CHierBasisAlignAccLoss/c)+',', 'uC2CHierBasisCount='+'{:.5f}'.format(uC2CHierBasisCountAccLoss/c)+',', 'bC2CHierBasisCount='+'{:.5f}'.format(bC2CHierBasisCountAccLoss/c)+',', 'uniqENorm='+'{:.5f}'.format(uniqENormAccLoss/c)+',', 'uniqUCBasisAlign='+'{:.5f}'.format(uniqUCBasisAlignAccLoss/c)+',', 'uniqBCBasisAlign='+'{:.5f}'.format(uniqBCBasisAlignAccLoss/c)+',', 'uniqUCBasisCount='+'{:.5f}'.format(uniqUCBasisCountAccLoss/c)+',', 'uniqBCBasisCount='+'{:.5f}'.format(uniqBCBasisCountAccLoss/c)+',', 'rdNegUCAlign='+'{:.5f}'.format(rdNegUCAlignAccLoss))

      accLoss /= accCount
      uE2CMemberAccLoss /= accCount
      bE2CMemberAccLoss /= accCount
      uE2CDiscMemberAccLoss /= accCount
      bE2CDiscMemberAccLoss /= accCount
      uC2CHierBasisAlignAccLoss /= accCount
      bC2CHierBasisAlignAccLoss /= accCount
      uC2CHierBasisCountAccLoss /= accCount
      bC2CHierBasisCountAccLoss /= accCount
      uniqENormAccLoss /= accCount
      uniqUCBasisAlignAccLoss /= accCount
      uniqBCBasisAlignAccLoss /= accCount
      uniqUCBasisCountAccLoss /= accCount
      uniqBCBasisCountAccLoss /= accCount
 
  def loadModel(self, modelPath, modelNamePrefix, modelNamePostfix):
    model = torch.load(modelPath+'/'+modelNamePrefix+'.reasonEModel.'+modelNamePostfix)
    print('Loaded model '+modelPath+'/'+modelNamePrefix+'.reasonEModel.'+modelNamePostfix)
    return model

  def saveModel(self, modelPath, modelNamePrefix, modelNamePostfix):
    torch.save(self.model, modelPath+'/'+modelNamePrefix+'.reasonEModel.'+modelNamePostfix)
    print('Saved model '+modelPath+'/'+modelNamePrefix+'.reasonEModel.'+modelNamePostfix)


