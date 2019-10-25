# purpose: model evaluation class - to load and evaluate an existing model on the test data and compute accuracy.

import os, sys
import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import accuracy


class ModelEval:
  def setParam(self, dataObj, entityCount, uConceptCount, bConceptCount, embedDim, batchSize):
    self.dataObj = dataObj
    self.entityCount = entityCount
    self.uConceptCount = uConceptCount
    self.bConceptCount = bConceptCount
    self.embedDim = embedDim
    self.batchSize = batchSize

  def setModel(self, model, device):
    self.model = model
    self.device = device

  def loadModel(self, modelPath, modelNamePrefix, modelNamePostfix):
    self.model = torch.load(modelPath+'/'+modelNamePrefix+'.reasonEModel.'+modelNamePostfix)
    print('Loaded model '+modelPath+'/'+modelNamePrefix+'.reasonEModel.'+modelNamePostfix)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', self.device)
    sys.stdout.flush()
    self.model = self.model.to(self.device)

  def evalModel(self):
    self.one = Variable(torch.FloatTensor([1.0]))
    self.one = self.one.to(self.device)
    self.accObj = accuracy.Accuracy()
    self.computeEmbeddingQuality()

  def getUClassSpaceMembershipScore(self, uCE, eLst):
    uE = [e for e in eLst]
    uE = Variable(torch.LongTensor(uE))
    uE = uE.to(self.device)
    uEE = self.model.getEntityEmbedding(uE)
    uCE = uCE.repeat(len(eLst), 1)
    tmp = (self.one-uCE)*uEE
    s = torch.sum(tmp*tmp, dim=1)
    sn = s.data.cpu().numpy()
    return sn

  def getBClassSpaceMembershipScore(self, bCE, eLst):
    bHE = []
    bTE = []
    for e in eLst:
      h, t = e
      bHE.append(h)
      bTE.append(t)
    bHE = Variable(torch.LongTensor(bHE))
    bTE = Variable(torch.LongTensor(bTE))
    bHE = bHE.to(self.device)
    bTE = bTE.to(self.device)
    bHEE = self.model.getEntityEmbedding(bHE)
    bHTE = self.model.getEntityEmbedding(bTE)
    bCHE, bCTE = bCE
    bCHE = bCHE.repeat(len(eLst), 1)
    bCTE = bCTE.repeat(len(eLst), 1)
    tmpH = (self.one-bCHE)*bHEE
    tmpT = (self.one-bCTE)*bHTE
    s = torch.sum(tmpH*tmpH, dim=1) + torch.sum(tmpT*tmpT, dim=1)
    sn = s.data.cpu().numpy()
    return sn

  def getClassSpaceMembershipScore(self, cE, eLst):
    uHE = []
    bHE = []
    bTE = []
    uCount = 0
    bCount = 0
    eLstMap = []
    for e in eLst:
      h, t = e
      if t==None:
       uHE.append(h)
       eLstMap.append((0,uCount))
       uCount+=1
      else:
       bHE.append(h)
       bTE.append(t)
       eLstMap.append((1,bCount))
       bCount+=1
    uHE = Variable(torch.LongTensor(uHE))
    bHE = Variable(torch.LongTensor(bHE))
    bTE = Variable(torch.LongTensor(bTE))
    uHE = uHE.to(self.device)
    bHE = bHE.to(self.device)
    bTE = bTE.to(self.device)
    uHEE = self.model.getEntityEmbedding(uHE)
    uTEE = Variable(torch.FloatTensor(torch.zeros(len(uHE), self.embedDim)))
    uTEE = uTEE.to(self.device)
    uEE = torch.cat((uHEE, uTEE), 1)
    bHEE = self.model.getEntityEmbedding(bHE)
    bTEE = self.model.getEntityEmbedding(bTE)
    bEE = torch.cat((bHEE, bTEE), 1)
    eE = torch.cat((uEE, bEE), 0)
    cE = cE.repeat(len(eLst),1)
    one = Variable(torch.FloatTensor([1.0]))
    one = one.to(self.device)    
    tmp = (one-cE)*eE
    s = torch.sum(tmp*tmp, dim=1)
    sn = s.data.cpu().numpy()
    return sn

  def getSortedKeyList(self, keyValueMap):
    keyLst = list(keyValueMap.keys())
    i=0
    while i<len(keyLst):
      j=i+1
      while j<len(keyLst):
        if keyValueMap[keyLst[i]]>keyValueMap[keyLst[j]]:
          tmp = keyLst[i]
          keyLst[i] = keyLst[j]
          keyLst[j] = tmp
        j+=1
      i+=1
    return keyLst

  def computeEmbeddingQuality(self):
    print('Embedding Quality:')
    allBRanks = self.getTestBClassEmbeddingQuality(list(self.dataObj.testBCMemberMap.keys()))
    result = self.accObj.computeMetrics(allBRanks)
    print(' > Binary Abox Classes')
    print('      ',self.getAccuracyPrintText(result))
    allRanks = []
    for r in allBRanks:
      allRanks.append(r)
    result = self.accObj.computeMetrics(allRanks)
    print(' > All Abox Classes')
    print('      ',self.getAccuracyPrintText(result))

  def getUClassEmbeddingQuality(self, classLst):
    e2id = self.dataObj.getEntityMap()
    uc2id = self.dataObj.getUConceptMap()
    entityLst = list(e2id.values())

    allRanks = []
    allCandidateLstLen = 0
    allTrueMembersCount = 0
    for c in classLst:
      trueMembers = set(self.dataObj.getUClassMembers(c))
      print(self.dataObj.id2uc[c], len(trueMembers))
      ranks = []
      candidateLstLen = 0
      for trueMember in trueMembers:
        candidateLst = self.getUClassMembershipCandidateList(trueMember, trueMembers, entityLst)
        candidateLstLen += len(candidateLst)
        scoreLst = self.getUClassSpaceMembershipScore(self.getUClassSpace(c), candidateLst)
        rankLst = self.accObj.getRankList(scoreLst)
        rank = numpy.where(rankLst==0)[0][0] + 1
        ranks.append(rank)
        allRanks.append(rank)
      allCandidateLstLen += candidateLstLen
      allTrueMembersCount += len(trueMembers)
      print('   ', self.accObj.computeMetrics(ranks), candidateLstLen/len(trueMembers))
    print(allCandidateLstLen/allTrueMembersCount)
    return allRanks

  def getBClassEmbeddingQuality(self, classLst):
    e2id = self.dataObj.getEntityMap()
    bc2id = self.dataObj.getBConceptMap()
    entityLst = list(e2id.values())

    allRanks = []
    allCandidateLstLen = 0 
    allTrueMembersCount = 0
    for c in classLst:
      trueMembers = set(self.dataObj.getBClassMembers(c))
      print(self.dataObj.id2bc[c], len(trueMembers))
      ranks = []
      candidateLstLen = 0
      for trueMember in trueMembers:
        candidateLst = self.getBClassMembershipCandidateList(trueMember, trueMembers, entityLst)
        candidateLstLen += len(candidateLst)
        scoreLst = self.getBClassSpaceMembershipScore(self.getBClassSpace(c), candidateLst)
        rankLst = self.accObj.getRankList(scoreLst)
        rank = numpy.where(rankLst==0)[0][0] + 1
        ranks.append(rank)
        allRanks.append(rank)
      allCandidateLstLen += candidateLstLen
      allTrueMembersCount += len(trueMembers)
      print('   ',self.accObj.computeMetrics(ranks), candidateLstLen/len(trueMembers))
    print(allCandidateLstLen/allTrueMembersCount)
    return allRanks

  def getTestBClassEmbeddingQuality(self, classLst):
    e2id = self.dataObj.getEntityMap()
    bc2id = self.dataObj.getBConceptMap()
    entityLst = list(e2id.values())

    allRanks = []
    allCandidateLstLen = 0
    allTrueMembersCount = 0
    for c in classLst:
      testTrueMembers = set(self.dataObj.getTestBClassMembers(c))
      allTrueMembers = set(self.dataObj.getAllBClassMembers(c))
      print(self.dataObj.id2bc[c], len(testTrueMembers))
      ranks = []
      candidateLstLen = 0
      for testTrueMember in testTrueMembers:
        candidateLst = self.getBClassMembershipHCandidateList(testTrueMember, allTrueMembers, entityLst)
        candidateLstLen += len(candidateLst)
        scoreLst = self.getBClassSpaceMembershipScore(self.getBClassSpace(c), candidateLst)
        rankLst = self.accObj.getRankList(scoreLst)
        rank = numpy.where(rankLst==0)[0][0] + 1
        ranks.append(rank)
        allRanks.append(rank)
        candidateLst = self.getBClassMembershipTCandidateList(testTrueMember, allTrueMembers, entityLst)
        candidateLstLen += len(candidateLst)
        scoreLst = self.getBClassSpaceMembershipScore(self.getBClassSpace(c), candidateLst)
        rankLst = self.accObj.getRankList(scoreLst)
        rank = numpy.where(rankLst==0)[0][0] + 1
        ranks.append(rank)
        allRanks.append(rank)
      allCandidateLstLen += candidateLstLen
      allTrueMembersCount += 2*len(testTrueMembers)
      print('   ',self.accObj.computeMetrics(ranks), candidateLstLen/(2*len(testTrueMembers)))
    print(allCandidateLstLen/allTrueMembersCount)
    return allRanks

  def getUClassMembershipCandidateList(self, trueMember, trueMembers, entityLst):
    candidateLst = []
    candidateLst.append(trueMember)
    for e in entityLst:
      if e in trueMembers:
        continue
      candidateLst.append(e)
    return candidateLst

  def getBClassMembershipCandidateList(self, trueMember, trueMembers, entityLst):
    candidateLst = []
    candidateLst.append(trueMember)
    h, t = trueMember
    for e in entityLst:
      if not (h, e) in trueMembers:
        candidateLst.append((h, e))
      if not (e, t) in trueMembers:
        candidateLst.append((e, t))
    return candidateLst

  def getBClassMembershipHCandidateList(self, trueMember, trueMembers, entityLst):
    candidateLst = []
    candidateLst.append(trueMember)
    h, t = trueMember
    for e in entityLst:
      if not (e, t) in trueMembers:
        candidateLst.append((e, t))
    return candidateLst

  def getBClassMembershipTCandidateList(self, trueMember, trueMembers, entityLst):
    candidateLst = []
    candidateLst.append(trueMember)
    h, t = trueMember
    for e in entityLst:
      if not (h, e) in trueMembers:
        candidateLst.append((h, e))
    return candidateLst

  def getUClassSpace(self, c):
    cT = Variable(torch.LongTensor([c]))
    cT = cT.to(self.device)
    uCE = self.model.getUConceptEmbedding(cT)
    return uCE

  def getBClassSpace(self, c):
    cT = Variable(torch.LongTensor([c]))
    cT = cT.to(self.device)
    bCHE = self.model.getBConceptHEmbedding(cT)
    bCTE = self.model.getBConceptTEmbedding(cT)
    return bCHE, bCTE

  def getEntityEmbedding(self, eName):
    eT = Variable(torch.LongTensor([self.dataObj.getEntityId(eName)]))
    eT = eT.to(self.device)
    eE = self.model.getEntityEmbedding(eT)
    return eE

  def getClassEmbedding(self, cName):
    return self.getClassSpace(self.dataObj.getClassId(cName))

  def getAccuracyPrintText(self, resObj):
    retVal = 'MR='+'{:.1f}'.format(resObj['MR'])
    retVal += ', MRR='+'{:.2f}'.format(resObj['MRR'])
    retVal += ', R1%='+'{:.1f}'.format(resObj['R1%'])
    retVal += ', R2%='+'{:.1f}'.format(resObj['R2%'])
    retVal += ', R3%='+'{:.1f}'.format(resObj['R3%'])
    retVal += ', R5%='+'{:.1f}'.format(resObj['R5%'])
    retVal += ', R10%='+'{:.1f}'.format(resObj['R10%'])
    return retVal


