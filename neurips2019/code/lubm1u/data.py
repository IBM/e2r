# purpose: data loader class - to process the data and shape it into a form to serve the data specific requirements of the model training and evaluation.

import os, sys, math
import random

import json
from pprint import pprint
import re

import torch
from torch.autograd import Variable


class LUBM1UReasonData:

  def __init__(self, path, negSampleSizeRatio, mapLoadPath=None, mapLoadNamePrefix=None):
    self.negSampleSizeRatio = negSampleSizeRatio
    if mapLoadPath==None and mapLoadNamePrefix==None:
      self.e2id, self.uc2id, self.bc2id = self.getEntitiesConceptsMap(path+'/lubm-1u-pruned-reasoning-abox-triples.txt', path+'/lubm-1u-pruned-reasoning-tbox-triples.txt')
    else:
      self.loadEntityConceptMaps(mapLoadPath, mapLoadNamePrefix)
    print('Entities count:', len(self.e2id), 'ConceptsU count:', len(self.uc2id), 'ConceptsB count:', len(self.bc2id))
    self.id2e = self.getInvMap(self.e2id)
    self.id2uc = self.getInvMap(self.uc2id)
    self.id2bc = self.getInvMap(self.bc2id)
    self.aboxUTripleLst, self.aboxBTripleLst = self.readAboxTripleList(path+'/lubm-1u-pruned-reasoning-abox-triples.txt')
    self.tboxUTripleLst, self.tboxBTripleLst = self.readTboxTripleList(path+'/lubm-1u-pruned-reasoning-tbox-triples.txt')
    self.reldefTripleLst = self.readReldefTripleList(path+'lubm-1u-pruned-reasoning-reldef-triples.txt')
    print('Triples count:', 'AboxU', len(self.aboxUTripleLst), 'AboxB', len(self.aboxBTripleLst), 'TboxU', len(self.tboxUTripleLst), 'TboxB', len(self.tboxBTripleLst), 'RelDef', len(self.reldefTripleLst))
    self.uCMemberMap, self.bCMemberMap = self.getClassMemberMap()
    self.aboxUCLst = [uc for uc in self.uCMemberMap.keys()]
    self.aboxBCLst = [bc for bc in self.bCMemberMap.keys()]
    self.uCHierInfo, self.bCHierInfo = self.getClassHierInfo()
    self.addParentClassMemberMap()
    self.negAboxUTripleLst, self.negAboxBTripleLst = self.getNegAboxTripleList()
    self.trainLen = len(self.aboxUTripleLst) if len(self.aboxUTripleLst)>len(self.aboxBTripleLst) else len(self.aboxBTripleLst)
    self.aboxUTripleIndexMap = self.getRepeatIndexMap(len(self.aboxUTripleLst), self.trainLen)
    self.aboxBTripleIndexMap = self.getRepeatIndexMap(len(self.aboxBTripleLst), self.trainLen)
    self.trainILst = self.genRandomIndexList(self.trainLen)

    self.allAboxUTripleLst, self.allAboxBTripleLst = self.readAllAboxTripleList(path+'/lubm-1u-pruned-reasoning-abox-triples.txt', path+'/lubm-1u-pruned-reasoning-test-triples.txt')
    self.allUCMemberMap, self.allBCMemberMap = self.getAllClassMemberMap()
    self.testAboxUTripleLst, self.testAboxBTripleLst = self.readAboxTripleList(path+'/lubm-1u-pruned-reasoning-test-triples.txt')
    self.testUCMemberMap, self.testBCMemberMap = self.getTestClassMemberMap()
    print('Test triples count: ', 'ABoxU', len(self.testAboxUTripleLst), 'AboxB', len(self.testAboxBTripleLst), 'ConceptB count', len(self.testBCMemberMap.keys()))

  def getEntitiesConceptsMap(self, aboxTripleLstFile, tboxTripleLstFile):
    eCount = 0
    ucCount = 0
    bcCount = 0
    e2id = {}
    uc2id = {}
    bc2id = {}
    with open(aboxTripleLstFile, 'r') as fp:
      for line in fp:
        wLst = line.strip().split('\t')
        if wLst[1].strip() == 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type':
          if not wLst[0].strip() in e2id:
            e2id[wLst[0].strip()] = eCount
            eCount += 1
          if not wLst[2].strip() in uc2id:
            uc2id[wLst[2].strip()] = ucCount
            ucCount += 1
        else:
          if not wLst[0].strip() in e2id:
            e2id[wLst[0].strip()] = eCount
            eCount += 1
          if not wLst[2].strip() in e2id:
            e2id[wLst[2].strip()] = eCount
            eCount += 1
          if not wLst[1].strip() in bc2id:
            bc2id[wLst[1].strip()] = bcCount
            bcCount += 1
      fp.close()
    while True:
      updateFlag = False
      resolvedFlag = True
      with open(tboxTripleLstFile, 'r') as fp:
        for line in fp:
          wLst = line.strip().split('\t')
          if wLst[0].strip() in uc2id:
            if wLst[2].strip() not in uc2id:
              uc2id[wLst[2].strip()] = ucCount
              ucCount += 1
              updateFlag = True
          elif wLst[0].strip() in bc2id:
            if wLst[2].strip() not in bc2id:
              bc2id[wLst[2].strip()] = bcCount
              bcCount += 1
              updateFlag = True
          else:
            resolvedFlag = False
        fp.close()
        if resolvedFlag:
          if updateFlag:
            break
          else:
            print('Error: Invalid triples in T-Box')
            return None
    return e2id, uc2id, bc2id

  def readAllAboxTripleList(self, trainTripleLstFile, testTripleLstFile):
    uTripleLst = []
    bTripleLst = []
    currUTripleLst, currBTripleLst = self.readAboxTripleList(trainTripleLstFile)
    for triple in currUTripleLst:
      uTripleLst.append(triple)
    for triple in currBTripleLst:
      bTripleLst.append(triple)
    currUTripleLst, currBTripleLst = self.readAboxTripleList(testTripleLstFile)
    for triple in currUTripleLst:
      uTripleLst.append(triple)
    for triple in currBTripleLst:
      bTripleLst.append(triple)
    return uTripleLst, bTripleLst

  def readAboxTripleList(self, tripleLstFile):
    uTripleLst = []
    bTripleLst = []
    with open(tripleLstFile, 'r') as fp:
      for line in fp:
        wLst = line.strip().split('\t')
        if wLst[1].strip() == 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type':
          if (not wLst[0].strip() in self.e2id) or (not wLst[2].strip() in self.uc2id):
            print('Error: Entity or Concept not in vocab:', line.strip())
            return None
          uTripleLst.append((self.e2id[wLst[0].strip()], self.uc2id[wLst[2].strip()]))
        else:
          if (not wLst[0].strip() in self.e2id) or (not wLst[2].strip() in self.e2id) or (not wLst[1].strip() in self.bc2id):
            print('Error: Entity or Concept not in vocab:', line.strip())
            return None
          bTripleLst.append((self.e2id[wLst[0].strip()], self.bc2id[wLst[1].strip()], self.e2id[wLst[2].strip()]))
      fp.close()
    return uTripleLst, bTripleLst

  def readTboxTripleList(self, tripleLstFile):
    uTripleLst = []
    bTripleLst = []
    with open(tripleLstFile, 'r') as fp:
      for line in fp:
        wLst = line.strip().split('\t')
        if (wLst[0].strip() in self.uc2id) and (wLst[2].strip() in self.uc2id) and (wLst[1].strip() == 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'):
          uTripleLst.append((self.uc2id[wLst[0].strip()], self.uc2id[wLst[2].strip()]))
        elif (wLst[0].strip() in self.bc2id) and (wLst[2].strip() in self.bc2id) and (wLst[1].strip() == 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'):
          bTripleLst.append((self.bc2id[wLst[0].strip()], self.bc2id[wLst[2].strip()]))
        else:
          print('Error: Invalid Tbox triple.', line.strip())
          return None
      fp.close()
    return uTripleLst, bTripleLst

  def readReldefTripleList(self, tripleLstFile):
    tripleLst = []
    with open(tripleLstFile, 'r') as fp:
      for line in fp:
        wLst = line.strip().split('\t')
        if (not wLst[0].strip() in self.uc2id) or (not wLst[1].strip() in self.bc2id) or (not wLst[2].strip() in self.uc2id):
          print('Error: Incorrect relation definition:', line.strip())
          return None
        tripleLst.append((self.uc2id[wLst[0].strip()], self.bc2id[wLst[1].strip()], self.uc2id[wLst[2].strip()]))
    fp.close()
    return tripleLst

  def updateRandomNegAboxTripleList(self):
    self.negAboxUTripleLst, self.negAboxBTripleLst = self.getNegAboxTripleList()

  def getNegAboxTripleList(self):
    uTriples = set(self.aboxUTripleLst)
    bTriples = set(self.aboxBTripleLst)
    entityIdLst = list(self.e2id.values())
    entityCount = len(entityIdLst)
    negUTriples = []
    negBTriples = []
    trialCountThresh = 10*len(self.e2id)
    for triple in self.aboxUTripleLst:
      e, c = triple
      trialCount = 0
      currNegTriples = set([])
      while True:
        re = entityIdLst[random.randint(0, entityCount-1)]
        if ((re, c) not in currNegTriples) and self.validNegUTriple(re, c, uTriples):
          currNegTriples.add((re, c))
          if len(currNegTriples)>=self.negSampleSizeRatio:
            break
        trialCount += 1
        if trialCount > trialCountThresh:
          print('Error: Taking a long time to find negUTriple.')
          return None
      for currNegTriple in currNegTriples:
        negUTriples.append(currNegTriple)
    for triple in self.aboxBTripleLst:
      he, c, te = triple
      if random.randint(0, 1)==0:
        trialCount = 0
        currNegTriples = set([])
        while True:
          rte = entityIdLst[random.randint(0, entityCount-1)]
          if ((he, c, rte) not in currNegTriples) and self.validNegBTriple(he, c, rte, bTriples):
            currNegTriples.add((he, c, rte))
            if len(currNegTriples)>=self.negSampleSizeRatio:
              break
          trialCount += 1
          if trialCount > trialCountThresh:
            print('Error: Taking a long time to find negBTriple.')
            return None
        for currNegTriple in currNegTriples:
          negBTriples.append(currNegTriple)
      else:
        trialCount = 0
        currNegTriples = set([])
        while True:
          rhe = entityIdLst[random.randint(0, entityCount-1)]
          if ((rhe, c, te) not in currNegTriples) and self.validNegBTriple(rhe, c, te, bTriples):
            currNegTriples.add((rhe, c, te))
            if len(currNegTriples)>=self.negSampleSizeRatio:
              break
          trialCount += 1
          if trialCount > trialCountThresh:
            print('Error: Taking a long time to find negBTriple.')
            return None
        for currNegTriple in currNegTriples:
          negBTriples.append(currNegTriple)
    return negUTriples, negBTriples

  def validNegUTriple(self, e, c, triples):
    if (e, c) in triples:
      return False
    if c in self.uCHierInfo:
      for cc in self.uCHierInfo[c]:
         if not self.validNegUTriple(e, cc, triples):
           return False
    return True

  def validNegBTriple(self, he, c, te, triples):
    if (he,c,te) in triples:
      return False
    if c in self.bCHierInfo:
      for cc in self.bCHierInfo[c]:
        if not self.validNegBTriple(he, cc, te, triples):
          return False
    return True

  def getUniqEntityList(self):
    uniqELst = set([])
    for triple in self.aboxUTripleLst:
      e, c = triple
      if not e in uniqELst:
        uniqELst.add(e)
    for triple in self.aboxBTripleLst:
      he, c, te = triple
      if not he in uniqELst:
        uniqELst.add(he)
      if not te in uniqELst:
        uniqELst.add(te)
    return list(uniqELst)

  def getUniqConceptList(self):
    uniqUCLst = set([])
    uniqBCLst = set([])
    for triple in self.aboxTripleLst:
      he, c, te = triple
      if te == None:
        if c not in uniqUCLst:
          uniqUCLst.add(c)
      else:
        if c not in uniqBCLst:
          uniqBCLst.add(c)
    while True:
      flag = True
      for triple in self.tboxTripleLst:
        c1, c2, d = triple
        if c1 in uniqUCLst:
          if c2 not in uniqUCLst:
            uniqUCLst.add(c2)
        elif c1 in uniqBCLst:
          if c2 not in uniqBCLst:
            uniqBCLst.add(c2)
        else:
          flag = False
      if flag:
        break  
    return list(uniqUCLst), list(uniqBCLst)

  def getRepeatIndexMap(self, seqSize, repeatLen):
    return [i%seqSize for i in range(repeatLen)]

  def genRandomIndexList(self, leng):
    lst = [i for i in range(0, leng)]
    random.shuffle(lst)
    return lst

  def updateRandomTrainIndexList(self):
    self.trainILst = self.genRandomIndexList(self.trainLen)

  def getTrainDataTensor(self, batchIndex, batchSize):
    indexLst = [batchIndex*batchSize+i for i in range(batchSize)]
    aUELst = []
    aUCLst = []
    nAUELst = []
    nAUCLst = []
    aBHELst = []
    aBTELst = []
    aBCLst = []
    nABHELst = []
    nABTELst = []
    nABCLst = []

    tUCCLst = []
    tUPCLst = []
    tBCCLst = []
    tBPCLst = []

    uniqELst = []
    uniqUCLst = []
    uniqBCLst = []

    rdBCLst = []
    nRdHUCLst = []
    nRdTUCLst = []

    for index in indexLst:
      rI = self.trainILst[index]
      e, c = self.aboxUTripleLst[self.aboxUTripleIndexMap[rI]]
      aUELst.append(e)
      aUCLst.append(c)
      for j in range(self.negSampleSizeRatio):
        ne, nc = self.negAboxUTripleLst[self.negSampleSizeRatio*self.aboxUTripleIndexMap[rI]+j]
        nAUELst.append(ne)
        nAUCLst.append(nc)

      he, c, te = self.aboxBTripleLst[self.aboxBTripleIndexMap[rI]]
      aBHELst.append(he)
      aBTELst.append(te)
      aBCLst.append(c)
      for j in range(self.negSampleSizeRatio):
        nhe, nc, nte = self.negAboxBTripleLst[self.negSampleSizeRatio*self.aboxBTripleIndexMap[rI]+j]
        nABHELst.append(nhe)
        nABTELst.append(nte)
        nABCLst.append(nc)

    for triple in self.tboxUTripleLst:
      cc, pc = triple
      tUCCLst.append(cc)
      tUPCLst.append(pc)
    for triple in self.tboxBTripleLst:
      cc, pc = triple
      tBCCLst.append(cc)
      tBPCLst.append(pc)

    for e in list(self.id2e.keys()):
      uniqELst.append(e)
    for c in list(self.id2uc.keys()):
      uniqUCLst.append(c)
    for c in list(self.id2bc.keys()):
      uniqBCLst.append(c)

    return Variable(torch.LongTensor(aUELst)), Variable(torch.LongTensor(aUCLst)), Variable(torch.LongTensor(nAUELst)), Variable(torch.LongTensor(nAUCLst)), Variable(torch.LongTensor(aBHELst)), Variable(torch.LongTensor(aBTELst)), Variable(torch.LongTensor(aBCLst)), Variable(torch.LongTensor(nABHELst)), Variable(torch.LongTensor(nABTELst)), Variable(torch.LongTensor(nABCLst)), Variable(torch.LongTensor(tUCCLst)), Variable(torch.LongTensor(tUPCLst)), Variable(torch.LongTensor(tBCCLst)), Variable(torch.LongTensor(tBPCLst)), Variable(torch.LongTensor(uniqELst)), Variable(torch.LongTensor(uniqUCLst)), Variable(torch.LongTensor(uniqBCLst)), None, None, None, None, None

  def getTrainDataLen(self, batchSize):
    return self.trainLen // batchSize

  def getEntityCount(self):
    return len(self.e2id)

  def getEntityList(self):
    return list(self.e2id.keys())

  def getEntityMap(self):
    return self.e2id

  def getUConceptCount(self):
    return len(self.uc2id)

  def getUConceptList(self):
    return list(self.uc2id.keys())

  def getBConceptCount(self):
    return len(self.bc2id)

  def getBConceptList(self):
    return list(self.bc2id.keys())

  def getUConceptMap(self):
    return self.uc2id

  def getBConceptMap(self):
    return self.bc2id

  def saveEntityConceptMaps(self, savePath, saveNamePrefix):
    self.saveIdMap(self.e2id, savePath+'/'+saveNamePrefix+'.entityMap')
    self.saveIdMap(self.uc2id, savePath+'/'+saveNamePrefix+'.unaryConceptMap')
    self.saveIdMap(self.bc2id, savePath+'/'+saveNamePrefix+'.binaryConceptMap')

  def loadEntityConceptMaps(self, loadPath, loadNamePrefix):
    self.e2id = self.loadIdMap(loadPath+'/'+loadNamePrefix+'.entityMap')
    self.uc2id = self.loadIdMap(loadPath+'/'+loadNamePrefix+'.unaryConceptMap')
    self.bc2id = self.loadIdMap(loadPath+'/'+loadNamePrefix+'.binaryConceptMap')

  def saveIdMap(self, idMap, fileN):
    with open(fileN, 'w') as f:
      f.write("%s\n" % str(len(idMap)))
      for e in idMap.keys():
        txt = e+' '+str(idMap[e])
        f.write("%s\n" % txt)
      f.close()

  def loadIdMap(self, fileN):
    with open(fileN, 'r') as f:
      lst = f.readlines()
      f.close()
      count = int(lst[0].strip())
      idMap = {}
      for i in range(count):
        e, m = lst[i+1].strip().split()
        idMap[e.strip()] = int(m.strip())
      return idMap

  def getClassHierInfo(self):
    uCHierInfo = {}
    for triple in self.tboxUTripleLst:
      cc, pc = triple
      if not pc in uCHierInfo:
        uCHierInfo[pc] = set([])
      if cc in uCHierInfo[pc]:
        print('Error: Duplicate entry in T-box for unary')
        return None
      uCHierInfo[pc].add(cc)
    bCHierInfo = {}
    for triple in self.tboxBTripleLst:
      cc, pc = triple
      if not pc in bCHierInfo:
        bCHierInfo[pc] = set([])
      if cc in bCHierInfo[pc]:
        print('Error: Duplicate entry in T-box for binary')
        return None
      bCHierInfo[pc].add(cc)
    return uCHierInfo, bCHierInfo

  def getClassMemberMap(self):
    uCMemberMap = {}
    for triple in self.aboxUTripleLst:
      e, c = triple
      if not c in uCMemberMap.keys():
        uCMemberMap[c] = set([])
      if not e in uCMemberMap[c]:
        uCMemberMap[c].add(e)
    bCMemberMap = {}
    for triple in self.aboxBTripleLst:
      he, c, te = triple
      if not c in bCMemberMap.keys():
        bCMemberMap[c] = set([])
      if not (he, te) in bCMemberMap[c]:
        bCMemberMap[c].add((he, te))
    return uCMemberMap, bCMemberMap

  def addParentClassMemberMap(self):
    uCParents = set([])
    for uc in self.uCHierInfo.keys():
      uCParents.add(uc)
    while True:
      if len(uCParents)<=0:
        break
      removeUC = set([])
      for uc in uCParents:
        flag = False
        for cuc in self.uCHierInfo[uc]:
          if cuc in uCParents:
            flag = True
            break
        if flag:
          continue
        uCMembers = set([])
        if uc in self.uCMemberMap.keys():
          for member in self.uCMemberMap[uc]:
            uCMembers.add(member)
        for cuc in self.uCHierInfo[uc]:
          for member in self.uCMemberMap[cuc]:
            if member not in uCMembers:
              uCMembers.add(member)
        self.uCMemberMap[uc] = uCMembers
        removeUC.add(uc)
      for uc in removeUC:
        uCParents.remove(uc)

    bCParents = set([])
    for bc in self.bCHierInfo.keys():
      bCParents.add(bc)
    while True:
      if len(bCParents)<=0:
        break
      removeBC = set([])
      for bc in bCParents:
        flag = False
        for cbc in self.bCHierInfo[bc]:
          if cbc in bCParents:
            flag = True
            break
        if flag:
          continue
        bCMembers = set([])
        if bc in self.bCMemberMap.keys():
          for member in self.bCMemberMap[bc]:
            bCMembers.add(member) 
        for cbc in self.bCHierInfo[bc]:
          for member in self.bCMemberMap[cbc]:
            if member not in bCMembers:
              bCMembers.add(member)
        self.bCMemberMap[bc] = bCMembers
        removeBC.add(bc)
      for bc in removeBC:
        bCParents.remove(bc) 

  def getAllClassMemberMap(self):
    uCMemberMap = {}
    for triple in self.allAboxUTripleLst:
      e, c = triple
      if not c in uCMemberMap.keys():
        uCMemberMap[c] = set([])
      if not e in uCMemberMap[c]:
        uCMemberMap[c].add(e)
    bCMemberMap = {}
    for triple in self.allAboxBTripleLst:
      he, c, te = triple
      if not c in bCMemberMap.keys():
        bCMemberMap[c] = set([])
      if not (he, te) in bCMemberMap[c]:
        bCMemberMap[c].add((he, te))
    return uCMemberMap, bCMemberMap

  def getTestClassMemberMap(self):
    uCMemberMap = {}
    for triple in self.testAboxUTripleLst:
      e, c = triple
      if not c in uCMemberMap.keys():
        uCMemberMap[c] = set([])
      if not e in uCMemberMap[c]:
        uCMemberMap[c].add(e)
    bCMemberMap = {}
    for triple in self.testAboxBTripleLst:
      he, c, te = triple
      if not c in bCMemberMap.keys():
        bCMemberMap[c] = set([])
      if not (he, te) in bCMemberMap[c]:
        bCMemberMap[c].add((he, te))
    return uCMemberMap, bCMemberMap

  def getUClassMembers(self,c): 
    retVal = set([])
    for e in self.uCMemberMap[c]:
      retVal.add(e)
    return retVal

  def getAllUClassMembers(self, c):
    retVal = set([])
    for e in self.allUCMemberMap[c]:
      retVal.add(e)
    return retVal

  def getTestUClassMembers(self, c):
    retVal = set([])
    for e in self.testUCMemberMap[c]:
      retVal.add(e)
    return retVal

  def getBClassMembers(self, c):
    retVal = set([])
    for e in self.bCMemberMap[c]:
      retVal.add(e)
    return retVal  

  def getAllBClassMembers(self, c):
    retVal = set([])
    for e in self.allBCMemberMap[c]:
      retVal.add(e)
    return retVal

  def getTestBClassMembers(self, c):
    retVal = set([])
    for e in self.testBCMemberMap[c]:
      retVal.add(e)
    return retVal

  def getInvMap(self, m):
    retVal = {}
    for key in m.keys():
      retVal[m[key]] = key
    return retVal

  def getLeafClassList(self):
    return self.leafClassLst

  def getBranchClassList(self):
    return self.branchClassLst

  def readGroundTruth(self, gtFileName):
    retVal = []
    with open(gtFileName, 'r') as fp:
      for line in fp:
        wLst = line.strip().split('|')
        retVal.append([wLst[i].strip() for i in range(1, len(wLst)-1)])
      fp.close()
    return retVal

###

if __name__ == '__main__':

  dataPath = '/u/sha/work/ai-science/kg-embed/lubm/1u/data/dump/'
  data = LUBM1UReasonData(dataPath, 1)
  print(data.readGroundTruth('data/benchmark-gt/q4.txt'))
  exit(0)
  for i in range(100): #range(len(data.aboxTripleLst)):
    print(data.aboxTripleLst[i], data.negAboxTripleLst[i])
  print(data.c2id)
  print(data.tboxTripleLst)
  print(data.negTboxTripleLst)

