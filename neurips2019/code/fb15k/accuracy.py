# purpose: accuracy computation specific functions

import os, sys
import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Accuracy:

  def computeRankingAccuracy(self, candidateLst, scoreLst, memberLst):
    rankMap = self.getMemberRank(candidateLst, scoreLst, memberLst)
    return self.computeMetrics(list(rankMap.values()))

  def getMemberRank(self, candidateLst, scoreLst, memberLst):
    rankLst = self.getRankList(scoreLst)
    return self.getRankMap(candidateLst, rankLst, memberLst)

  def getRankList(self, scoreLst):
    return numpy.argsort(scoreLst)

  def getRankMap(self, candidateLst, rankLst, memberLst):
    memberSet = set([])
    for m in memberLst:
      memberSet.add(m)
    rankMap = {}
    missCount = 0
    for i in range(len(rankLst)):
      e = candidateLst[rankLst[i]]
      if e in memberSet:
        rankMap[e] = missCount+1
        memberSet.remove(e)
        if len(memberSet)<=0:
          break
      else:
        missCount += 1
    return rankMap

  def computeMetrics(self, rankAccLst):
    mr = 0.0
    mrr = 0.0
    r1 = 0
    r2 = 0
    r3 = 0
    r5 = 0
    r10 = 0
    for r in rankAccLst:
      mr += r
      mrr += (1.0/r)
      if r==1: r1 += 1
      if r<=2: r2 += 1
      if r<=3: r3 += 1
      if r<=5: r5 += 1
      if r<=10: r10 += 1
    result = {}
    result['MR'] = mr / len(rankAccLst)
    result['MRR'] = mrr / len(rankAccLst)
    result['R1%'] = r1 * 100.0/len(rankAccLst)
    result['R2%'] = r2 * 100.0/len(rankAccLst)
    result['R3%'] = r3 * 100.0/len(rankAccLst)
    result['R5%'] = r5 * 100.0/len(rankAccLst)
    result['R10%'] = r10 * 100.0/len(rankAccLst)
    return result


###

if __name__ == '__main__':

  accObj = Accuracy()


