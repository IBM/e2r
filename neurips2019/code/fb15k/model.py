# purpose: model definition class - to define model and forward step.

import os, sys
import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class ReasonEModel(nn.Module):
  def __init__(self, entityCount, uConceptCount, bConceptCount, embedDim):
    super(ReasonEModel, self).__init__()
    self.embedDim = embedDim
    self.entityCount = entityCount
    self.uConceptCount = uConceptCount
    self.bConceptCount = bConceptCount

    self.baseMat = Variable(torch.FloatTensor(torch.eye(embedDim)))
    self.entityEmbed = nn.Embedding(entityCount, embedDim)
    self.bConceptHEmbed = nn.Embedding(bConceptCount, embedDim)
    self.bConceptTEmbed = nn.Embedding(bConceptCount, embedDim)
    nn.init.xavier_uniform_(self.entityEmbed.weight)
    nn.init.xavier_uniform_(self.bConceptHEmbed.weight)
    nn.init.xavier_uniform_(self.bConceptTEmbed.weight)
    self.entityEmbed.weight.data = F.normalize(self.entityEmbed.weight.data, p=2, dim=1)
    self.bConceptHEmbed.weight.data = F.normalize(self.bConceptHEmbed.weight.data, p=2, dim=1)
    self.bConceptTEmbed.weight.data = F.normalize(self.bConceptTEmbed.weight.data, p=2, dim=1)

  def forward(self, aBHE, aBTE, aBC, nABHE, nABTE, nABC, uniqE, uniqBC, lossMargin, device):
    aBHEE = self.entityEmbed(aBHE)
    aBTEE = self.entityEmbed(aBTE)
    aBCHE = self.bConceptHEmbed(aBC)
    aBCTE = self.bConceptTEmbed(aBC)
    nABHEE = self.entityEmbed(nABHE)
    nABTEE = self.entityEmbed(nABTE)
    nABCHE = self.bConceptHEmbed(nABC)
    nABCTE = self.bConceptTEmbed(nABC)

    uniqEE = self.entityEmbed(uniqE)
    uniqBCHE = self.bConceptHEmbed(uniqBC)
    uniqBCTE = self.bConceptTEmbed(uniqBC)

    zero = Variable(torch.FloatTensor([0.0]))
    zero = zero.to(device)
    one = Variable(torch.FloatTensor([1.0]))
    one = one.to(device)
    halfDim = Variable(torch.FloatTensor([self.embedDim/2.0]))
    halfDim = halfDim.to(device)
    margin = Variable(torch.FloatTensor([lossMargin]))
    margin = margin.to(device)

    tmpBE2CH = (one-aBCHE)*aBHEE
    tmpBE2CT = (one-aBCTE)*aBTEE
    bE2CMemberL = torch.sum(tmpBE2CH*tmpBE2CH, dim =1) + torch.sum(tmpBE2CT*tmpBE2CT, dim=1)

    tmpNBE2CH = (one-nABCHE)*nABHEE
    tmpNBE2CT = (one-nABCTE)*nABTEE
    tmpNBL = torch.sum(tmpNBE2CH* tmpNBE2CH, dim=1) + torch.sum(tmpNBE2CT*tmpNBE2CT, dim=1)
    bE2CDiscMemberL = torch.max(margin-tmpNBL, zero)

    tmpE = torch.sum(uniqEE*uniqEE, dim=1) - one
    uniqENormL = tmpE*tmpE

    tmpBCH = uniqBCHE*(one-uniqBCHE)
    tmpBCT = uniqBCTE*(one-uniqBCTE)
    uniqBCBasisAlignL = torch.sum(tmpBCH*tmpBCH, dim=1) + torch.sum(tmpBCT*tmpBCT, dim=1)

    tmpBCHDim = torch.sum(torch.abs(uniqBCHE), dim=1)
    tmpBCHL = torch.max(one-tmpBCHDim, zero)
    tmpBCTDim = torch.sum(torch.abs(uniqBCTE), dim=1)
    tmpBCTL = torch.max(one-tmpBCTDim, zero)
    uniqBCBasisCountL = tmpBCHL + tmpBCTL
    
    return bE2CMemberL, bE2CDiscMemberL, uniqENormL, uniqBCBasisAlignL, uniqBCBasisCountL
    
  def getEntityEmbedding(self, e):
    return self.entityEmbed(e)

  def getBConceptHEmbedding(self, c):
    return self.bConceptHEmbed(c)

  def getBConceptTEmbedding(self, c):
    return self.bConceptTEmbed(c)

  def getBaseMat(self):
    return self.baseMat


