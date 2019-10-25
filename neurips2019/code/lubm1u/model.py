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
    self.uConceptEmbed = nn.Embedding(uConceptCount, embedDim)
    self.bConceptHEmbed = nn.Embedding(bConceptCount, embedDim)
    self.bConceptTEmbed = nn.Embedding(bConceptCount, embedDim)
    nn.init.xavier_uniform_(self.entityEmbed.weight)
    nn.init.xavier_uniform_(self.uConceptEmbed.weight)
    nn.init.xavier_uniform_(self.bConceptHEmbed.weight)
    nn.init.xavier_uniform_(self.bConceptTEmbed.weight)
    self.entityEmbed.weight.data = F.normalize(self.entityEmbed.weight.data, p=2, dim=1)
    self.uConceptEmbed.weight.data = F.normalize(self.uConceptEmbed.weight.data, p=2, dim=1)
    self.bConceptHEmbed.weight.data = F.normalize(self.bConceptHEmbed.weight.data, p=2, dim=1)
    self.bConceptTEmbed.weight.data = F.normalize(self.bConceptTEmbed.weight.data, p=2, dim=1)

  def forward(self, aUE, aUC, nAUE, nAUC, aBHE, aBTE, aBC, nABHE, nABTE, nABC, tUCC, tUPC, tBCC, tBPC, uniqE, uniqUC, uniqBC, rdHUC, rdTUC, rdBC, nRdHUC, nRdTUC, lossMargin, device):
    aUEE = self.entityEmbed(aUE)
    aUCE = self.uConceptEmbed(aUC)
    nAUEE = self.entityEmbed(nAUE)
    nAUCE = self.uConceptEmbed(nAUC)

    aBHEE = self.entityEmbed(aBHE)
    aBTEE = self.entityEmbed(aBTE)
    aBCHE = self.bConceptHEmbed(aBC)
    aBCTE = self.bConceptTEmbed(aBC)
    nABHEE = self.entityEmbed(nABHE)
    nABTEE = self.entityEmbed(nABTE)
    nABCHE = self.bConceptHEmbed(nABC)
    nABCTE = self.bConceptTEmbed(nABC)

    tUCCE = self.uConceptEmbed(tUCC)
    tUPCE = self.uConceptEmbed(tUPC)
    tBCCHE = self.bConceptHEmbed(tBCC)
    tBCCTE = self.bConceptTEmbed(tBCC)
    tBPCHE = self.bConceptHEmbed(tBPC)
    tBPCTE = self.bConceptTEmbed(tBPC)

    uniqEE = self.entityEmbed(uniqE)
    uniqUCE = self.uConceptEmbed(uniqUC)
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

    tmpUE2C = (one-aUCE)*aUEE
    uE2CMemberL = torch.sum(tmpUE2C*tmpUE2C, dim=1)

    tmpBE2CH = (one-aBCHE)*aBHEE
    tmpBE2CT = (one-aBCTE)*aBTEE
    bE2CMemberL = torch.sum(tmpBE2CH*tmpBE2CH, dim =1) + torch.sum(tmpBE2CT*tmpBE2CT, dim=1)

    tmpNUE2C = (one-nAUCE)*nAUEE
    tmpNUL = torch.sum(tmpNUE2C*tmpNUE2C, dim=1)
    uE2CDiscMemberL = torch.max(margin-tmpNUL, zero)

    tmpNBE2CH = (one-nABCHE)*nABHEE
    tmpNBE2CT = (one-nABCTE)*nABTEE
    tmpNBL = torch.sum(tmpNBE2CH* tmpNBE2CH, dim=1) + torch.sum(tmpNBE2CT*tmpNBE2CT, dim=1)
    bE2CDiscMemberL = torch.max(margin-tmpNBL, zero)

    tmpUC2C = tUCCE*(one-tUPCE)
    uC2CHierBasisAlignL = torch.sum(tmpUC2C*tmpUC2C, dim=1)

    tmpBC2CH = tBCCHE*(one-tBPCHE)
    tmpBC2CT = tBCCTE*(one-tBPCTE)
    bC2CHierBasisAlignL = torch.sum(tmpBC2CH*tmpBC2CH, dim=1) + torch.sum(tmpBC2CT*tmpBC2CT, dim=1)

    tmpUCCDim = torch.sum(torch.abs(tUCCE), dim=1) + one
    tmpUPCDim = torch.sum(torch.abs(tUPCE), dim=1)
    uC2CHierBasisCountL = torch.max(tmpUCCDim-tmpUPCDim, zero)

    tmpBCCHDim = torch.sum(torch.abs(tBCCHE), dim=1)
    tmpBCCTDim = torch.sum(torch.abs(tBCCTE), dim=1)
    tmpBCCDim = tmpBCCHDim + tmpBCCTDim + one
    tmpBPCHDim = torch.sum(torch.abs(tBPCHE), dim=1)
    tmpBPCTDim = torch.sum(torch.abs(tBPCTE), dim=1)
    tmpBPCDim = tmpBPCHDim + tmpBPCTDim
    bC2CHierBasisCountL = torch.max(tmpBCCDim-tmpBPCDim, zero)

    tmpE = torch.sum(uniqEE*uniqEE, dim=1) - one
    uniqENormL = tmpE*tmpE

    tmpUC = uniqUCE*(one-uniqUCE)
    uniqUCBasisAlignL = torch.sum(tmpUC*tmpUC, dim=1)

    tmpBCH = uniqBCHE*(one-uniqBCHE)
    tmpBCT = uniqBCTE*(one-uniqBCTE)
    uniqBCBasisAlignL = torch.sum(tmpBCH*tmpBCH, dim=1) + torch.sum(tmpBCT*tmpBCT, dim=1)

    tmpUCDim = torch.sum(torch.abs(uniqUCE), dim=1)
    uniqUCBasisCountL = torch.max(one-tmpUCDim, zero)

    tmpBCHDim = torch.sum(torch.abs(uniqBCHE), dim=1)
    tmpBCHL = torch.max(one-tmpBCHDim, zero)
    tmpBCTDim = torch.sum(torch.abs(uniqBCTE), dim=1)
    tmpBCTL = torch.max(one-tmpBCTDim, zero)
    uniqBCBasisCountL = tmpBCHL + tmpBCTL
    
    return uE2CMemberL, bE2CMemberL, uE2CDiscMemberL, bE2CDiscMemberL, uC2CHierBasisAlignL, bC2CHierBasisAlignL, uC2CHierBasisCountL, bC2CHierBasisCountL, uniqENormL, uniqUCBasisAlignL, uniqBCBasisAlignL, uniqUCBasisCountL, uniqBCBasisCountL
    
  def getEntityEmbedding(self, e):
    return self.entityEmbed(e)

  def getUConceptEmbedding(self, c):
    return self.uConceptEmbed(c)

  def getBConceptHEmbedding(self, c):
    return self.bConceptHEmbed(c)

  def getBConceptTEmbedding(self, c):
    return self.bConceptTEmbed(c)

  def getBaseMat(self):
    return self.baseMat


