import numpy as np
import torch.nn as nn
import torch


def nll(sample):
    return 0.5*torch.sum(torch.pow(sample, 2), dim=[1,2,3])

class Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample, logdet):
        nll_loss = torch.mean(nll(sample))
        assert len(logdet.shape) == 1
        nlogdet_loss = -torch.mean(logdet)
        loss = nll_loss + nlogdet_loss
        reference_nll_loss = torch.mean(nll(torch.randn_like(sample)))
        log = {"images": {},
               "scalars": {
                    "loss": loss, "reference_nll_loss": reference_nll_loss,
                    "nlogdet_loss": nlogdet_loss, "nll_loss": nll_loss,
               }}
        
        return loss, log


class FactorLoss(torch.nn.Module):
    def __init__(self, rho: float = 0.975):
        super().__init__()
        self.rho = rho

    def forward(self, sample1, sample2, logdet1, logdet2, factors):
        nll_loss1 = torch.mean(nll(torch.cat(sample1, dim=1)))
        assert len(logdet1.shape) == 1
        nlogdet_loss1 = -torch.mean(logdet1)
        loss1 = nll_loss1 + nlogdet_loss1
        factor_mask = [
                torch.tensor(((factors==i) | ((factors<0) & (factors!=-i)))[:,None,None,None]).to(
                    sample2[i]) for i in range(len(sample2))]
        sample2_cond = [
                sample2[i] - factor_mask[i]*self.rho*sample1[i]
                for i in range(len(sample2))]
        nll_loss2 = [nll(sample2_cond[i]) for i in range(len(sample2_cond))]
        nll_loss2 = [nll_loss2[i]/(1.0-factor_mask[i][:,0,0,0]*self.rho**2)
                for i in range(len(sample2_cond))]
        nll_loss2 = [torch.mean(nll_loss2[i])
                for i in range(len(sample2_cond))]
        nll_loss2 = sum(nll_loss2)
        assert len(logdet2.shape) == 1
        nlogdet_loss2 = -torch.mean(logdet2)
        loss2 = nll_loss2 + nlogdet_loss2

        loss = loss1 + loss2

        log = {"images": {},
                "scalars": {
                    "loss": loss, "loss1": loss1, "loss2": loss2,
                }}
        
        return loss, log