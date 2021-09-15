import torch
import torch.nn as nn
import torch.nn.functional as F


class EMLossForTarget(nn.Module):

    def __init__(self, nClass=10):
        super(EMLossForTarget, self).__init__()
        self.nClass = nClass

    def forward(self, input, eps, separate=True, only_classifier=None, softmax_first=True):
        """
        Arguments:
            input: a float tensor, score vector of classifier predictions.
            eps: a float, a small value to prevent underflow.
            separate: can optionally separate source and target classifier predictions to compute two entropy losses. Default: True.
            only_classifier: can optionally use only one entropy loss. Default: None (return the sum of two entropy losses).
            softmax_first: can optionally use softmax to normalize the input vector before summation. Default: True.
        Returns:
            a float tensor with shape [1], computed loss.
        """
        if separate:
            prob_source = F.softmax(input[:, :self.nClass], dim=1)
            if (prob_source == 0).sum() != 0:
                weight_source = torch.zeros(prob_source.size()).cuda(prob_source.device)
                weight_source[prob_source == 0] = eps
                loss_source = - (prob_source + weight_source).log().mul(prob_source).sum(1).mean()
            else:
                loss_source = - prob_source.log().mul(prob_source).sum(1).mean()
                
            prob_target = F.softmax(input[:, self.nClass:], dim=1)
            if (prob_target == 0).sum() != 0:
                weight_target = torch.zeros(prob_target.size()).cuda(prob_target.device)
                weight_target[prob_target == 0] = eps
                loss_target = - (prob_target + weight_target).log().mul(prob_target).sum(1).mean()
            else:
                loss_target = - prob_target.log().mul(prob_target).sum(1).mean()

            if not only_classifier:
                loss_sum = loss_source + loss_target
            elif only_classifier == 'T':
                loss_sum = loss_target
            elif only_classifier == 'S':
                loss_sum = loss_source
            else:
                raise ValueError('Unrecognized classifier!')
        else:
            if softmax_first:
                prob = F.softmax(input, dim=1)
                prob_sum = prob[:, :self.nClass] + prob[:, self.nClass:]
            else:
                prob_sum = F.softmax(input[:, :self.nClass] + input[:, self.nClass:], dim=1)
            if (prob_sum == 0).sum() != 0:
                weight_sum = torch.zeros(prob_sum.size()).cuda(prob_sum.device)
                weight_sum[prob_sum == 0] = eps
                loss_sum = - (prob_sum + weight_sum).log().mul(prob_sum).sum(1).mean()
            else:
                loss_sum = - prob_sum.log().mul(prob_sum).sum(1).mean()
        return loss_sum

