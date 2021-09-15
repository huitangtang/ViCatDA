import torch.nn as nn
import torch.nn.functional as F
import torch


class InteractionLossOnTarget(nn.Module):

    def __init__(self, nClass=10):
        super(InteractionLossOnTarget, self).__init__()
        self.nClass = nClass

    def forward(self, input, weight, eps, consistent=False, classifier='T', one_weight=False, softmax_first=True):
        """
        Arguments:
            input: a float tensor, score vector of classifier predictions.
            weight: a float tensor, score vector for computing the weight for each classifier prediction.
            eps: a float, a small value to prevent underflow.
            consistent: can optionally use cross-domain weighting or same-domain weighting. Default: False (cross-domain weighting).
            classifier: can optionally use the source or target classifier predictions to compute the categorical domain adversarial loss. Default: 'T'.
            one_weight: can optionally use one weight for two domain adversarial losses. Default: False.
            softmax_first: can optionally use softmax to normalize the weight vector before summation. Default: True.
        Returns:
            a float tensor with shape [1], computed loss.
        """
        prob_input = F.softmax(input, dim=1)
        temp = torch.zeros(prob_input.size()).cuda(prob_input.device)
        temp[prob_input == 0] = eps 

        if one_weight:
            if softmax_first:
                prob_weight_temp = F.softmax(weight, dim=1)
                prob_weight = prob_weight_temp[:, :self.nClass] + prob_weight_temp[:, self.nClass:]
            else:
                prob_weight = F.softmax(weight[:, :self.nClass] + weight[:, self.nClass:], dim=1)
            if classifier == 'T':
                loss = - (prob_weight * ((prob_input[:, self.nClass:] + temp[:, self.nClass:]).log())).sum(1).mean()
            elif classifier == 'S':
                loss = - (prob_weight * ((prob_input[:, :self.nClass] + temp[:, :self.nClass]).log())).sum(1).mean()
            return loss

        if not consistent:
            if classifier == 'T':
                prob_weight = F.softmax(weight[:, :self.nClass], dim=1)
                loss = - (prob_weight * ((prob_input[:, self.nClass:] + temp[:, self.nClass:]).log())).sum(1).mean()
            elif classifier == 'S':
                prob_weight = F.softmax(weight[:, self.nClass:], dim=1)
                loss = - (prob_weight * ((prob_input[:, :self.nClass] + temp[:, :self.nClass]).log())).sum(1).mean()
        else:
            if classifier == 'T':
                prob_weight = F.softmax(weight[:, self.nClass:], dim=1)
                loss = - (prob_weight * ((prob_input[:, self.nClass:] + temp[:, self.nClass:]).log())).sum(1).mean()
            elif classifier == 'S':
                prob_weight = F.softmax(weight[:, :self.nClass], dim=1)
                loss = - (prob_weight * ((prob_input[:, :self.nClass] + temp[:, :self.nClass]).log())).sum(1).mean()
        
        return loss
        