import torch.nn as nn
import torch.nn.functional as F


class DomainAdvLoss(nn.Module):

    def __init__(self, nClass=10):
        super(DomainAdvLoss, self).__init__()
        self.nClass = nClass
    
    def forward(self, input, classifier='T'):
        """
        Arguments:
            input: a float tensor, score vector of classifier predictions.
            classifier: can optionally use the source or target classifier predictions to compute the domain adversarial loss. Default: 'T'.
        Returns:
            a float tensor with shape [1], computed loss.
        """     
        prob_input = F.softmax(input, dim=1)
        if classifier == 'T':
            loss = - prob_input[:, self.nClass:].sum(1).log().mean()
        elif classifier == 'S':
            loss = - prob_input[:, :self.nClass].sum(1).log().mean()
        
        return loss
    