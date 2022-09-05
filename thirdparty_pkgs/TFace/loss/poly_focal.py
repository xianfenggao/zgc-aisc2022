import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    """ Implementaion of "https://arxiv.org/abs/1708.02002"
    """

    def __init__(self, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.func = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.func(input, target)
        prob = torch.exp(-logp)
        loss = (1 - prob) ** self.gamma * logp
        return loss

class PolyFocalLoss(nn.Module):
    """ Implementaion of "https://openreview.net/forum?id=gSdSJoenupI"
    """

    def __init__(self, epsilon=1.0, gamma=2):
        super(PolyFocalLoss, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.func = FocalLoss(gamma)

    def forward(self, input, target):
        p = torch.sigmoid(input)
        pt = target.unsqueeze(1) * p + (1 - target.unsqueeze(1)) * (1 - p)
        FL = self.func(pt, target)
        Poly1 = FL + self.epsilon * torch.pow(1 - pt, self.gamma + 1)
        return Poly1.mean()
