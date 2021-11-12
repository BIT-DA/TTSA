import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, in_dim, num_classes, bottleneck_dim=256, bias=False):
        super(Classifier, self).__init__()
        self.in_dim = in_dim
        self.bottleneck_dim = bottleneck_dim
        self.num_classes = num_classes

        self.bottleneck = nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )

        weight = torch.FloatTensor(num_classes, self.bottleneck_dim).normal_(0.0, np.sqrt(2.0 / self.bottleneck_dim))
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x):
        features = self.bottleneck(x)
        weight = F.normalize(self.weight, p=2, dim=1, eps=1e-12)
        scores = torch.mm(features, weight.t())
        return scores, features

