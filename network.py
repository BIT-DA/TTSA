from typing import Optional, List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ClassifierBase(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck: Optional[nn.Module] = None,
                 bottleneck_dim: Optional[int] = -1, head: Optional[nn.Module] = None):
        super(ClassifierBase, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        if bottleneck is None:
            self.bottleneck = nn.Identity()
            self._features_dim = backbone.out_features
        else:
            self.bottleneck = bottleneck
            assert bottleneck_dim > 0
            self._features_dim = bottleneck_dim
        if head is None:
            self.head = nn.Linear(self._features_dim, num_classes, bias=False)
        else:
            self.head = head

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        f = self.backbone(x)
        f = f.view(-1, self.backbone.out_features)
        f = self.bottleneck(f)

        weights = list(self.head.parameters())[0]
        assert weights.shape[0] == self.num_classes
        assert weights.shape[1] == self.features_dim
        weights_norm = torch.norm(weights, p=2, dim=1)
        weights_norm_NxC = weights_norm.expand(f.shape[0], weights_norm.shape[0])
        normalized_predictions = self.head(f) / weights_norm_NxC
        scale_normalized_predictions = normalized_predictions

        return scale_normalized_predictions, f

    def get_parameters(self) -> List[Dict]:
        params = [
            {"params": self.backbone.parameters(), "lr_mult": 0.1},
            {"params": self.bottleneck.parameters(), "lr_mult": 1.},
            {"params": self.head.parameters(), "lr_mult": 1.},
        ]
        return params

class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256):
        bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim)




