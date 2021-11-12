# -*- coding: utf-8 -*

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import gc
from typing import Optional, List, Tuple



def MI(outputs_target):
    batch_size = outputs_target.size(0)
    softmax_outs_t = nn.Softmax(dim=1)(outputs_target)
    avg_softmax_outs_t = torch.sum(softmax_outs_t, dim=0) / float(batch_size)
    log_avg_softmax_outs_t = torch.log(avg_softmax_outs_t + 1e-7)
    item1 = -torch.sum(avg_softmax_outs_t * log_avg_softmax_outs_t)
    item2 = -torch.sum(softmax_outs_t * torch.log(softmax_outs_t + 1e-7)) / float(batch_size)
    return item1 - item2



class EstimatorMean():
    def __init__(self, feature_num, class_num, alpha=0.1):
        super(EstimatorMean, self).__init__()
        self.class_num = class_num
        self.Ave = torch.zeros(class_num, feature_num).cuda()
        self.Amount = torch.zeros(class_num).cuda()
        self.alpha = alpha
        self.t = 1

    def update_Mean(self, features, labels):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        NxCxFeatures = features.view(N, 1, A).expand(N, C, A)
        onehot = torch.zeros(N, C).cuda()
        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        if self.t == 1:
            self.Ave = ave_CxA.detach()
        else:
            self.Ave = (self.alpha * self.Ave + (1 - self.alpha) * ave_CxA).detach()
        self.t = self.t + 1



class EstimatorCV():
    def __init__(self, feature_num, class_num, alpha=0.3):
        super(EstimatorCV, self).__init__()
        self.class_num = class_num
        self.CoVariance = torch.zeros(class_num, feature_num, feature_num).cuda()
        self.Ave = torch.zeros(class_num, feature_num).cuda()
        self.Amount = torch.zeros(class_num).cuda()
        self.alpha = alpha
        self.V1 = torch.zeros(class_num, feature_num, feature_num).cuda()
        self.U1 = torch.zeros(class_num, feature_num, feature_num).cuda()
        self.U2 = torch.zeros(class_num, feature_num).cuda()
        self.initial_Ave = torch.zeros(class_num, feature_num).cuda()
        self.initial_covariance = torch.zeros(class_num, feature_num, feature_num).cuda()
        self.t = 1

    def update_CV(self, features, labels):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        NxCxFeatures = features.view(N, 1, A).expand(N, C, A)
        onehot = torch.zeros(N, C).cuda()
        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = torch.bmm(var_temp.permute(1, 2, 0), var_temp.permute(1, 0, 2)).div(Amount_CxA.view(C, A, 1).expand(C, A, A))

        self.V1 = self.alpha * self.V1 + var_temp
        self.U1 = self.alpha * self.U1 + torch.bmm(ave_CxA.view(C, A, 1), ave_CxA.view(C, 1, A))
        self.U2 = self.alpha * self.U2 + ave_CxA

        if self.t == 1:
            self.Ave = ave_CxA.detach()
            self.CoVariance = var_temp.detach()
            self.initial_covariance = var_temp.detach()
            self.initial_Ave = ave_CxA.detach()
        else:
            self.Ave = (self.alpha * self.Ave + (1 - self.alpha) * ave_CxA).detach()
            A1 = (self.alpha ** self.t) * self.initial_covariance
            A2 = (1 - self.alpha) * self.V1
            A3 = (self.alpha ** self.t) * torch.bmm((self.initial_Ave).view(C, A, 1), (self.initial_Ave).view(C, 1, A))
            A4 = (1 - self.alpha) * self.U1
            A5 = torch.bmm(self.Ave.view(C, A, 1), self.Ave.view(C, 1, A))
            A6 = (self.alpha ** self.t) * torch.bmm((self.initial_Ave).view(C, A, 1), (self.Ave).view(C, 1, A)) + \
                 (1 - self.alpha) * torch.bmm((self.U2).view(C, A, 1), (self.Ave).view(C, 1, A))
            self.CoVariance = (A1 + A2 + A3 + A4 + A5 - (A6 + A6.transpose(1, 2))).detach()

        self.t = self.t + 1



class Loss_aug_pro(nn.Module):
    def __init__(self, feature_num, class_num, alpha):
        super(Loss_aug_pro, self).__init__()
        self.target_estimator = EstimatorCV(feature_num, class_num, alpha)
        self.source_estimator_mean = EstimatorMean(feature_num, class_num, alpha)
        self.class_num = class_num
        self.cross_entropy = nn.CrossEntropyLoss()

    def aug_s(self, s_mean_matrix, t_mean_matrix, fc, f_s, y_s, labels_s, t_cv_matrix, Lambda, margin):
        N = f_s.size(0)
        C = self.class_num
        A = f_s.size(1)

        weight_m = list(fc.parameters())[0]
        assert weight_m.shape[0] == C
        assert weight_m.shape[1] == A

        normalized_weight_m = F.normalize(weight_m, p=2, dim=1)

        NxW_ij = normalized_weight_m.expand(N, C, A)
        NxW_kj = torch.gather(NxW_ij, 1, labels_s.view(N, 1, 1).expand(N, C, A))

        t_CV_temp = t_cv_matrix[labels_s]

        sigma2 = Lambda * torch.bmm(torch.bmm(NxW_ij - NxW_kj, t_CV_temp), (NxW_ij - NxW_kj).permute(0, 2, 1))
        sigma2 = sigma2.mul(torch.eye(C).cuda().expand(N, C, C)).sum(2).view(N, C)

        sourceMean_NxA = s_mean_matrix[labels_s]
        targetMean_NxA = t_mean_matrix[labels_s]
        dataMean_NxA = (targetMean_NxA - sourceMean_NxA)
        dataMean_NxAx1 = dataMean_NxA.expand(1, N, A).permute(1, 2, 0)

        del t_CV_temp, sourceMean_NxA, targetMean_NxA, dataMean_NxA
        gc.collect()

        dataW_NxCxA = NxW_ij - NxW_kj
        dataW_x_detaMean_NxCx1 = torch.bmm(dataW_NxCxA, dataMean_NxAx1)
        datW_x_detaMean_NxC = dataW_x_detaMean_NxCx1.view(N, C)

        aug_result = y_s + 0.5 * sigma2 + Lambda * datW_x_detaMean_NxC

        one_hot = torch.zeros_like(y_s)
        one_hot.scatter_(1, labels_s.view(-1, 1), 1.0)
        margin_aug_result = aug_result - one_hot * margin

        return margin_aug_result

    def PrototypeConstraint(self, class_num, f_s, f_t, labels_s, pseudo_labels_t, s_mean_matrix, t_mean_matrix, t_cv_matrix, Lambda):

        n_s = f_s.shape[0]
        n_t = f_t.shape[0]
        A = f_s.shape[1]
        C = class_num

        # class-wisely calculate the center of target samples
        NxCxFeaturesT = f_t.view(n_t, 1, A).expand(n_t, C, A)
        onehotT = torch.zeros(n_t, C).cuda()
        onehotT.scatter_(1, pseudo_labels_t.view(-1, 1), 1)

        NxCxA_onehotT = onehotT.view(n_t, C, 1).expand(n_t, C, A)
        featuresT_by_sort = NxCxFeaturesT.mul(NxCxA_onehotT)
        AmountT_CxA = NxCxA_onehotT.sum(0)
        flag = AmountT_CxA[:, 0]
        AmountT_CxA[AmountT_CxA == 0] = 1
        aveT_CxA = featuresT_by_sort.sum(0) / AmountT_CxA

        count = 0
        aug_align_loss = 0.0
        for i in range(n_s):
            c = labels_s[i]
            if flag[c] == 0:
                continue
            count = count + 1
            mean_matrix = f_s[i] + Lambda * (t_mean_matrix[c] - s_mean_matrix[c])
            variance_matrix = Lambda * torch.diagonal(t_cv_matrix[c], dim1=-2, dim2=-1)
            aug_align_loss = aug_align_loss + torch.sum(torch.pow(mean_matrix - aveT_CxA[c], 2) + variance_matrix)

        if count == 0:
            return aug_align_loss
        else:
            aug_align_loss = aug_align_loss / float(count)
            return aug_align_loss

    def forward(self, fc, f_s, f_t, y_s, labels_source, pseudo_label_target, Lambda, margin):
        self.target_estimator.update_CV(f_t.detach(), pseudo_label_target)   # update target covariance and target mean
        self.source_estimator_mean.update_Mean(f_s.detach(), labels_source)  # update source mean

        aug_y_s = self.aug_s(self.source_estimator_mean.Ave.detach(), self.target_estimator.Ave.detach(), fc, \
                             f_s, y_s, labels_source, self.target_estimator.CoVariance.detach(), Lambda, margin)

        aug_loss = self.cross_entropy(aug_y_s, labels_source)

        pro_loss = self.PrototypeConstraint(y_s.shape[1], f_s, f_t, labels_source, pseudo_label_target, \
                                            self.source_estimator_mean.Ave.detach(), self.target_estimator.Ave.detach(),\
                                            self.target_estimator.CoVariance.detach(), Lambda)

        return aug_loss, pro_loss




