import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader
import itertools
import matplotlib.pyplot as plt
# 查看时间和进度
from tqdm import tqdm
import time

class ArcFaceNet(nn.Module):
    def __init__(self, cls_num=10, feature_dim=2):
        super(ArcFaceNet, self).__init__()
        self.w = nn.Parameter(torch.randn(feature_dim, cls_num))

    def forward(self, features, m=1, s=10):
        # 特征与权重 归一化
        _features = nn.functional.normalize(features)
        _w = nn.functional.normalize(self.w)
        print('_features', _features.shape)
        print('_w', _w.shape)
        # 特征向量与参数向量的夹角theta，分子numerator，分母denominator
        theta = torch.acos(torch.matmul(_features, _w) / 10)  # /10防止下溢
        print(theta.shape)
        numerator = torch.exp(s * torch.cos(theta + m))
        denominator = torch.sum(torch.exp(s * torch.cos(theta)), dim=1, keepdim=True) - torch.exp(
            s * torch.cos(theta)) + numerator
        return torch.log(torch.div(numerator, denominator))


class CenterLossNet(nn.Module):
    def __init__(self, cls_num=10, feature_dim=2):
        super(CenterLossNet, self).__init__()
        self.centers = nn.Parameter(torch.randn(cls_num, feature_dim))

    def forward(self, features, labels, reduction='mean'):
        # 特征向量归一化
        _features = nn.functional.normalize(features)

        centers_batch = self.centers.index_select(dim=0, index=labels.long())
        # 根据论文《A Discriminative Feature Learning Approach for Deep Face Recognition》修改如下
        if reduction == 'sum':  # 返回loss的和
            return torch.sum(torch.pow(_features - centers_batch, 2)) / 2
        elif reduction == 'mean':  # 返回loss和的平均值，默认为mean方式
            return torch.sum(torch.pow(_features - centers_batch, 2)) / 2 / len(features)
        else:
            raise ValueError("ValueError: {0} is not a valid value for reduction".format(reduction))


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          s: norm of input feature
          m: additive angular margin
          cos(theta + m)
      """


    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()

        self.in_features = in_features  # 特征输入通道数
        self.out_features = out_features  # 特征输出通道数
        self.s = s  # 输入特征范数 ||x_i||
        self.m = m  # 加性角度边距 m (additive angular margin)
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))  # FC 权重
        # self.weight = Parameter(torch.FloatTensor(in_features, out_features))  # FC 权重
        nn.init.xavier_uniform_(self.weight)  # Xavier 初始化 FC 权重

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m


    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # 分别归一化输入特征 xi 和 FC 权重 W, 二者点乘得到 cosθ, 即预测值 Logit
        # input = nn.functional.normalize(input, dim=1)
        # weight = nn.functional.normalize(self.weight, dim=0)
        # print('input', input.shape)
        # print('weight', self.weight.shape)
        # print('label', label.shape)
        # theta = torch.acos(torch.matmul(_features, _w) / 10)  # /10防止下溢
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # print('cosine', cosine.shape)
        # 由 cosθ 计算相应的 sinθ
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        # 展开计算 cos(θ+m) = cosθ*cosm - sinθ*sinm, 其中包含了 Target Logit (cos(θyi+ m)) (由于输入特征 xi 的非真实类也参与了计算, 最后计算新 Logit 时需使用 One-Hot 区别)
        phi = cosine * self.cos_m - sine * self.sin_m
        # 是否松弛约束??
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        # 将 labels 转换为独热编码, 用于区分是否为输入特征 xi 对应的真实类别 yi
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        # 计算新 Logit
        #  - 只有输入特征 xi 对应的真实类别 yi (one_hot=1) 采用新 Target Logit cos(θ_yi + m)
        #  - 其余并不对应输入特征 xi 的真实类别的类 (one_hot=0) 则仍保持原 Logit cosθ_j
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # can use torch.where if torch.__version__  > 0.4
        # 使用 s rescale 放缩新 Logit, 以馈入传统 Softmax Loss 计算
        output *= self.s

        return output