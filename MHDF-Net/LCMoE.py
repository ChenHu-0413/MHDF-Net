import torch.nn as nn
import torch
import numpy as np
from mca import CrossTransformer_meta
from model_concate_multilabel import MetaSubNet
import torch.nn.functional as F
# LC-MoE
class OURS(nn.Module):

    def __init__(self, out_dim):
        super(OURS, self).__init__()

        self.weight1 = nn.Parameter(torch.rand(1))
        self.weight2 = nn.Parameter(torch.rand(1))
        self.meta_subnet = MetaSubNet(81, 128, 128, 0.3)
        #  Top-k Cross-Attention (TKCA)
        self.fusion_meta = CrossTransformer_meta(x_dim=128, c_dim=128, depth=1, num_heads=8)
        self.fusion_img = CrossTransformer_meta(x_dim=128, c_dim=128, depth=1, num_heads=8)
        self.localgate = Gate(branch_num=2)

        self.convblock = nn.Sequential(
            nn.Conv2d(128*2, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, out_dim, 1),
            nn.Sigmoid()
        )
        self.depthwise_separable_conv = nn.Sequential(
            nn.Conv2d(16, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat_maps, metadata1):

        device = feat_maps.device
        feat_maps_transformed = self.depthwise_separable_conv(feat_maps)

        global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        feat_maps_final = global_avg_pool(feat_maps_transformed).view(feat_maps_transformed.size(0),-1)
        meta_h = self.meta_subnet(metadata1.to(device))

        #  Top-k Cross-Attention (TKCA)
        metadata = self.fusion_meta(meta_h, feat_maps_final)
        img_logit = self.fusion_img(feat_maps_final, meta_h)

        metadata = torch.unsqueeze(metadata, -1)
        metadata = torch.unsqueeze(metadata, -1)
        img_logit = torch.unsqueeze(img_logit, -1)
        img_logit = torch.unsqueeze(img_logit, -1)

        att = torch.cat([metadata, img_logit], dim=1)

        #  LC-Gate
        weight = self.localgate(att)
        w1 = weight[:, 0].view(-1, 1, 1, 1)
        w2 = weight[:, 1].view(-1, 1, 1, 1)
        local_feature = w1*img_logit + w2*(img_logit + metadata)

        att = self.convblock(att)
        x = att * feat_maps

        return x, local_feature

class Gate(nn.Module):
    def __init__(self, branch_num, hidden_dim=8, modal_num=2):
        super(Gate, self).__init__()
        self.bnum = branch_num
        self.conv = nn.Sequential(
            nn.Conv2d(128*modal_num, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.Tanh())
        self.fc = nn.Conv2d(hidden_dim, self.bnum, kernel_size=1, bias=False)

    def forward(self, x, temp=1.0, hard=False):

        y = self.conv(x)
        y = F.adaptive_avg_pool2d(y, 1)
        y = self.fc(y)
        y = DiffSoftmax(y, tau=temp, hard=hard, dim=1)
        y = y.squeeze(-1).squeeze(-1)
        return y

def DiffSoftmax(logits, tau=1.0, hard=False, dim=-1):
    y_soft = (logits / tau).softmax(dim)
    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

