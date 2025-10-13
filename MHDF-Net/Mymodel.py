import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from LCMoE import OURS, Gate
import config
from model_concate_multilabel import MetaSubNet

class mymodel_ours(nn.Module):
    def __init__(self):
        super(mymodel_ours, self).__init__()
        #VGG-16 as the backbone
        vgg = vgg16(pretrained = True)
        self.features = nn.Sequential(*list(vgg.children())[:1][0][:5])
        self.layer1 = nn.Sequential(*list(vgg.children())[:1][0][5:10])
        self.layer2 = nn.Sequential(*list(vgg.children())[:1][0][10:17])
        self.layer3 = nn.Sequential(*list(vgg.children())[:1][0][17:24])
        self.layer4 = nn.Sequential(*list(vgg.children())[:1][0][24:34])

        # Local Cross-modal Mixture-of-Experts (LC-MoE)
        self.comb_feat_maps = 64
        self.comb_channel1 = OURS(self.comb_feat_maps)
        self.comb_channel2 = OURS(self.comb_feat_maps)
        self.comb_channel3 = OURS(self.comb_feat_maps)
        self.comb_channel4 = OURS(self.comb_feat_maps)

        self.globalgate = Gate(branch_num=4, modal_num=4)
        self.MLP_img = nn.Sequential(nn.Linear(2048, 256),
                                     nn.Dropout(0.2),
                                     nn.Linear(256, 128))
        self.meta_subnet = MetaSubNet(81, 128, 128, 0.3)

        # GM-Expert
        self.expert1 = Expert()
        self.expert2 = Expert()
        self.expert3 = Expert()
        self.expert4 = Expert()
        self.classifer = nn.Linear(128, config.num_class)

    def forward(self, x, meta_data):
        # Image Encoder
        x = self.features(x)

        x = self.layer1(x)
        feature_size = x.size()
        x = x.view(x.size(0), self.comb_feat_maps, 112, 56).squeeze(-1)
        # TKCA
        x, local_feature1 = self.comb_channel1(x, meta_data.float())
        x = x.view(feature_size)

        x = self.layer2(x)
        feature_size = x.size()
        x = x.view(x.size(0), self.comb_feat_maps, 56, 56).squeeze(-1)
        x, local_feature2 = self.comb_channel2(x, meta_data.float())
        x = x.view(feature_size)

        x = self.layer3(x)
        feature_size = x.size()
        x = x.view(x.size(0), self.comb_feat_maps, 56, 28).squeeze(-1)
        x, local_feature3 = self.comb_channel3(x, meta_data.float())
        x = x.view(feature_size)

        x = self.layer4(x)
        feature_size = x.size()
        x = x.view(x.size(0), self.comb_feat_maps, 28, 14).squeeze(-1)
        x, local_feature4 = self.comb_channel4(x, meta_data.float())
        x = x.view(feature_size)

        # GM-Gate
        global_feature = torch.cat((local_feature1, local_feature2, local_feature3, local_feature4), dim=1)
        weight = self.globalgate(global_feature)
        w1 = weight[:, 0].view(-1, 1)
        w2 = weight[:, 1].view(-1, 1)
        w3 = weight[:, 2].view(-1, 1)
        w4 = weight[:, 3].view(-1, 1)

        # Global Multi-scale Experts (GM-Expert)
        x1 = self.expert1(global_feature)
        x2 = self.expert2(global_feature)
        x3 = self.expert3(global_feature)
        x4 = self.expert4(global_feature)

        x = w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4

        x = self.classifer(x)

        return x


class Expert(nn.Module):
    def __init__(self):
        super(Expert, self).__init__()
        self.expert = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True))
        self.project = nn.Linear(1024, 128)
    def forward(self, x):
         x = self.expert(x)
         x = x.view(x.size(0), -1)
         x = self.project(x)
         return x