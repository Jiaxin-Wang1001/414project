# Originaly developed by Haozhe Xie <cshzxie@gmail.com>
# Modified by Jiaxin Wang, Senyu Li, Tianying Xia

import torch
import torch.nn.functional as F

class Decoder2(torch.nn.Module):
    def __init__(self, cfg):
        super(Decoder2, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(1568, 512, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(512),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(512, 128, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 32, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 1, kernel_size=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
            torch.nn.Sigmoid()
        )
        
        self.layerx = torch.nn.MaxPool3d((32,1,1), 1)
        self.layery = torch.nn.MaxPool3d((1,32,1), 1)
        self.layerz = torch.nn.MaxPool3d((1,1,32), 1)

        self.layer6 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 3, kernel_size=3, stride=4, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, image_features):
        image_features = image_features.permute(1, 0, 2, 3, 4).contiguous()
        image_features = torch.split(image_features, 1, dim=0)
        gen_volumes = []
        projections = []

        for features in image_features:
            gen_volume = features.view(-1, 1568, 2, 2, 2)
            # print(gen_volume.size())   # torch.Size([batch_size, 1568, 2, 2, 2])
            gen_volume = self.layer1(gen_volume)
            # print(gen_volume.size())   # torch.Size([batch_size, 512, 4, 4, 4])
            gen_volume = self.layer2(gen_volume)
            # print(gen_volume.size())   # torch.Size([batch_size, 128, 8, 8, 8])
            gen_volume = self.layer3(gen_volume)
            # print(gen_volume.size())   # torch.Size([batch_size, 32, 16, 16, 16])
            gen_volume = self.layer4(gen_volume)
            raw_feature = gen_volume
            temp = torch.reshape(raw_feature, (raw_feature.shape[0], 16, 128, 128))
            projections.append(self.layer6(temp))
            # projections.append(self.layer6(temp))
            # print(gen_volume.size())   # torch.Size([batch_size, 8, 32, 32, 32])
            # gen_volume = F.threshold(self.layer5(gen_volume), 0.4, 0)
            gen_volume = self.layer5(gen_volume)
            # gen_volume = self.layer5(gen_volume)
            # print(gen_volume.size())   # torch.Size([batch_size, 1, 32, 32, 32])
            gen_volumes.append(torch.squeeze(gen_volume, dim=1))
            

            
        projections = torch.stack(projections).contiguous()
        projections=torch.squeeze(projections,dim=0)

        gen_volumes = torch.stack(gen_volumes).permute(1, 0, 2, 3, 4).contiguous()
        # print(gen_volumes.size())      # torch.Size([batch_size, n_views, 32, 32, 32])
        return gen_volumes, projections
