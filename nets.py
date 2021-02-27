
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from blocks import ConvsBlock, UpConvLayer
from attentions import ChannelSpatialAttention

efficientnet_encoder_blocks = {
    "efficientnet-b2": [1, 4, 7, 15], 
    "efficientnet-b3": [1, 4, 7, 17], 
    "efficientnet-b4": [1, 5, 9, 21]
}

class EfficientUNetPP(nn.Module):
    def __init__(self, encoder, encoder_trainable, n_classes):
        super().__init__()
        self.effnet = EfficientNet.from_pretrained(encoder)
        for param in self.effnet.parameters():
            param.requires_grad = encoder_trainable
        self.encoder_blocks = efficientnet_encoder_blocks[encoder]

        self.upconv1 = UpConvLayer(self.effnet._blocks[-1]._bn2.num_features, self.effnet._blocks[self.encoder_blocks[-1]]._bn2.num_features)
        self.upconv2 = UpConvLayer(self.effnet._blocks[self.encoder_blocks[-1]]._bn2.num_features, self.effnet._blocks[self.encoder_blocks[-2]]._bn2.num_features)
        self.upconv3 = UpConvLayer(self.effnet._blocks[self.encoder_blocks[-2]]._bn2.num_features, self.effnet._blocks[self.encoder_blocks[-3]]._bn2.num_features)
        self.upconv4 = UpConvLayer(self.effnet._blocks[self.encoder_blocks[-3]]._bn2.num_features, self.effnet._blocks[self.encoder_blocks[-4]]._bn2.num_features)
        self.decoder1 = ConvsBlock(2*self.effnet._blocks[self.encoder_blocks[-1]]._bn2.num_features, self.effnet._blocks[self.encoder_blocks[-1]]._bn2.num_features)
        self.decoder2 = ConvsBlock(2*self.effnet._blocks[self.encoder_blocks[-2]]._bn2.num_features, self.effnet._blocks[self.encoder_blocks[-2]]._bn2.num_features)
        self.decoder3 = ConvsBlock(2*self.effnet._blocks[self.encoder_blocks[-3]]._bn2.num_features, self.effnet._blocks[self.encoder_blocks[-3]]._bn2.num_features)
        self.decoder4 = ConvsBlock(2*self.effnet._blocks[self.encoder_blocks[-4]]._bn2.num_features, self.effnet._blocks[self.encoder_blocks[-4]]._bn2.num_features)
        
     


        self.upconv5 = UpConvLayer(self.effnet._blocks[self.encoder_blocks[-4]]._bn2.num_features, self.effnet._blocks[self.encoder_blocks[-4]]._bn2.num_features // 2)
        self.decoder5 = ConvsBlock(self.effnet._blocks[self.encoder_blocks[-4]]._bn2.num_features // 2, self.effnet._blocks[self.encoder_blocks[-4]]._bn2.num_features // 2)

        self.segmentor3 = nn.Conv2d(self.effnet._blocks[self.encoder_blocks[-3]]._bn2.num_features, n_classes, kernel_size=1)
        self.segmentor4 = nn.Conv2d(self.effnet._blocks[self.encoder_blocks[-4]]._bn2.num_features, n_classes, kernel_size=1)
        self.segmentor5 = nn.Conv2d(self.effnet._blocks[self.encoder_blocks[-4]]._bn2.num_features // 2, n_classes, kernel_size=1)

    def forward(self, x):
        encoder_features = []

        x = self.effnet._swish(self.effnet._bn0(self.effnet._conv_stem(x)))
        for idx, block in enumerate(self.effnet._blocks):
            x = block(x)
            if idx in self.encoder_blocks:
                encoder_features.append(x)
        
        x4 = x

        x3 = encoder_features[-1]
        x3_ = self.upconv1(x4)
        x3_ = torch.cat([encoder_features[-1],x3_], dim=1)
        x3_ = self.decoder1(x3_)

        x2 = encoder_features[-2]
        x2_ = self.upconv2(x3_)
        x21 = self.upconv2(x3)
        x21 = torch.cat([x3,x21], dim=1)
        x21 = self.decoder2(x21)
        x2_ = torch.cat([x21,x2_], dim=1)
        x2_ = self.decoder2(x2_)

        x1 = encoder_features[-3]
        x1_ = self.upconv3(x2_)
        x11 = self.upconv3(x2)
        x11 = torch.cat([x2,x11], dim=1)
        x11 = self.decoder3(x11)
        x12 = self.upconv3(x21)
        x12 = torch.cat([x11,x12], dim=1)
        x12 = self.decoder3(x12)
        x1_ = torch.cat([x12,x1_], dim=1)
        x1_ = self.decoder3(x1_)
        out3 = self.segmentor3(x1_)

        x0 = encoder_features[-4]
        x0_ = self.upconv4(x1_)
        x01 = self.upconv4(x1)
        x01 = torch.cat([x1,x01], dim=1)
        x01 = self.decoder4(x01)
        x02 = self.upconv4(x11)
        x02 = torch.cat([x01,x02], dim=1)
        x02 = self.decoder4(x02)
        x03 = self.upconv4(x12)
        x03 = torch.cat([x02,x03], dim=1)
        x03 = self.decoder4(x03)
        x0_ = torch.cat([x03,x0_], dim=1)
        x0_ = self.decoder4(x0_)
        out4 = self.segmentor4(x)
        
        y0 = torch.cat([x01,x02,x03,x0_],dim=1)
        y0 = self.upconv5(y0)
        y0 = self.decoder5(y0)
        out5 = self.segmentor5(y0)
        # out5 = self.segmentor5(y0)

        # x = self.upconv1(x)
        # x = torch.cat([encoder_features[-1], x], dim=1)
        # x = self.decoder1(x)

        # x = self.upconv2(x)
        # x = torch.cat([encoder_features[-2], x], dim=1)
        # x = self.decoder2(x)

        # x = self.upconv3(x)
        # x = torch.cat([encoder_features[-3], x], dim=1)
        # x = self.decoder3(x)
        # out3 = self.segmentor3(x)

        # x = self.upconv4(x)
        # x = torch.cat([encoder_features[-4], x], dim=1)
        # x = self.decoder4(x)
        # out4 = self.segmentor4(x)

        # x = self.upconv5(x)
        # x = self.decoder5(x)
        # out5 = self.segmentor5(x)

        