import torch
import torch.nn as nn
import timm

from torchsummary import summary


class LocNet(nn.Module):
    def __init__(self):
        # load and modify pretrained model to have more stages
        m = timm.create_model("efficientnet_b0", pretrained=True)
        list(m.blocks[0][0].children())[0].stride = (2, 2)
        for i in range(1, len(m.blocks)):
            list(m.blocks[i][0].children())[2].stride = (2, 2)

        super().__init__()
        self.block0 = nn.Sequential(m.conv_stem, m.bn1, nn.LeakyReLU(inplace=True))
        self.block1 = m.blocks[0]
        self.block2 = m.blocks[1]
        self.block3 = m.blocks[2]
        self.block4 = m.blocks[3]
        self.block5 = m.blocks[4]
        self.block6 = m.blocks[5]
        self.block7 = m.blocks[6]

    def forward(self, x):
        x = stage0 = self.block0(x)
        x = stage1 = self.block1(x)
        x = stage2 = self.block2(x)
        x = stage3 = self.block3(x)
        x = stage4 = self.block4(x)
        x = stage5 = self.block5(x)
        x = stage6 = self.block6(x)
        x = stage7 = self.block7(x)

        # print(stage0.size(), stage1.size(), stage2.size(), stage3.size(), stage4.size(), stage5.size(), stage6.size(), stage7.size())
        # torch.Size([1, 64, 512, 512]) torch.Size([1, 32, 256, 256]) torch.Size([1, 48, 128, 128]) torch.Size([1, 80, 64, 64]) torch.Size([1, 80, 32, 32]) torch.Size([1, 112, 16, 16]) torch.Size([1, 192, 8, 8]) torch.Size([1, 320, 4, 4])
        # channels in each stage representation (stage0 -> stage7):  32, 16, 24, 40, 80, 112, 192, 320
        return stage7, stage6, stage5, stage4, stage3, stage2, stage1, stage0


def ConvNormAct(in_channels, out_channels, ksize=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, ksize, 1, ksize // 2),
        # nn.BatchNorm2d(out_channels), # saves memory to ignore batchnorm in decoder
        nn.LeakyReLU(inplace=True),
    )


class LocUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = LocNet()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.h7 = ConvNormAct(320, 256, 3)
        self.h6 = ConvNormAct(256 + 192, 256)
        self.h5 = ConvNormAct(256 + 112, 128)
        self.h4 = ConvNormAct(128 + 80, 128)
        self.h3 = ConvNormAct(128 + 40, 64)
        self.h2 = ConvNormAct(64 + 24, 64)
        self.h1 = ConvNormAct(64 + 16, 32)
        self.h0 = ConvNormAct(32 + 32, 32)
        self.finalconv = nn.Conv2d(32, 4, 1, 1)

    def forward(self, x):
        stage7, stage6, stage5, stage4, stage3, stage2, stage1, stage0 = self.encoder(x)
        x = self.h7(stage7)
        x = self.upsample(x)
        x = self.h6(torch.cat([x, stage6.detach()], dim=1))
        x = self.upsample(x)
        x = self.h5(torch.cat([x, stage5.detach()], dim=1))
        x = self.upsample(x)
        x = self.h4(torch.cat([x, stage4.detach()], dim=1))
        x = self.upsample(x)
        x = self.h3(torch.cat([x, stage3.detach()], dim=1))
        x = self.upsample(x)
        x = self.h2(torch.cat([x, stage2.detach()], dim=1))
        x = self.upsample(x)
        x = self.h1(torch.cat([x, stage1.detach()], dim=1))
        x = self.upsample(x)
        x = self.h0(torch.cat([x, stage0.detach()], dim=1))
        x = self.finalconv(x)
        x = self.upsample(x)

        return x


def test():
    print(summary(LocUNet().cuda(), (3, 1024, 1024)))


if __name__ == "__main__":
    test()
