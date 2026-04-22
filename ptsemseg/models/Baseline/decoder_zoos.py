import torch
from torch import nn
# from torchsummaryX import summary
import torch.nn.functional as F

'''
https://github.com/suniique/Leveraging-Crowdsourced-GPS-Data-for-Road-Extraction-from-Aerial-Imagery/blob/master/networks/basic_blocks.py
'''

class decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder, self).__init__()
        # 反卷积
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.up_conv = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x_copy, x, interpolate=True):
        out = self.up(x)
        if interpolate:
            # 迭代代替填充， 取得更好的结果
            out = F.interpolate(out, size=(x_copy.size(2), x_copy.size(3)),
                                mode="bilinear", align_corners=True
                                )
        else:
            # 如果填充物体积大小不同
            diffY = x_copy.size()[2] - x.size()[2]
            diffX = x_copy.size()[3] - x.size()[3]
            out = F.pad(out, (diffX // 2, diffX - diffX // 2, diffY, diffY - diffY // 2))
        
        # 连接
        # print(x_copy.shape, out.shape)   # 4, 320, 32, 32
        out = torch.cat([x_copy, out], dim=1)
        # print(out.shape)   # 4, 320, 32, 32
        out_conv = self.up_conv(out)
        return out_conv


class DecoderBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(DecoderBlock, self).__init__()

        self.conv1=nn.Conv2d(ch_in,ch_in//4,kernel_size=1,bias=False)
        self.bn1=nn.BatchNorm2d(ch_in//4)
        self.relu1=nn.ReLU(inplace=True)

        self.deconv2=nn.ConvTranspose2d(ch_in//4,ch_in//4,kernel_size=4,stride=2,padding=1)
        self.bn2=nn.BatchNorm2d(ch_in//4)
        self.relu2=nn.ReLU(inplace=True)

        self.conv3=nn.Conv2d(ch_in//4,ch_out,kernel_size=1,bias=False)
        self.bn3=nn.BatchNorm2d(ch_out)
        self.relu3=nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        return x


if __name__ == '__main__':
    # model=SEBlock(128)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # x = torch.randn(1, 128, 64, 64, device=device)
    x = torch.randn(4, 256, 16, 16, device=device)
    y = torch.randn(4, 256, 8, 8, device=device)

    model = DecoderBlock(ch_in=256, ch_out=1).to(device)
    output = model(x)
    print("output", output.shape)

    model = decoder(in_channels=256, out_channels=256).to(device)
    output = model(x, y)
    print("output", output.shape)

    # summary(model.to(device),y)