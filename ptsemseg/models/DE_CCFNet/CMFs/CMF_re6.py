import torch
from torch import nn
# from torchsummaryX import summary
import torch.nn.functional as F

print("come from CFM_re6.py")

class channel_frm(nn.Module):
    def __init__(self, channel, r=16):
        '''
        input x: (b,c,h,w)
        input y: (b,c,h,w)
        :param channel:  c
        :param r: ratio
        '''
        super(channel_frm, self).__init__()
        self.ch=channel
        self.x_avg = nn.AdaptiveAvgPool2d(1)
        self.x_max = nn.AdaptiveMaxPool2d(1)
        self.x_excitation = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            # nn.Sigmoid()
        )
        self.x_conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        self.y_avg = nn.AdaptiveAvgPool2d(1)
        self.y_max = nn.AdaptiveMaxPool2d(1)
        self.y_excitation=nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            # nn.Sigmoid()
        )
        self.y_conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.sigmoid = nn.Sigmoid()
        # self.relu=nn.ReLU(inplace=True)
        # self.x_a=Parameter(torch.zeros(1))
        # self.y_a = Parameter(torch.zeros(1))
        # self.x_b = Parameter(torch.zeros(1))
        # self.y_b=Parameter(torch.zeros(1))

    def forward(self, x, y):
        b, c, _, _ = x.size()
        x_avg = self.x_excitation(self.x_avg(x).view(b, c))
        x_max = self.x_excitation(self.x_max(x).view(b, c))
        # print(x_avg.shape, x_max.shape)    # [4, 256, 64, 64]  - [4, 256]
        x_theta = self.sigmoid(x_avg + x_max).view(b, c, 1, 1)
        x_theta_inv = 1 - x_theta
        x_att = x * x_theta

        y_avg = self.y_excitation(self.y_avg(x).view(b,c))
        y_max = self.y_excitation(self.y_max(x).view(b, c))
        y_theta = self.sigmoid(y_avg+y_max).view(b,c,1,1)
        y_theta_inv = 1 - y_theta
        y_att = y * y_theta

        x_add = self.y_conv(y_att) * x_theta_inv
        y_add = self.x_conv(x_att) * y_theta_inv

        x_chre = x + x_att + x_add
        y_chre = y + y_att + y_add
        # x_chre=self.relu(x+x_att+x_add)
        # y_chre=self.relu(y+y_att+y_add)

        return x_chre, y_chre

class spatial_frm(nn.Module):
    def __init__(self, channel):
        super(spatial_frm, self).__init__()
        self.ch = channel
        self.gate_x = nn.Conv2d(channel*2, 1, kernel_size=1, bias=True)
        self.gate_y = nn.Conv2d(channel*2, 1, kernel_size=1, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):
        # print("x", x.shape, "y", y.shape)
        cat_fea = torch.cat([x, y], dim=1)
        # print("cat_fea", cat_fea.shape)
        attention_vector_l = self.gate_x(cat_fea)
        attention_vector_r = self.gate_y(cat_fea)

        attention_vector = torch.cat([attention_vector_l, attention_vector_r], dim=1)
        attention_vector = self.softmax(attention_vector)
        attention_vector_l, attention_vector_r = \
                                        attention_vector[:, 0:1, :, :], attention_vector[:, 1:2, :, :]
        merge_feature = x * attention_vector_l + y * attention_vector_r

        return merge_feature
        # return attention_vector_l,attention_vector_r

class CMF_re6_2(nn.Module):
    def __init__(self,channel, filters, r=16):
        super(CMF_re6_2, self).__init__()
        self.ch_frm = channel_frm(channel, r)
        self.sp_frm = spatial_frm(channel)
        self.conv = nn.Sequential(
            nn.Conv2d(filters, channel, kernel_size=1, bias=False),
            # nn.Conv2d(filters,filters,kernel_size=3,stride=2,padding=1,groups=filters),
            # nn.Conv2d(filters,channel,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(channel),
            # nn.ReLU(inplace=True)
        )

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x, y, fused=None):
        _, _, h, w = x .size()
        x_chre, y_chre = self.ch_frm(x, y)
        # attention_vector_l,attention_vector_r=self.sp_frm(x_chre,y_chre)
        merge_feature = self.sp_frm(x_chre, y_chre)
        # attention_vector_l, attention_vector_r = self.sp_frm(x, y)
        # merge_feature = x * attention_vector_l + y * attention_vector_r
        x_out = (x + merge_feature) / 2
        y_out = (y + merge_feature) / 2
        x_out = self.relu1(x_out)
        y_out = self.relu2(y_out)

        if fused is not None:
            fused = self.conv(fused)
            merge_feature = merge_feature + F.interpolate(fused,(h, w))
            merge_feature = self.relu3(merge_feature)
        else:
            merge_feature = self.relu3(merge_feature)

        return x_out, y_out, merge_feature


class CMF_re6(nn.Module):
    def __init__(self, channel, filters, r=16):
        super(CMF_re6, self).__init__()
        self.ch_frm = channel_frm(channel, r)
        self.sp_frm = spatial_frm(channel)
        self.conv = nn.Sequential(
            nn.Conv2d(filters,channel,kernel_size=1,bias=False),
            # nn.Conv2d(filters,filters,kernel_size=3,stride=2,padding=1,groups=filters),
            # nn.Conv2d(filters,channel,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(channel),
            # nn.ReLU(inplace=True)
        )

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x, y, fused=None):
        _,_,h,w = x.size()
        x_chre, y_chre = self.ch_frm(x, y)
        # attention_vector_l,attention_vector_r=self.sp_frm(x_chre,y_chre)
        merge_feature = self.sp_frm(x_chre, y_chre)
        # attention_vector_l, attention_vector_r = self.sp_frm(x, y)
        # merge_feature = x * attention_vector_l + y * attention_vector_r

        if fused is not None:
            fused = self.conv(fused)
            merge_feature = merge_feature + F.interpolate(fused,(h, w))
            merge_feature = self.relu3(merge_feature)
        else:
            merge_feature = self.relu3(merge_feature)

        x_out = (x + merge_feature) / 2
        y_out = (y + merge_feature) / 2
        x_out = self.relu1(x_out)
        y_out = self.relu2(y_out)

        return x_out, y_out, merge_feature


if __name__=="__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(4, 256, 64, 64, device=device)
    y = torch.randn(4, 256, 64, 64, device=device)
    fusedc = torch.randn(4, 128, 128, 128, device=device)
    
    model = CMF_re6(256, 128).to(device)

    output = model(x, y, fusedc)
    
    print(output[0].shape, output[1].shape, output[2].shape)
    # model=SKBlock(64,M=2,G=64)
    # model=cfm(64)
    # summary(model.to(device), x,y, fused)


