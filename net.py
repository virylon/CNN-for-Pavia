import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------搭建网络--------------------------------


class SeparableConv2d(nn.Module):  # Depth wise separable conv
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        # 每个input channel被自己的filters卷积操作
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
                               stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # in_channels,out_channels,kernel_size,stride(default:1),padding(default:0)
        self.conv1 = torch.nn.Sequential(
            SeparableConv2d(200, 16, 1, 1, 0),  # 1*1卷积核
            nn.ReLU(inplace=True),
            nn.GroupNorm(16, 16)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(200, 256, 1, 1, 0),
            nn.GroupNorm(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            SeparableConv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            SeparableConv2d(256, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            SeparableConv2d(128, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(64, 64)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(200, 256, 1, 1, 0),
            nn.GroupNorm(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            SeparableConv2d(256, 256, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            SeparableConv2d(256, 128, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            SeparableConv2d(128, 64, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.GroupNorm(64, 64)
        )

        self.classifier = nn.Sequential(
            nn.Linear(144*17*17, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 16),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        cat = torch.cat((x2, x1, x3), 1)
        cat = cat.view(-1, self.numFeatures(cat))  # 特征映射一维展开
        output = self.classifier(cat)

        return output

    def numFeatures(self, x):
        size = x.size()[1:]  # 获取卷积图像的h,w,depth
        num = 1
        for s in size:
            num *= s
            # print(s)
        return num

    def init_weights(self):  # 初始化权值
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()
