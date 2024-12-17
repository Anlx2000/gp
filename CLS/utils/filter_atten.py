import torch
import torch.nn as nn
from utils.filters import sobel_x, sobel_y, laplace, avgpool
import torch.nn.functional as F

class FilterAtten(nn.Module):
    def __init__(self, HW = 16, inplane=3):
        '''
        HW: 特征图大小
        inplane: 输入通道数
        '''
        super(FilterAtten, self).__init__()
        self.sobel_x_kernel = sobel_x
        self.sobel_y_kernel = sobel_y
        self.laplace_kernel = laplace
        self.avgpool_kernel = avgpool
        self.inplane = inplane
        self.gap = nn.AvgPool2d(HW)
        self.linear1 = nn.Linear(inplane, inplane // 2)
        self.linear2 = nn.Linear(inplane // 2, inplane)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        sobel_x_out = F.conv2d(x, self.sobel_x_kernel.repeat(self.inplane, 1, 1, 1), stride=1, padding=1, groups=self.inplane)
        sobel_y_out = F.conv2d(x, self.sobel_y_kernel.repeat(self.inplane, 1, 1, 1), stride=1, padding=1, groups=self.inplane)
        laplace_out = F.conv2d(x, self.laplace_kernel.repeat(self.inplane, 1, 1, 1), stride=1, padding=1, groups=self.inplane)
        avgpool_out = F.conv2d(x, self.avgpool_kernel.repeat(self.inplane, 1, 1, 1), stride=1, padding=1, groups=self.inplane)
        atten = sobel_x_out + sobel_y_out + laplace_out + avgpool_out
        atten = self.gap(atten)
        atten = atten.view(atten.size(0), -1)
        atten = self.linear1(atten)
        atten = self.linear2(atten)
        atten = self.softmax(atten)
        atten = atten.view(atten.size(0), -1, 1, 1)
        return atten * x

if __name__ == '__main__':
    filter_atten = FilterAtten(HW=16, inplane=3)
    x = torch.ones(1, 2, 16, 16)
    y = torch.zeros(1, 1, 16, 16)
    x = torch.cat([x, y], dim=1)
    out = filter_atten(x)
    print(out)
    print(out.shape)