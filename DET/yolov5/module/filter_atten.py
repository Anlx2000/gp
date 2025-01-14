import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sobel_x = torch.tensor([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=torch.float, requires_grad=False, device=device).view(1, 1, 3, 3)
sobel_y = torch.tensor([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=torch.float, requires_grad=False, device=device).view(1, 1, 3, 3)
laplace = torch.tensor([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]], dtype=torch.float, requires_grad=False, device=device).view(1, 1, 3, 3)
avgpool = torch.tensor([[1/9, 1/9, 1/9],
                         [1/9, 1/9, 1/9],
                         [1/9, 1/9, 1/9]], dtype=torch.float, requires_grad=False, device=device).view(1, 1, 3, 3)

sharpen = torch.tensor([[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]], dtype=torch.float, requires_grad=False, device=device).view(1, 1, 3, 3)


class FilterAtten(nn.Module):
    def __init__(self, inplane=3):
        '''
        HW: 特征图大小
        inplane: 输入通道数
        '''
        super(FilterAtten, self).__init__()
        self.sobel_x_kernel = sobel_x
        self.sobel_y_kernel = sobel_y
        self.laplace_kernel = laplace
        self.sharpen_kernel = sharpen
        self.inplane = inplane
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = nn.Linear(inplane, inplane // 2)
        self.linear2 = nn.Linear(inplane // 2, inplane)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.sobel_x = nn.Conv2d(inplane, inplane, 3, stride=1, padding=1,groups=inplane, bias=False)
        self.sobel_y = nn.Conv2d(inplane, inplane, 3, stride=1, padding=1,groups=inplane, bias=False)
        self.laplace = nn.Conv2d(inplane, inplane, 3, stride=1, padding=1,groups=inplane, bias=False)
        self.sharpen = nn.Conv2d(inplane, inplane, 3, stride=1, padding=1,groups=inplane, bias=False)
        self.sobel_x.weight.data.copy_(self.sobel_x_kernel.repeat(self.inplane, 1, 1, 1))
        self.sobel_y.weight.data.copy_(self.sobel_y_kernel.repeat(self.inplane, 1, 1, 1))
        self.laplace.weight.data.copy_(self.laplace_kernel.repeat(self.inplane, 1, 1, 1))
        self.sharpen.weight.data.copy_(self.sharpen_kernel.repeat(self.inplane, 1, 1, 1))
        self.sobel_x.weight.requires_grad_(False)
        self.sobel_y.weight.requires_grad_(False)
        self.laplace.weight.requires_grad_(False)
        self.sharpen.weight.requires_grad_(False)
        
    def forward(self, x):
        device = x.device
        self.sobel_x_kernel = self.sobel_x_kernel.to(device)
        self.sobel_y_kernel = self.sobel_y_kernel.to(device)
        self.laplace_kernel = self.laplace_kernel.to(device)
        self.sharpen_kernel = self.sharpen_kernel.to(device)
        sobel_x_out = self.sobel_x(x)
        sobel_y_out = self.sobel_y(x)
        laplace_out = self.laplace(x)
        sharpen_out = self.sharpen(x)
        atten = sobel_x_out + sobel_y_out + laplace_out + sharpen_out
        atten = self.gap(atten)
        atten = atten.view(atten.size(0), -1)
        atten = self.linear1(atten)
        atten = self.relu(atten)
        atten = self.linear2(atten)
        atten = self.softmax(atten)
        atten = atten.view(atten.size(0), -1, 1, 1)
        return atten * x

if __name__ == '__main__':
    filter_atten = FilterAtten(inplane=3)
    x = torch.ones(1, 2, 16, 16)
    y = torch.zeros(1, 1, 16, 16)
    x = torch.cat([x, y], dim=1)
    out = filter_atten(x)
    print(out)
    print(out.shape)