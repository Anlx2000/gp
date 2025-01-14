import torch.nn as nn
import torch



class DLK(nn.Module):
    def __init__(self, inplane):
        super().__init__()
        self.att_conv1 = nn.Conv2d(inplane, inplane, kernel_size=5, stride=1, padding=2, groups=inplane)
        self.att_conv2 = nn.Conv2d(inplane, inplane, kernel_size=7, stride=1, padding=9, groups=inplane, dilation=3)

        self.spatial_se = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):   
        att1 = self.att_conv1(x)
        att2 = self.att_conv2(att1)

        att = torch.cat([att1, att2], dim=1)
        avg_att = torch.mean(att, dim=1, keepdim=True)
        max_att,_ = torch.max(att, dim=1, keepdim=True)
        att = torch.cat([avg_att, max_att], dim=1)
        att = self.spatial_se(att)
        output = att1 * att[:,0,:,:].unsqueeze(1) + att2 * att[:,1,:,:].unsqueeze(1)
        output = output + x
        return output





class DLKAtten(nn.Module):
    def __init__(self, inplane):
        super().__init__()
        self.sharpen = torch.tensor([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)

        self.att_conv1 = nn.Conv2d(inplane, inplane, kernel_size=3, stride=1, padding=1,groups=inplane, bias=False)
        self.att_conv2 = nn.Conv2d(inplane, inplane, kernel_size=3, stride=1, padding=1, groups=inplane)
        self.att_conv1.weight.data.copy_(self.sharpen.repeat(inplane, 1, 1, 1))
        self.att_conv1.weight.requires_grad_(False)
        self.spatial_se = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):   
        att1 = self.att_conv1(x)
        att2 = self.att_conv2(x)

        att = torch.cat([att1, att2], dim=1)
        avg_att = torch.mean(att, dim=1, keepdim=True)
        max_att,_ = torch.max(att, dim=1, keepdim=True)
        att = torch.cat([avg_att, max_att], dim=1)
        att = self.spatial_se(att)
        output = att1 * att[:,0,:,:].unsqueeze(1) + att2 * att[:,1,:,:].unsqueeze(1)
        output = output + x
        return output