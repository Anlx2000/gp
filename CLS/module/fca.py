import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def get_freq_indices(method):
    # 验证输入的方法名是否有效
    # 支持的方法包括：
    # - topN: 选择高频分量
    # - botN: 选择底部频率分量
    # - lowN: 选择低频分量
    # 其中N可以是1,2,4,8,16,32
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    
    # 从方法名中提取需要的频率数量
    # 例如: 'top16' -> 16
    num_freq = int(method[3:])

    if 'top' in method:
        # 预定义的高频DCT系数的x坐标
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        # 预定义的高频DCT系数的y坐标
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        # 根据需要的频率数量截取相应的坐标
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        # 预定义的低频DCT系数的x坐标
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        # 预定义的低频DCT系数的y坐标
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        # 预定义的底部DCT系数的x坐标
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        # 预定义的底部DCT系数的y坐标
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    
    # 返回选定的DCT系数坐标对
    return mapper_x, mapper_y

class MultiSpectralAttentionLayer(torch.nn.Module):
    """
    多光谱注意力层
    通过DCT变换和频率选择来实现特征的注意力机制
    
    参数:
        channel: 输入特征图的通道数
        dct_h: DCT变换的高度
        dct_w: DCT变换的宽度
        reduction: 通道数降维的比例
        freq_sel_method: 频率选择方法，如'top16'表示选择16个高频分量
    """
    def __init__(self, channel, dct_h, dct_w, reduction = 16, freq_sel_method = 'top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        # 将7x7频率空间映射到目标尺寸
        # 例如：在14x14尺寸中的(2,2)对应于7x7中的(1,1)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x] 
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        # 通道注意力的全连接层序列
        # 包含降维->ReLU->升维->Sigmoid的结构
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n,c,h,w = x.shape
        x_pooled = x
        # 如果输入特征图尺寸与DCT期望的尺寸不匹配，进行自适应池化
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            
        # 1. 通过DCT层获取频域特征
        y = self.dct_layer(x_pooled)
        # 2. 通过FC层生成通道注意力权重
        y = self.fc(y).view(n, c, 1, 1)
        # 3. 将注意力权重应用到原始特征图上
        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """
    生成DCT滤波器并应用于输入特征图
    
    参数:
        height: 输入特征图高度
        width: 输入特征图宽度
        mapper_x: DCT系数的x坐标列表
        mapper_y: DCT系数的y坐标列表
        channel: 输入特征图通道数
    """
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        
        # 确保x,y坐标对的数量相同
        assert len(mapper_x) == len(mapper_y)
        # 确保通道数能被频率数量整除
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # 注册DCT滤波器作为固定权重
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        
        # 应用DCT滤波器
        x = x * self.weight
        # 在空间维度上求和，得到通道维度的特征
        result = torch.sum(x, dim=[2,3])
        return result

    def build_filter(self, pos, freq, POS):
        """
        构建DCT基函数
        
        参数:
            pos: 位置坐标
            freq: 频率
            POS: 总长度
        """
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        # 对非零频率分量进行归一化
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        """
        生成完整的DCT滤波器
        
        参数:
            tile_size_x: 滤波器x方向大小 filter H
            tile_size_y: 滤波器y方向大小 filter W
            mapper_x: x方向的频率索引
            mapper_y: y方向的频率索引
            channel: 通道数
        """
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)
        # c_part = 每个频率分量对应的通道数
        # 例如：如果总通道数channel=64，选择了16个频率位置(len(mapper_x)=16)
        # 则每个频率位置对应4个通道(c_part=4)
        c_part = channel // len(mapper_x)
        
        # 为每个选定的频率位置生成对应的DCT基函数
        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    # 将第i个频率位置的DCT基函数分配给对应的通道组
                    # i*c_part : (i+1)*c_part 表示当前频率对应的通道范围
                    dct_filter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                        
        return dct_filter
    

if __name__ == '__main__':
    def visualize_dct_filters(height=7, width=7, channel=16, freq_sel_method='top16'):
        # 获取频率索引
        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        
        # 创建DCT层
        dct_layer = MultiSpectralDCTLayer(height, width, mapper_x, mapper_y, channel)
        
        # 获取DCT滤波器
        dct_filters = dct_layer.weight.cpu().numpy()
        
        # 创建子图
        num_filters = min(16, channel)  # 最多显示16个滤波器
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        
        # 展平axes数组以便迭代
        axes_flat = axes.flatten()
        
        # 绘制每个滤波器
        for i in range(num_filters):
            im = axes_flat[i].imshow(dct_filters[i], cmap='viridis')
            axes_flat[i].set_title(f'Filter {i+1}')
            axes_flat[i].axis('off')
            plt.colorbar(im, ax=axes_flat[i])
        
        plt.suptitle(f'DCT Filters ({freq_sel_method})', fontsize=16)
        plt.tight_layout()
        plt.show()

    # 可视化不同频率选择方法的DCT滤波器
    methods = ['top16', 'low16', 'bot16']
    for method in methods:
        print(f"Visualizing {method} filters...")
        visualize_dct_filters(freq_sel_method=method)
    