import torch
import torch.nn as nn
from resnet18 import resnet18
from sku import SKNet
from utils.filter_atten import FilterAtten
import torchvision.models as models

class FlowerClassifier(nn.Module):
    def __init__(self, num_classes=102, model_name='resnet50', pretrained=True):
        """
        花卉分类模型
        
        参数:
            num_classes (int): 分类类别数，默认102种花
            model_name (str): 使用的ResNet版本 ('resnet18', 'resnet34', 'resnet50', 'resnet101')
            pretrained (bool): 是否使用预训练权重
        """
        super(FlowerClassifier, self).__init__()
        
        # 选择backbone
        if model_name == 'resnet18':
            self.backbone = models.resnet18(num_classes, pretrained=pretrained)
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"不支持的模型名称: {model_name}")
        
        # 获取特征维度
        in_features = self.backbone.fc.in_features
        
        # self.backbone.avgpool = nn.Sequential(
        #     FilterAtten(HW=7, inplane=512),
        #     nn.AdaptiveAvgPool2d((1, 1))
        # )
        self.backbone.layer1.append(FilterAtten(HW=56, inplane=64))
        self.backbone.layer2.append(FilterAtten(HW=28, inplane=128))
        self.backbone.layer3.append(FilterAtten(HW=14, inplane=256))

        # 替换最后的全连接层
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        # cls_output = self.cls_classifier(x)
        # corruption_output = self.corruption_classifier(x)
        # return cls_output, corruption_output
        # x = self.sku(x)
        return x

def get_model(config):
    """
    根据配置创建模型
    
    参数:
        config: 配置字典，包含模型参数
    返回:
        model: 创建的模型
    """
    model = FlowerClassifier(
        num_classes=config.get('num_classes', 102),
        model_name=config.get('model_name', 'resnet18'),
        pretrained=config.get('pretrained', True)
    )
    return model

if __name__ == '__main__':
    # 测试代码
    model = FlowerClassifier()
    x = torch.randn(4, 3, 224, 224)  # 批次大小为4，3通道，224x224图像
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")  # 应该是 [4, 102] 