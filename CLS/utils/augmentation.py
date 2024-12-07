import random
import numpy as np
import cv2
import torch
import yaml
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
import matplotlib.pyplot as plt

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            return F.hflip(image)
        return image

class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            return F.vflip(image)
        return image

class RandomRotation:
    def __init__(self, degrees=10):
        self.degrees = degrees

    def __call__(self, image):
        angle = random.uniform(-self.degrees, self.degrees)
        return F.rotate(image, angle)

class ColorJitter:
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
        self.transform = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def __call__(self, image):
        return self.transform(image)

class RandomErasing:
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        self.transform = transforms.RandomErasing(
            p=p,
            scale=scale,
            ratio=ratio,
            value='random'
        )

    def __call__(self, image):
        return self.transform(image)

class RandomResizedCrop:
    def __init__(self, size=224, scale=(0.08, 1.0), ratio=(3./4., 4./3.)):
        self.transform = transforms.RandomResizedCrop(
            size=size,
            scale=scale,
            ratio=ratio
        )

    def __call__(self, image):
        return self.transform(image)

class Normalize:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.transform = transforms.Normalize(mean=mean, std=std)

    def __call__(self, image):
        return self.transform(image)

def create_train_transforms(hyp):
    """根据超参数创建训练数据增强"""
    return Compose([
        RandomResizedCrop(
            size=hyp['img_size'],
            scale=(hyp['scale_min'], hyp['scale_max']),
            ratio=(hyp['ratio_min'], hyp['ratio_max'])
        ),
        RandomHorizontalFlip(p=hyp['flip_lr']),
        RandomVerticalFlip(p=hyp['flip_ud']),
        RandomRotation(degrees=hyp['degrees']),
        ColorJitter(
            brightness=hyp['brightness'],
            contrast=hyp['contrast'],
            saturation=hyp['saturation'],
            hue=hyp['hue']
        ),
        transforms.ToTensor(),
        Normalize(
            mean=hyp['normalize_mean'],
            std=hyp['normalize_std']
        ),
        RandomErasing(
            p=hyp['erasing_p'],
            scale=hyp['erasing_scale'],
            ratio=hyp['erasing_ratio']
        ),
    ])

def create_val_transforms(hyp):
    """创建验证数据增强"""
    return Compose([
        transforms.Resize(int(hyp['img_size'] * 1.143)),
        transforms.CenterCrop(hyp['img_size']),
        transforms.ToTensor(),
        Normalize(
            mean=hyp['normalize_mean'],
            std=hyp['normalize_std']
        )
    ])

if __name__ == '__main__':
    hyp = yaml.safe_load(open('./hyp.yaml', 'r'))
    print(hyp)
    train_transforms = Compose([
        RandomResizedCrop(
            size=hyp['img_size'],
            scale=(hyp['scale_min'], hyp['scale_max']),
            ratio=(hyp['ratio_min'], hyp['ratio_max'])
        ),
        RandomHorizontalFlip(p=hyp['flip_lr']),
        RandomVerticalFlip(p=hyp['flip_ud']),
        RandomRotation(degrees=hyp['degrees']),
        ColorJitter(
            brightness=hyp['brightness'],
            contrast=hyp['contrast'],
            saturation=hyp['saturation'],
            hue=hyp['hue']
        )
    ])
    print(train_transforms)
    img = Image.open('../../datasets/flowers102/images/image_00001.jpg')
    
    # 应用增强
    augmented_img = train_transforms(img)

    # 将增强后的图像转换为张量并显示
    augmented_tensor = F.to_tensor(augmented_img)  # 转换为张量
    plt.imshow(augmented_tensor.permute(1, 2, 0))  # 将张量转换为适合显示的格式
    plt.axis('off')  # 关闭坐标轴
    plt.show()  # 显示图像