import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json
from scipy.io import loadmat
import numpy as np
from imagecorruptions import corrupt

class Flowers102Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, corruption_type=None, severity=1):
        """
        参数:
            root_dir (str): 数据集根目录
            split (str): 'train' 或 'test'
            transform: 图像预处理转换
            corruption_type (str): 噪声类型，可选 ['snow', 'fog', 'brightness', 'contrast', 'elastic', 'pixelate', 'jpeg']
            severity (int): 噪声程度 (1-5)
        """
        self.root_dir = root_dir
        self.split = split
        self.corruption_type = corruption_type
        self.severity = severity
        
        # 默认的图像转换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform  

        splits = {'train':[], 'test':[]}
        data_split_by_class = {i:[] for i in range(1, 103)}
        data = loadmat(os.path.join(root_dir,'labels/imagelabels.mat'))
        labels = data['labels'][0].tolist()
        for i, label in enumerate(labels):
            data_split_by_class[label].append(i + 1)
        for label, imgs in data_split_by_class.items():
            sz = len(imgs)
            train_sz = int(sz * 0.8)
            test_sz = sz - train_sz
            
            splits['train'].extend(
                [ {'image_path': os.path.join(root_dir, f'images/image_{img_id:05d}.jpg'),
                   'label' : label - 1
                   } for img_id in imgs[:train_sz] ]
            )
            splits['test'].extend(
                [ {'image_path': os.path.join(root_dir, f'images/image_{img_id:05d}.jpg'),
                   'label' : label - 1
                   } for img_id in imgs[-test_sz:] ]
            )

        self.samples = splits[split]

    def apply_corruption(self, image):
        """
        对图像应用指定类型的噪声
        
        参数:
            image: numpy格式的图像 (H, W, C)
        """
        corruptions = {
            'snow': lambda x: corrupt(x, corruption_name='snow', severity=self.severity),
            'fog': lambda x: corrupt(x, corruption_name='fog', severity=self.severity),
            'brightness': lambda x: corrupt(x, corruption_name='brightness', severity=self.severity),
            'contrast': lambda x: corrupt(x, corruption_name='contrast', severity=self.severity),
            'elastic': lambda x: corrupt(x, corruption_name='elastic_transform', severity=self.severity),
            'pixelate': lambda x: corrupt(x, corruption_name='pixelate', severity=self.severity),
            'jpeg': lambda x: corrupt(x, corruption_name='jpeg_compression', severity=self.severity)
        }
        
        if self.corruption_type not in corruptions:
            raise ValueError(f"不支持的噪声类型: {self.corruption_type}")
            
        return corruptions[self.corruption_type](image)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 读取图片
        image = Image.open(sample['image_path']).convert('RGB')
        
        # 如果指定了噪声类型，先将图像转换为numpy格式并添加噪声
        if self.corruption_type is not None:
            # 转换为numpy数组
            image_np = np.array(image)
            # 添加噪声
            image_np = self.apply_corruption(image_np)
            # 转回PIL图像
            image = Image.fromarray(image_np)
        
        # 应用转换
        if self.transform:
            image = self.transform(image)
            
        return image, sample['label']

def get_flowers102_dataloader(root_dir, batch_size=32, num_workers=4, shuffle=True, 
                            corruption_type=None, severity=1):
    """
    获取 Flowers102 数据集的 dataloader
    
    参数:
        root_dir (str): 数据集根目录
        batch_size (int): 批次大小
        num_workers (int): 数据加载的线程数
        shuffle (bool): 是否打乱数据
        corruption_type (str): 噪声类型
        severity (int): 噪声程度
    
    返回:
        train_loader, test_loader
    """
    # 创建数据集实例
    train_dataset = Flowers102Dataset(
        root_dir, 
        split='train',
        corruption_type=corruption_type,
        severity=severity
    )
    test_dataset = Flowers102Dataset(
        root_dir, 
        split='test',
        corruption_type=corruption_type,
        severity=severity
    )
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

if __name__ == '__main__':  
    # 测试代码
    label_dict = json.load(open('../datasets/flowers102/labels/label_dict.json', 'r', encoding='utf-8'))
    
    # 测试不同类型的噪声
    corruptions = ['snow', 'fog', 'brightness', 'contrast', 'elastic', 'pixelate', 'jpeg']
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 10))
    
    # 获取一张原始图像和添加不同噪声后的效果
    train_loader, _ = get_flowers102_dataloader('../datasets/flowers102', batch_size=1, shuffle=False)
    for data, label in train_loader:
        # 显示原始图像
        plt.subplot(2, 4, 1)
        plt.imshow(data[0].permute(1, 2, 0).numpy())
        plt.title('Original')
        
        # 显示不同噪声效果
        for i, corruption in enumerate(corruptions):
            _, test_loader = get_flowers102_dataloader(
                '../datasets/flowers102', 
                batch_size=1, 
                shuffle=False,
                corruption_type=corruption,
                severity=3
            )
            for corrupted_data, _ in test_loader:
                plt.subplot(2, 4, i+2)
                plt.imshow(corrupted_data[0].permute(1, 2, 0).numpy())
                plt.title(corruption)
                break
        break
    
    plt.tight_layout()
    plt.show()