import torch
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt
from model import get_model
from dataloader import get_flowers102_dataloader, Flowers102Dataset

class ModelEvaluator:
    def __init__(self, config):
        """
        模型评估器
        
        参数:
            config: 配置字典，包含模型和评估相关的参数
        """
        self.config = config
        self.device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self.model = self._load_model()
        
        # 加载标签字典
        self.label_dict = self._load_label_dict()
        
        # 获取数据加载器
        self.train_loader, self.test_loader = get_flowers102_dataloader(
            config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )
        
        # 定义评估指标
        self.criterion = nn.CrossEntropyLoss()
    
    def _load_model(self):
        """加载预训练模型"""
        model = get_model(self.config)
        checkpoint = torch.load(self.config['checkpoint_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        return model
    
    def _load_label_dict(self):
        """加载标签字典"""
        return json.load(open(self.config['label_dict_path'], 'r', encoding='utf-8'))
    
    def evaluate_clean(self):
        """评估在原始测试集上的性能"""
        return self._evaluate_loader(self.test_loader)
    
    def evaluate_corruptions(self, corruptions, severity=3):
        """
        评估在不同噪声类型下的性能
        
        参数:
            corruptions: 噪声类型列表
            severity: 噪声程度 (1-5)
        """
        results = {}
        for corruption in corruptions:
            _, test_loader = get_flowers102_dataloader(
                self.config['data_dir'],
                batch_size=self.config['batch_size'],
                num_workers=self.config['num_workers'],
                corruption_type=corruption,
                severity=severity
            )
            acc = self._evaluate_loader(test_loader)
            results[corruption] = acc
        return results
    
    def _evaluate_loader(self, loader):
        """评估指定数据加载器的性能"""
        correct = 0
        total = 0
        loss = 0.0
        
        with torch.no_grad():
            for images, labels in tqdm(loader, desc='Evaluating'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100.0 * correct / total
        avg_loss = loss / len(loader)
        return {'accuracy': accuracy, 'loss': avg_loss}
    
    def visualize_predictions(self, corruption_type=None, severity=3, num_samples=5):
        """
        可视化模型预测结果
        
        参数:
            corruption_type: 噪声类型
            severity: 噪声程度
            num_samples: 显示的样本数量
        """
        test_dataset = Flowers102Dataset(
            self.config['data_dir'], 
            split='test',
            corruption_type=corruption_type,
            severity=severity
        )
        
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples*3))
        
        with torch.no_grad():
            for i in range(num_samples):
                # 随机选择样本
                idx = np.random.randint(len(test_dataset))
                image, label = test_dataset[idx]
                
                # 显示原始/噪声图像
                axes[i, 0].imshow(image.permute(1, 2, 0).numpy())
                axes[i, 0].set_title(f'True: {self.label_dict[str(label)]}')
                axes[i, 0].axis('off')
                
                # 进行预测
                image = image.unsqueeze(0).to(self.device)
                output = self.model(image)
                _, predicted = output.max(1)
                pred_label = predicted.item()
                
                # 显示预测结果
                axes[i, 1].imshow(image.squeeze(0).cpu().permute(1, 2, 0).numpy())
                axes[i, 1].set_title(
                    f'Predicted: {self.label_dict[str(pred_label)]}\n' + \
                    f'{"Correct" if pred_label == label else "Wrong"}'
                )
                axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    # 评估配置
    config = {
        'data_dir': '../datasets/flowers102',
        'checkpoint_path': './checkpoints/best_model.pth',
        'label_dict_path': '../datasets/flowers102/labels/label_dict.json',
        'model_name': 'resnet18',
        'num_classes': 102,
        'batch_size': 32,
        'num_workers': 4
    }
    
    # 创建评估器
    evaluator = ModelEvaluator(config)
    
    # 1. 评估原始测试集性能
    clean_results = evaluator.evaluate_clean()
    print("\n原始测试集性能:")
    print(f"Accuracy: {clean_results['accuracy']:.2f}%")
    print(f"Loss: {clean_results['loss']:.4f}")
    
    # 2. 评估不同噪声类型下的性能
    corruptions = ['snow', 'fog', 'brightness', 'contrast', 'elastic', 'pixelate', 'jpeg']
    corruption_results = evaluator.evaluate_corruptions(corruptions)
      
    print("\n不同噪声类型下的准确率:")
    for corruption, results in corruption_results.items():
        print(f"{corruption}: {results['accuracy']:.2f}%")
    
    # 3. 可视化预测结果
    print("\n可视化原始图像的预测结果...")
    evaluator.visualize_predictions()
    
    # 4. 可视化添加噪声后的预测结果
    for corruption in ['snow', 'fog']:
        print(f"\n可视化添加 {corruption} 噪声后的预测结果...")
        evaluator.visualize_predictions(corruption_type=corruption)

if __name__ == '__main__':
    main() 