import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class FeatureVisualizer:
    def __init__(self, model, dataloader, device, layer_name='layer4'):
        """
        特征可视化器
        
        参数:
            model: 模型
            dataloader: 数据加载器
            device: 设备 (cuda/cpu)
            layer_name: 要提取特征的层名称
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.layer_name = layer_name
        self.features = []
        self.labels = []
        self.activation = {}
        
        # 注册钩子
        self._register_hook()
    
    def _get_activation(self, name):
        """创建钩子函数"""
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook
    
    def _register_hook(self):
        """注册特征提取钩子"""
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                module.register_forward_hook(self._get_activation(name))
    
    def extract_features(self):
        """提取特征"""
        self.model.eval()
        with torch.no_grad():
            for images, labels in tqdm(self.dataloader, desc="提取特征"):
                images = images.to(self.device)
                # 前向传播
                self.model(images)
                # 获取特征
                features = self.activation[self.layer_name]
                # 将特征展平
                features = torch.mean(features, dim=[2, 3])  # 全局平均池化
                
                self.features.append(features.cpu().numpy())
                self.labels.append(labels.numpy())
        
        self.features = np.concatenate(self.features)
        self.labels = np.concatenate(self.labels)
    
    def visualize_tsne(self, n_components=2, perplexity=30):
        """使用t-SNE可视化特征"""
        print("正在进行t-SNE降维...")
        tsne = TSNE(n_components=n_components, perplexity=perplexity)
        features_embedded = tsne.fit_transform(self.features)
        
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(features_embedded[:, 0], features_embedded[:, 1], 
                            c=self.labels, cmap='tab20')
        plt.colorbar(scatter)
        plt.title('t-SNE visualization of features')
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')
        
        plt.savefig('./t-SNE_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("t-SNE结果已保存")


    
    def visualize_pca(self, n_components=2):
        """使用PCA可视化特征"""
        print("正在进行PCA降维...")
        pca = PCA(n_components=n_components)
        features_embedded = pca.fit_transform(self.features)
        
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(features_embedded[:, 0], features_embedded[:, 1], 
                            c=self.labels, cmap='tab20')
        plt.colorbar(scatter)
        plt.title(f'PCA visualization of features\nExplained variance ratio: {pca.explained_variance_ratio_}')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        

        plt.savefig('./PCA_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("PCA结果已保存")
    
    def plot_feature_distribution(self):
        """绘制特征分布"""
        plt.figure(figsize=(15, 5))
        
        # 特征均值分布
        plt.subplot(131)
        sns.histplot(self.features.mean(axis=0), kde=True)
        plt.title('Feature Means Distribution')
        
        # 特征标准差分布
        plt.subplot(132)
        sns.histplot(self.features.std(axis=0), kde=True)
        plt.title('Feature Std Distribution')
        
        # 类别分布
        plt.subplot(133)
        sns.histplot(self.labels, kde=False)
        plt.title('Label Distribution')
        
        plt.tight_layout()
        plt.savefig('./feature_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("特征分布已保存")

def visualize_model_features(model, dataloader, device,layer_name='backbone.layer4', save_dir=None):
    """
    可视化模型特征
    
    参数:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        save_dir: 保存目录
    """
    visualizer = FeatureVisualizer(model, dataloader, device,layer_name=layer_name)
    
    # 提取特征
    visualizer.extract_features()
    
    # t-SNE可视化
    visualizer.visualize_tsne()
    
    # PCA可视化
    visualizer.visualize_pca()
    
    # 特征分布可视化
    visualizer.plot_feature_distribution()

if __name__ == '__main__':
    # 测试代码
    from model import get_model
    from dataloader import get_flowers102_dataloader
    import yaml
    
    # 加载配置
    with open('utils/hyp.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = get_model(config)
    model.load_state_dict(torch.load('./checkpoints/best_model.pth')['model_state_dict'])
    model = model.to(device)
    
    # 获取数据加载器
    _, test_loader = get_flowers102_dataloader(
        root_dir='../datasets/flowers102',
        batch_size=32
    )
    
    # 可视化特征
    visualize_model_features(model, test_loader, device,layer_name='backbone.layer4') 