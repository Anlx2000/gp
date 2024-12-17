import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from model import get_model
from dataloader import get_flowers102_dataloader
import argparse  # 导入argparse

def train(config):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = get_model(config)
    print(model)
    # 加载训练过的模型
    if config['resume']:
        model_path = os.path.join(config['checkpoint_dir'], 'best_model.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print(f"成功加载模型: {model_path}")
        else:
            print(f"未找到模型文件: {model_path}")
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    # 学习率调度器 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=5, verbose=True
    )
    
    # 获取数据加载器
    train_loader, test_loader = get_flowers102_dataloader(
        root_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # 创建TensorBoard写入器
    writer = SummaryWriter(os.path.join(config['log_dir'], 'runs'))
    
    # 训练循环
    best_acc = 0.0
    for epoch in range(config['epochs']):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]} [Train]')
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            train_pbar.set_postfix({
                'loss': train_loss/train_total,
                'acc': 100.*train_correct/train_total
            })
        
        # 测试阶段
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{config["epochs"]} [Test]')
            for images, labels in test_pbar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
                
                test_pbar.set_postfix({
                    'loss': test_loss/test_total,
                    'acc': 100.*test_correct/test_total
                })
        
        # 计算平均损失和准确率
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        test_loss = test_loss / len(test_loader)
        test_acc = 100. * test_correct / test_total
        
        # 更新学习率
        scheduler.step(test_acc)
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, os.path.join(config['checkpoint_dir'], 'best_model.pth'))
        
        print(f'Epoch {epoch+1}/{config["epochs"]}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        print(f'Best Test Acc: {best_acc:.2f}%')
        print('-' * 50)
    
    writer.close()

if __name__ == '__main__':
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='训练配置')
    parser.add_argument('--data_dir', type=str, default='../datasets/flowers102', help='数据集路径')
    parser.add_argument('--log_dir', type=str, default='./logs', help='日志路径')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='模型保存路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--num_workers', type=int, default=4, help='工作线程数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--model_name', type=str, default='resnet18', help='使用的模型')
    parser.add_argument('--num_classes', type=int, default=102, help='类别数')
    parser.add_argument('--pretrained', type=bool, default=False, help='是否使用预训练模型')
    parser.add_argument('--resume', type=bool, default=False, help='是否从上次训练中恢复')

    # 解析参数
    config = vars(parser.parse_args())
    
    # 创建必要的目录
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # 开始训练
    train(config) 