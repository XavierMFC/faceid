'''
Author: Garfiled
Date: 2026-03-24 22:48:54
LastEditors: xavier
LastEditTime: 2026-03-25 22:26:17
FilePath: /proj/adaface/train.py
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np

# 确保路径正确，引入刚才写好的获取数据的函数
from dastaset.image_folder_dataset import get_loaders
from head import build_head
from net import build_model

def train():
    # --- 1. 基本参数与设备配置 ---
    arch = "ir_18"
    head_type = "adaface"
    class_num = 128
    embedding_size = 512
    m = 0.1
    h = 0.33
    t_alpha = 0.1
    s = 64
    
    epochs = 100
    batch_size = 128
    learning_rate = 0.01
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- 2. 获取数据并挂载 Transform ---
    print("正在加载数据集并划分...")
    train_loader, val_loader = get_loaders("/code/proj/dataset/lfw_data/lfw_split/train", "/code/proj/dataset/lfw_data/lfw_split/val", batch_size=32)
    
    # --- 3. 构建模型与 Head ---
    model = build_model(model_name=arch).to(device)
    head = build_head(
        head_type=head_type,
        embedding_size=embedding_size,
        class_num=class_num,  # 传入实际读取到的类别数
        m=m,
        h=h,
        t_alpha=t_alpha,
        s=s
    ).to(device)

    try:
        # 2. 加载大字典
        ckpt = torch.load("/code/proj/adaface/ckp/adaface_IR18.pth")

        # 3. 分别注入
        model.load_state_dict(ckpt['model'])
        head.load_state_dict(ckpt['head'])
    except:
        print("未找到 IR18 模型检查点，将从头开始")
        
    # --- 4. 优化器、损失函数与学习率调度 ---
    optimizer = optim.SGD([
        {'params': model.parameters()},
        {'params': head.parameters()}
    ], lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()


    import matplotlib.pyplot as plt

    def debug_loader_samples(loader, save_path='./loader_check.jpg'):
        # 1. 迭代出一个 batch
        images, labels = next(iter(loader))
        
        # 2. 取前 8 张图片进行展示
        num_check = min(8, len(images))
        fig, axes = plt.subplots(1, num_check, figsize=(20, 5))
        
        for i in range(num_check):
            img = images[i]
            
            # 3. 反归一化：因为我们在 transform 中做了 (x - 0.5) / 0.5
            # 所以恢复公式是：x * 0.5 + 0.5
            img = img * 0.5 + 0.5
            
            # 4. 调整维度从 [C, H, W] 到 [H, W, C] 并转为 numpy
            img_np = img.permute(1, 2, 0).cpu().numpy()
            
            # 剪裁到 0-1 之间防止溢出
            img_np = np.clip(img_np, 0, 1)
            
            axes[i].imshow(img_np)
            axes[i].set_title(f"Label: {labels[i].item()}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"✅ 已从 Loader 提取样本并保存至: {save_path}")

    # 调用方法
    debug_loader_samples(train_loader, './train_batch_check.jpg')
    debug_loader_samples(val_loader, './val_batch_check.jpg')

    # --- 5. 训练与测试主循环 ---
    print(f"开始训练，使用设备: {device}")
    
    for epoch in range(epochs):
        # ================== 训练阶段 ==================
        model.train()
        head.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            
            # 前向传播
            features, norms = model(images)
            outputs = head(features, norms, labels)
            
            # 计算损失并反向传播
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # 统计准确率
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] | Train Loss: {loss.item():.4f}")

        train_acc = 100 * correct_train / total_train
        scheduler.step()
        
        # ================== 验证阶段 ==================
        model.eval()
        head.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                features, norms = model(images)
                outputs = head(features, norms, labels)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        val_acc = 100 * correct_val / total_val
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # 打印 Epoch 总结
        print("-" * 60)
        print(f"Epoch [{epoch+1}/{epochs}] 总结:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  当前学习率: {scheduler.get_last_lr()[0]:.6f}")
        print("-" * 60)

        state_dict = {
            'model': model.state_dict(),
            'head': head.state_dict()
        }
        torch.save(state_dict, f"/code/proj/adaface/ckp/adaface_IR18.pth")

if __name__ == "__main__":
    train()