'''
Author: Garfiled
Date: 2026-03-24 22:48:54
LastEditors: xavier
LastEditTime: 2026-03-29 22:33:52
FilePath: /faceid/train.py
'''
from pyexpat import model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

from dastaset.image_folder_dataset import get_loaders
from head import build_head
from net import build_model

import config

def train():
    weight_save_path = f"{config.root_dir}/exp/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(weight_save_path, exist_ok=True)

    print("正在加载数据集并划分...")
    train_loader, val_loader = get_loaders(config)
    
    model = build_model(model_name=config.arch).to(config.device)
    head = build_head(
        head_type=config.head_type,
        embedding_size=config.embedding_size,
        class_num=config.class_num,
        m=config.m,
        h=config.h,
        t_alpha=config.t_alpha,
        s=config.s
    ).to(config.device)

    try:
        ckpt = torch.load(config.pretrain)
        model.load_state_dict(ckpt['model'])
        head.load_state_dict(ckpt['head'], strict=False)  # head 可能因为类别数不同而无法完全加载
    except:
        print("未找到 IR18 模型检查点，将从头开始")
    
    for param in model.parameters():
        param.requires_grad = False
    # --- 4. 优化器、损失函数与学习率调度 ---
    optimizer = optim.SGD([
        {'params': model.parameters()},
        {'params': head.parameters()}
    ], lr=config.learning_rate, momentum=0.9, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # --- 5. 训练与测试主循环 ---
    print(f"开始训练，使用设备: {config.device}")
    
    for epoch in tqdm(range(config.epochs)):
        if epoch == config.unfreeze_epoch:
                    print(f"\n>>> Epoch [{epoch+1}]: 达到预定条件，解冻 Backbone，开始全局微调！")
                    for param in model.parameters():
                        param.requires_grad = True
        # ================== 训练阶段 ==================
        model.train()
        head.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (images, labels) in tqdm(enumerate(train_loader)):
            images, labels = images.to(config.device), labels.to(config.device)

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

            print(f"Epoch [{epoch+1}/{config.epochs}] Batch [{batch_idx}/{len(train_loader)}] | Train Loss: {loss.item():.4f} | Train Acc: {100 * correct_train / total_train:.2f}%")
            
            # if batch_idx % 20 == 0:

            # state_dict = {
            #     'model': model.state_dict(),
            #     'head': head.state_dict()
            # }
            # # save weight
            # torch.save(state_dict, f"{weight_save_path}/adaface_epoch_{epoch+1}_batch.pth")
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
                images, labels = images.to(config.device), labels.to(config.device)
                
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
        print(f"Epoch [{epoch+1}/{config.epochs}] 总结:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  当前学习率: {scheduler.get_last_lr()[0]:.6f}")
        print("-" * 60)

        state_dict = {
            'model': model.state_dict(),
            'head': head.state_dict()
        }

        # save weight
        torch.save(state_dict, f"{weight_save_path}/adaface_epoch_{epoch+1}.pth")
        print(f"已保存模型权重到: {weight_save_path}/adaface_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()