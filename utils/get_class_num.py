'''
Author: Garfiled
Date: 2026-03-25 22:36:40
LastEditors: xavier
LastEditTime: 2026-03-25 22:36:42
FilePath: /proj/faceid/utils/get_class_num.py
'''
import os

def count_folders(path):
    if not os.path.exists(path):
        print(f"路径不存在: {path}")
        return 0
    
    # 统计所有文件夹
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    return len(folders)

train_path = "/code/proj/dataset/vggface_48w_split/train"
num_classes = count_folders(train_path)

print(f"--- 数据集统计 ---")
print(f"训练集路径: {train_path}")
print(f"检测到类别总数 (class_num): {num_classes}")
print(f"------------------")

# 你可以直接赋值给你的变量
# class_num = num_classes