'''
Author: Garfiled
Date: 2026-03-25 20:16:44
LastEditors: xavier
LastEditTime: 2026-03-25 21:50:58
FilePath: /proj/dataset/convert.py
'''
import os
import numpy as np
from sklearn.datasets import fetch_lfw_people
from PIL import Image

def export_lfw_to_folders(output_base_dir="lfw_dataset", min_faces=20, color=True):
    """
    将 sklearn 的 LFW 数据集导出为按人物分类的文件夹。
    
    参数:
    output_base_dir: 数据保存的根目录
    min_faces: 每个人物至少需要包含的图片数量，用于过滤数据
    color: 是否下载彩色图片 (True 为 RGB, False 为灰度图)
    """
    print("正在下载/加载 LFW 数据集，这可能需要一点时间...")
    # 获取数据集
    lfw_people = fetch_lfw_people(
        data_home="/code/proj/dataset/lfw_data",
        min_faces_per_person=min_faces, color=color)
    
    images = lfw_people.images
    target = lfw_people.target
    target_names = lfw_people.target_names
    
    print(f"成功加载数据集：共 {len(images)} 张图片，包含 {len(target_names)} 个人物。")
    
    # 创建主目录
    os.makedirs(output_base_dir, exist_ok=True)
    
    print("开始将图片保存到对应文件夹...")
    for i, (img_array, label_idx) in enumerate(zip(images, target)):
        # 获取人物姓名并处理成合法的文件名（将空格替换为下划线）
        person_name = target_names[label_idx].replace(" ", "_")
        person_dir = os.path.join(output_base_dir, person_name)
        
        # 如果该人物的文件夹不存在，则创建
        os.makedirs(person_dir, exist_ok=True)
        
        # sklearn 返回的图像数据是浮点型 (0-255)，需要转换为 uint8
        img_array_uint8 = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
        
        # 将 numpy 数组转换为 PIL Image 对象
        if color:
            img = Image.fromarray(img_array_uint8, 'RGB')
        else:
            img = Image.fromarray(img_array_uint8, 'L')
            
        # 拼接文件路径并保存
        file_name = f"{person_name}_{i:04d}.jpg"
        file_path = os.path.join(person_dir, file_name)
        img.save(file_path)
        
    print(f"完成！所有图片已成功保存到 '{output_base_dir}' 目录下。")

# 运行导出函数
if __name__ == "__main__":
    # 你可以根据需要修改 min_faces 和 color
    export_lfw_to_folders(output_base_dir="/code/proj/dataset/lfw_data/lfw_images", min_faces=20, color=True)