'''
Author: Garfiled
Date: 2026-03-29 16:35:17
LastEditors: xavier
LastEditTime: 2026-03-29 16:35:20
FilePath: /faceid/utils/check_folder.py
'''
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def check_folder_empty(folder_path):
    """
    检查单个文件夹。
    返回: (folder_name, is_empty, count)
    """
    try:
        # 使用 scandir 比 listdir 更快，因为它不加载全部文件名
        with os.scandir(folder_path) as it:
            if not any(it):
                return folder_path, True, 0
            
            # 如果需要统计图片数量（可选，会稍微减慢速度）
            # count = len([entry for entry in os.scandir(folder_path) if entry.is_file()])
            # return folder_path, False, count
            return folder_path, False, 1 # 只要有一个就不为空
    except Exception as e:
        return folder_path, "Error", str(e)

def main():
    root_dir = '/workspace/faceid/dataset/faces_emore_images/'
    
    if not os.path.exists(root_dir):
        print(f"路径不存在: {root_dir}")
        return

    # 获取所有子文件夹名
    subfolders = [os.path.join(root_dir, d) for d in os.listdir(root_dir) 
                  if os.path.isdir(os.path.join(root_dir, d))]
    
    print(f"开始检查 {len(subfolders)} 个文件夹...")
    
    empty_folders = []
    error_folders = []
    
    # 使用线程池加速磁盘扫描
    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(tqdm(executor.map(check_folder_empty, subfolders), total=len(subfolders)))

    # 统计结果
    for path, is_empty, info in results:
        if is_empty is True:
            empty_folders.append(path)
        elif is_empty == "Error":
            error_folders.append(f"{path}: {info}")

    # 输出报告
    print("\n--- 检查报告 ---")
    print(f"总计文件夹数: {len(subfolders)}")
    print(f"正常（非空）: {len(subfolders) - len(empty_folders) - len(error_folders)}")
    print(f"空文件夹数量: {len(empty_folders)}")
    print(f"处理出错数量: {len(error_folders)}")
    
    if empty_folders:
        print("\n前10个空文件夹示例:")
        for f in empty_folders[:10]:
            print(f)
            
    if error_folders:
        print("\n错误信息示例:")
        for e in error_folders[:5]:
            print(e)

if __name__ == "__main__":
    main()