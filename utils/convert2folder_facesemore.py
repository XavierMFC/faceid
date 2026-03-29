'''
Author: Garfiled
Date: 2026-03-29 12:09:28
LastEditors: xavier
LastEditTime: 2026-03-29 12:09:40
FilePath: /faceid/utils/convert2folder_facesemore.py
'''
import os
import cv2
import mxnet as mx
from tqdm import tqdm

# --- 配置路径 ---
# pip install mxnet opencv-python tqdm
# 直接 pip install mxnet（不带 -cuxx 后缀）:解决潜在冲突

rec_path = 'datasets/faces_emore/train.rec'
idx_path = 'datasets/faces_emore/train.idx'
output_dir = 'datasets/faces_emore_images/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 加载 RecordIO 文件
imgrec = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')

# 获取索引范围
keys = list(imgrec.keys)
print(f"检测到图片总数: {len(keys)}")

for i in tqdm(keys):
    item = imgrec.read_idx(i)
    header, img = mx.recordio.unpack_img(item)
    
    # 获取标签（类别 ID）
    # InsightFace 的 label 通常是浮点数，代表整数类别
    label = int(header.label) if isinstance(header.label, float) else int(header.label[0])
    
    # 创建类别文件夹
    target_dir = os.path.join(output_dir, str(label))
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # 保存图片
    img_name = f"{i}.jpg"
    save_path = os.path.join(target_dir, img_name)
    
    # mxnet 出来的默认是 RGB，opencv 保存需要 BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img)

print("转换完成！")