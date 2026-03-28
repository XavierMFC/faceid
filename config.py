'''
Author: Garfiled
Date: 2026-03-28 15:11:13
LastEditors: xavier
LastEditTime: 2026-03-28 23:20:46
FilePath: /faceid/config.py
'''
import torch


arch = "ir_18"
head_type = "adaface"
class_num = 10000
embedding_size = 512
m = 0.4
h = 0.33
t_alpha = 0.1
s = 64

epochs = 100
unfreeze_epoch = 0
batch_size = 128
learning_rate = 0.01
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# trianing_samples_dir = "/code/proj/dataset/vggface_48w_split/train"
# val_samples_dir = "/code/proj/dataset/vggface_48w_split/val"

trianing_samples_dir = "/code/proj/dataset/lfw_data/lfw_split/train"
val_samples_dir = "/code/proj/dataset/lfw_data/lfw_split/val"
num_workers = 4
target_size = (112, 112)

pretrain = "/code/proj/faceid/exp/2026-03-28_18-04-01/adaface_epoch_71_batch_0.pth"
root_dir = "/code/proj/faceid"