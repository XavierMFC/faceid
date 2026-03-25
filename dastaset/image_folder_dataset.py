import os

import torchvision.datasets as datasets
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
from .augmenter import Augmenter
import torch

class CustomImageFolderDataset(datasets.ImageFolder):
    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=datasets.folder.default_loader,
                 is_valid_file=None,
                 low_res_augmentation_prob=0.0,
                 crop_augmentation_prob=0.0,
                 photometric_augmentation_prob=0.0,
                 swap_color_channel=False,
                 output_dir='./',
                 ):

        super(CustomImageFolderDataset, self).__init__(root,
                                                       transform=transform,
                                                       target_transform=target_transform,
                                                       loader=loader,
                                                       is_valid_file=is_valid_file)
        self.root = root
        self.augmenter = Augmenter(crop_augmentation_prob, photometric_augmentation_prob, low_res_augmentation_prob)
        self.swap_color_channel = swap_color_channel
        self.output_dir = output_dir  # for checking the sanity of input images

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        
        path, target = self.samples[index]
        sample = self.loader(path)        # sample = Image.fromarray(np.asarray(sample)[:,:,::-1])

        if self.swap_color_channel:
            # swap RGB to BGR if sample is in RGB
            # we need sample in BGR
            sample = Image.fromarray(np.asarray(sample)[:,:,::-1])

        sample = self.augmenter.augment(sample)

        sample_save_path = os.path.join(self.output_dir, 'training_samples', 'sample.jpg')
        if not os.path.isfile(sample_save_path):
            os.makedirs(os.path.dirname(sample_save_path), exist_ok=True)
            cv2.imwrite(sample_save_path, np.array(sample))  # the result has to look okay (Not color swapped)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target



from torchvision.transforms import functional as F

class LetterboxResize:
    """保持宽高比的 Resize (Letterbox)"""
    def __init__(self, size=(112, 112)):
        self.size = size # (h, w)

    def __call__(self, image):
        # 1. 计算缩放比例
        w, h = image.size
        target_h, target_w = self.size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 2. 等比例缩放
        image = F.resize(image, (new_h, new_w))
        
        # 3. 计算 Padding 数量使之达到目标尺寸
        delta_w = target_w - new_w
        delta_h = target_h - new_h
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        
        # 4. 填充黑边 (fill=0)
        return F.pad(image, padding, fill=0)

def get_loaders(train_data_path, val_data_path, batch_size=32):
    # 统一尺寸变量
    target_size = (112, 112)

    train_transform = transforms.Compose([
        LetterboxResize(target_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    val_transform = transforms.Compose([
        LetterboxResize(target_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 实例化
    train_dataset = CustomImageFolderDataset(
        root=train_data_path,
        transform=train_transform,
        low_res_augmentation_prob=0.2,
        crop_augmentation_prob=0.2,
        photometric_augmentation_prob=0.2,
        output_dir="/code/proj/adaface/training_samples" # 建议传 None 并在类内判断，避免多进程写文件冲突
    )

    val_dataset = CustomImageFolderDataset(
        root=val_data_path,
        transform=val_transform,
        low_res_augmentation_prob=0.0,
        crop_augmentation_prob=0.0,
        photometric_augmentation_prob=0.0,
        output_dir="/code/proj/adaface/training_samples"
    )
    
    # 关键点：确保验证集的类别映射与训练集一致
    val_dataset.class_to_idx = train_dataset.class_to_idx
    val_dataset.classes = train_dataset.classes

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    return train_loader, val_loader