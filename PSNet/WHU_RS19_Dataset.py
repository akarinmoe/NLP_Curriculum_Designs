import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class WHU_RS19_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        初始化数据集类
        :param root_dir: 数据集的根目录，包含各个类别的子文件夹
        :param transform: 用于图像预处理的转换函数（例如：标准化、缩放等）
        """
        self.root_dir = root_dir
        self.transform = transform

        # 获取所有类别文件夹
        self.categories = sorted(os.listdir(root_dir))  # 排序文件夹
        self.img_paths = []
        self.labels = []

        # 遍历类别文件夹，读取图像路径及其对应的标签
        for label, category in enumerate(self.categories):
            category_folder = os.path.join(root_dir, category)
            if os.path.isdir(category_folder):
                for img_name in os.listdir(category_folder):
                    img_path = os.path.join(category_folder, img_name)
                    self.img_paths.append(img_path)
                    self.labels.append(label)  # 使用文件夹名作为类别标签

    def __len__(self):
        """返回数据集的总长度"""
        return len(self.img_paths)

    def __getitem__(self, idx):
        """
        返回一个数据样本（图像和标签）
        :param idx: 索引
        :return: 图像和标签
        """
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        # 图像预处理（如果指定了transform）
        if self.transform:
            image = self.transform(image)

        return image, label
