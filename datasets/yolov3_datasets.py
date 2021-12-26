import random
import os
import glob

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def horisontal_flip(images, targets):
    # 水平翻转
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets

class myDataset(Dataset):
    def __init__(self,img_size,data_path,augment=False,multiscale=True,is_train=True):
        self.is_train=is_train
        self.img_path=os.path.join(data_path,'images')
        self.label_path=os.path.join(data_path,'labels')
        self.images=os.listdir(self.img_path)
        self.train,self.vaild=train_test_split(self.images,test_size=0.2,random_state=2021,shuffle=True)
        self.augment = augment
        self.multiscale = multiscale
        self.img_size=img_size
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        
    def __getitem__(self,idx):
        if self.is_train:
            tmp=self.train[idx].split('.')[0]
        else:
            tmp=self.vaild[idx].split('.')[0]
        img_path=os.path.join(self.img_path,f'{tmp}.png')
        label_path=os.path.join(self.label_path,f'{tmp}.txt')
        img=transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
        
        targets = torch.zeros((len(boxes), 6))
        targets[:, 1:] = boxes
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)
        return img, targets
    
    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        if self.multiscale and self.batch_count % 10 == 0:
            # 多尺度训练
            self.img_size = np.random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return imgs, targets
    
    def __len__(self):
        if self.is_train:
            return len(self.train)
        else:
            return len(self.vaild)

class ImageFolder(Dataset):
    def __init__(self,img_size,folder_path):
        self.files = sorted(glob.glob("%s/*.png" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        # Resize
        img = resize(img, self.img_size)
        return img_path, img

    def __len__(self):
        return len(self.files)