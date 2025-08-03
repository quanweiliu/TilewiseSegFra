
import os
from PIL import Image
import tifffile
import numpy as np
import scipy.io as scio
import torch
from torch.utils import data
from torchvision import transforms


class Road_loader(data.DataLoader):
    def __init__(self, root, split='train', img_size=512, is_augmentation=False):
        self.root = root
        self.split = split
        self.n_classes = 2
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size))
        
        # self.tf = transforms.Compose([transforms.ToTensor()])
        self.gaofen_data_path = os.path.join(self.root, self.split, 'image128')
        self.gaofen_imgs = sorted(os.listdir(self.gaofen_data_path), key=self.sort_key)
        self.lidar_data_path = os.path.join(self.root, self.split, 'sar128')
        self.lidar_imgs = sorted(os.listdir(self.lidar_data_path), key=self.sort_key)
        self.mask_data_path = os.path.join(self.root, self.split, 'mask128')
        self.masks = sorted(os.listdir(self.mask_data_path), key=self.sort_key)
        self.augmentation = is_augmentation
        
    def __getitem__(self, index):
        img_name = self.gaofen_imgs[index]
        img_id = img_name[:-4]
        gaofen_path = os.path.join(self.gaofen_data_path, str(img_id) + '.mat')
        lidar_path = os.path.join(self.lidar_data_path, str(img_id) + '.mat')
        mask_path = os.path.join(self.mask_data_path, str(img_id) + '.mat')

        gaofen2np = np.array(scio.loadmat(gaofen_path)['img'], np.float32)
        lidar2np = np.array(scio.loadmat(lidar_path)['sar'], np.float32)
        mask2np = np.array(scio.loadmat(mask_path)['map'], np.float32)
        # print("mask2np", mask2np.shape)   # 128, 128, 3
        # mask2np = np.array(scio.loadmat(mask_path)['map'], np.float32)[:, :, 0]
        # print("mask2np", mask2np.shape)

        gaofen2np, lidar2np = self.norm(gaofen2np, lidar2np)

        if self.augmentation:
            # print("gaofen2np", gaofen2np.shape, "lidar2np", lidar2np.shape, "mask2np", mask2np.shape)
            # gaofen2np (128, 128, 193) lidar2np (128, 128, 3) mask2np (128, 128, 3)
            gaofen, lidar, mask = self.is_aug(gaofen2np, lidar2np, mask2np)
        else:
            gaofen, lidar, mask = self.no_aug(gaofen2np, lidar2np, mask2np)

        return gaofen, lidar, mask

    def norm(self, gaofen2np, lidar2np):

        _, _, gao_band = gaofen2np.shape
        _, _, lidar_band = lidar2np.shape

        # 归一化
        for i in range(gao_band):
            max = np.max(gaofen2np[:, :, i])
            min = np.min(gaofen2np[:, :, i])
            if max == 0 and min == 0:
                # print(" ############################## skip ############################## ")
                continue
            gaofen2np[:,:,i] = (gaofen2np[:,:,i] - min) / (max-min)

        for i in range(lidar_band):
            max = np.max(lidar2np[:, :, i])
            min = np.min(lidar2np[:, :, i])
            if max == 0 and min == 0:
                # print(" ****************************** skip ****************************** ")
                continue
            lidar2np[:,:,i] = (lidar2np[:,:,i] - min) / (max-min)

        return gaofen2np, lidar2np

    def is_aug(self, gaofen2np, lidar2np, mask2np):
        # gaofen2np=np.array(gaofen)     # (512,512,*)
        _, _, gaofen_band = gaofen2np.shape
        # lidar2np=np.array(lidar)       # (512,512,*)
        _, _, lidar_band = lidar2np.shape
        # mask2np = np.expand_dims(np.array(mask2np), axis=2)  # (512,512,1)

        trans_norm1 = transforms.Normalize(mean=[0.514, 0.498, 0.463, 0.620],
                                           std=[0.188, 0.170, 0.157, 0.164])  # all_data

        trans_norm2 = transforms.Normalize(mean=[0.102, 0.339],
                                          std=[0.099, 0.211])  # all_data

        aug = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.RandomRotation(15)])

        # print("gaofen2np", gaofen2np.shape, "lidar2np", lidar2np.shape, "mask2np", mask2np.shape)
        # gaofen2np (128, 128, 193) lidar2np (128, 128, 3) mask2np (128, 128, 1, 3)
        img = torch.cat((torch.tensor(gaofen2np), torch.tensor(lidar2np), torch.tensor(mask2np)), dim=2)  # (512,512,*)

        img = aug(img.permute(2, 0, 1))
        # print("img", img.shape)                 # 228, 128, 128

        # mask_aug=(img_aug[-1,:,:]).unsqueeze(0)
        # gaofen_aug = trans_norm1(img[0: gaofen_band, :, :])     
        # lidar_aug = trans_norm2(img[gaofen_band : gaofen_band + lidar_band, :, :])
        gaofen_aug = img[0: gaofen_band, :, :]
        lidar_aug = img[gaofen_band : gaofen_band + lidar_band, :, :]
        mask_aug = img[-1, :, :].unsqueeze(0)

        # print("gaofen2np", gaofen_aug.shape, "lidar2np", lidar_aug.shape, "mask2np", mask_aug.shape)

        # return gaofen_aug, lidar_aug[1:2,:,:], mask_aug
        return gaofen_aug, lidar_aug, mask_aug

    def no_aug(self, gaofen2np, lidar2np, mask2np):

        # trans_norm1 = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) #imagenet
        trans_norm1 = transforms.Normalize(mean = [0.514, 0.498, 0.463, 0.620],
                                           std = [0.188, 0.170, 0.157, 0.164])  # all_data

        trans_norm2 = transforms.Normalize(mean = [0.102, 0.339],
                                          std = [0.099, 0.211])  # all_data

        # gaofen = trans_norm1(torch.tensor(gaofen2np).permute(2, 0, 1))
        # lidar = trans_norm2(torch.tensor(lidar2np).permute(2, 0, 1))
        gaofen = torch.tensor(gaofen2np).permute(2, 0, 1)
        lidar = torch.tensor(lidar2np).permute(2, 0, 1)
        # mask = torch.tensor(mask2np).unsqueeze(0)
        mask = torch.tensor(mask2np[:, : , 0]).unsqueeze(0)

        # return gaofen, lidar[1:2,:,:], mask
        return gaofen, lidar, mask

    def __len__(self):
        return len(self.gaofen_imgs)

    # 自定义排序键
    def sort_key(self, filename):
        # 将文件名前缀（数字部分）提取出来并转换为整数
        return int(filename.split('.')[0][5:])


if __name__ == '__main__':
    root = "/home/leo/DatasetMMF/OSDT"
    dataset = Road_loader(root, split='train', img_size=128, is_augmentation=False)
    trainloader = data.DataLoader(dataset, batch_size=2, shuffle=True)
    print(len(dataset))

    for gaofen, lidar, mask in trainloader:
        print(gaofen.shape)
        print(lidar.shape)
        print(mask.shape)
        break
