import os
import torch
import numpy as np
import scipy.io as scio
from torch.utils import data
from torchvision import transforms


class OSTD_loader(data.DataLoader):
    def __init__(self,
                  root,
                  split='train',
                  img_size=512,
                  is_augmentation=False):
        self.root = root
        self.split = split
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size))
        
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

        if self.augmentation:
            gaofen, lidar, mask = self.is_aug(gaofen2np, lidar2np, mask2np)
        else:
            gaofen, lidar, mask = self.no_aug(gaofen2np, lidar2np, mask2np)

        gaofen, lidar = self.norm(gaofen, lidar)
        return gaofen, lidar, mask.long()

    def norm(self, gaofen, lidar):
        _, _, gao_band = gaofen.shape

        # # 归一化
        # for i in range(gao_band):
        #     max = np.max(gaofen[i, :, :])
        #     min = np.min(gaofen[i, :, :])
        #     if max == 0 and min == 0:
        #         # print(" ############################## skip ############################## ")
        #         continue
        #     gaofen[i, :, :] = (gaofen[i, :, :] - min) / (max-min)
        # _, _, lidar_band = lidar.shape
        
        lidar = (lidar - lidar.min()) / (lidar.max() - lidar.min())  # → [0, 1]
        return gaofen, lidar

    def is_aug(self, gaofen2np, lidar2np, mask2np):
        _, _, gaofen_band = gaofen2np.shape
        _, _, lidar_band = lidar2np.shape

        aug = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.RandomRotation(15)])

        img = torch.cat((torch.from_numpy(gaofen2np), 
                         torch.from_numpy(lidar2np), 
                         torch.from_numpy(mask2np)), dim=2)  # (512,512,*)
        img = aug(img.permute(2, 0, 1))
        gaofen_aug = img[0: gaofen_band, :, :]
        lidar_aug = img[gaofen_band: gaofen_band + lidar_band, :, :]
        mask_aug = img[-1, :, :].unsqueeze(0)

        return gaofen_aug, lidar_aug, mask_aug

    def no_aug(self, gaofen2np, lidar2np, mask2np):
        gaofen = torch.from_numpy(gaofen2np).permute(2, 0, 1)
        lidar = torch.from_numpy(lidar2np).permute(2, 0, 1)
        mask = torch.from_numpy(mask2np[:, :, 0]).unsqueeze(0)

        return gaofen, lidar, mask

    def __len__(self):
        return len(self.gaofen_imgs)

    # 自定义排序键
    def sort_key(self, filename):
        # 将文件名前缀（数字部分）提取出来并转换为整数
        return int(filename.split('.')[0][5:])


if __name__ == '__main__':
    root = "/home/icclab/Documents/lqw/DatasetMMF/OSTD"
    dataset = OSTD_loader(root, split='train', img_size=128, is_augmentation=False)
    trainloader = data.DataLoader(dataset, batch_size=2, shuffle=True)
    print(len(dataset))

    for gaofen, lidar, mask in trainloader:
        print(gaofen.shape, gaofen.dtype, gaofen.min(), gaofen.max())
        print(lidar.shape, lidar.dtype, lidar.min(), lidar.max())
        print(mask.shape, mask.dtype, mask.min(), mask.max())
        break
