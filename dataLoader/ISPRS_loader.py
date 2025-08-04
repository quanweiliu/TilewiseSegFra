
import os
import torch
from PIL import Image
import tifffile
import rasterio
import numpy as np
import scipy.io as scio
from torch.utils import data
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


def rgb_to_2D_label(label):
    """
    Suply our label masks as input in RGB format. 
    Replace pixels with specific RGB values ...
    """
    Impervious = [255, 255, 255]
    Building = [0, 0, 255]
    Vegetation = [0, 255, 255]
    Tree = [0, 255, 0]
    Car = [255, 255, 0]
    Clutter = [255, 0, 0]

    label_seg = np.zeros(label.shape,dtype=np.uint8)
    label_seg [np.all(label==Impervious,axis=-1)] = 0
    label_seg [np.all(label==Building,axis=-1)] = 1
    label_seg [np.all(label==Vegetation,axis=-1)] = 2
    label_seg [np.all(label==Tree,axis=-1)] = 3
    label_seg [np.all(label==Car,axis=-1)] = 4
    label_seg [np.all(label==Clutter,axis=-1)] = 5

    # label_seg = label_seg[:,:,0]  #Just take the first channel, no need for all 3 channels
    
    return label_seg


class ISPRS_loader(data.DataLoader):
    def __init__(self,
                 root,
                 split='train',
                 img_size=512,
                 classes=6,
                 is_augmentation=False):
        self.root = root
        self.split = split
        self.classes = classes
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size))
        
        # self.tf = transforms.Compose([transforms.ToTensor()])
        self.gaofen_data_path = os.path.join(self.root, self.split, 'images256')
        # self.gaofen_imgs = sorted(os.listdir(self.gaofen_data_path), key=self.sort_key)
        self.gaofen_imgs = sorted(os.listdir(self.gaofen_data_path))
        self.lidar_data_path = os.path.join(self.root, self.split, 'DSM256')
        # self.lidar_imgs = sorted(os.listdir(self.lidar_data_path), key=self.sort_key)
        self.lidar_imgs = sorted(os.listdir(self.lidar_data_path))
        self.mask_data_path = os.path.join(self.root, self.split, 'masks256')
        # self.masks = sorted(os.listdir(self.mask_data_path), key=self.sort_key)
        self.masks = sorted(os.listdir(self.mask_data_path))
        self.augmentation = is_augmentation
        
    def __getitem__(self, index):
        img_name = self.gaofen_imgs[index]
        # img_name1 = self.gaofen_imgs[index]
        # img_name2 = self.lidar_imgs[index]
        img_id = img_name[20:]
        # img_id1 = img_name1[20:]
        # img_id2 = img_name2[22:]
        # print("img_id1", img_id1)
        # print("img_id2", img_id2)

        gaofen_path = os.path.join(self.gaofen_data_path, "top_mosaic_09cm_area" + str(img_id))
        lidar_path = os.path.join(self.lidar_data_path, "dsm_09cm_matching_area" + str(img_id))
        mask_path = os.path.join(self.mask_data_path, "top_mosaic_09cm_area" + str(img_id))

        gaofen2np = rasterio.open(gaofen_path).read().transpose(1, 2, 0)
        lidar2np = rasterio.open(lidar_path).read().transpose(1, 2, 0)
        mask2np = rasterio.open(mask_path).read().transpose(1, 2, 0)
        # print("gaofen2np", gaofen2np.shape)
        # print("lidar2np", lidar2np.shape)
        # print("mask2np", mask2np.shape)
        
        # 在这里把三维变成了一维！！！ 0 - 5
        if self.classes == 6:
            mask2np = rgb_to_2D_label(mask2np)
        mask2np = mask2np[:, :, 0]
        # print("gaofen2np", gaofen2np.shape)
        # print("lidar2np", lidar2np.shape)
        # print("mask2np", mask2np.shape)


        if self.augmentation:
            gaofen, lidar, mask = self.is_aug(gaofen2np, lidar2np, mask2np)
        else:
            gaofen, lidar, mask = self.no_aug(gaofen2np, lidar2np, mask2np)

        return gaofen, lidar, mask


    def is_aug(self, gaofen2np, lidar2np, mask2np):
        _, _, gaofen_band = gaofen2np.shape
        _, _, lidar_band = lidar2np.shape
        mask2np = np.expand_dims(mask2np, axis=2)
        # print("gaofen2np", gaofen2np.shape)
        # print("lidar2np", lidar2np.shape)
        # print("mask2np", mask2np.shape)
        # trans_norm1 = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) #imagenet
        # trans_norm1 = transforms.Normalize(mean=[0.514, 0.498, 0.463, 0.620],
        #                                    std=[0.188, 0.170, 0.157, 0.164])  # all_data
        # trans_norm1 = transforms.Normalize(mean=[0.515,0.499,0.463,0.625],
        #                                    std=[0.188,0.171,0.157,0.164])  # test_img
        # trans_norm2 = transforms.Normalize(mean=[0.102, 0.339],
        #                                   std=[0.099, 0.211])  # all_data


        aug = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.RandomRotation(15)])

        img = torch.cat((torch.tensor(gaofen2np), torch.tensor(lidar2np), torch.tensor(mask2np)), dim=2)  # (512,512,*)
        img = aug(img.permute(2, 0, 1))
        gaofen_aug = img[0 : gaofen_band, :, :]
        lidar_aug = img[gaofen_band : gaofen_band + lidar_band, :, :]
        mask_aug = img[-1, :, :].long()


        gaofen_aug = gaofen_aug.float() / 255.0
        gaofen_aug = transforms.Normalize(mean=[0.485,0.456,0.406], 
                                         std=[0.229,0.224,0.225])(gaofen_aug)
        
        lidar_aug = (lidar_aug - lidar_aug.min()) / (lidar_aug.max() - lidar_aug.min())  # → [0, 1]


        # return gaofen_aug, lidar_aug[1:2,:,:], mask_aug
        return gaofen_aug, lidar_aug, mask_aug

    def no_aug(self, gaofen2np, lidar2np, mask2np):

        # trans_norm1 = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) #imagenet
        # trans_norm1 = transforms.Normalize(mean = [0.514, 0.498, 0.463, 0.620],
                                        #    std = [0.188, 0.170, 0.157, 0.164])  # all_data
        # trans_norm1 = transforms.Normalize(mean=[0.515,0.499,0.463,0.625],
        #                                    std=[0.188,0.171,0.157,0.164])  # test_img
        
        # trans_norm2 = transforms.Normalize(mean = [0.102, 0.339],
                                        #   std = [0.099, 0.211])  # all_data

        gaofen = torch.tensor(gaofen2np).permute(2, 0, 1)
        lidar = torch.tensor(lidar2np).permute(2, 0, 1)
        mask= torch.tensor(mask2np.astype(np.int64))

        gaofen = gaofen.float() / 255.0
        gaofen = transforms.Normalize(mean=[0.485,0.456,0.406], 
                                         std=[0.229,0.224,0.225])(gaofen)
        
        lidar = (lidar - lidar.min()) / (lidar.max() - lidar.min())  # → [0, 1]

        # return gaofen, lidar[1:2,:,:], mask
        return gaofen, lidar, mask

    def __len__(self):
        return len(self.gaofen_imgs)

    # # 自定义排序键
    # def sort_key(self, filename):
    #     # 将文件名前缀（数字部分）提取出来并转换为整数
    #     return int(filename.split('.')[0][5:])


if __name__ == '__main__':
    root = "/home/icclab/Documents/lqw/DatasetMMF/Vaihingen"
    # dataset = ISPRS_loader(root, split='train', img_size=256, is_augmentation=False)
    dataset = ISPRS_loader(root, split='train', img_size=256, is_augmentation=True)
    trainloader = data.DataLoader(dataset, batch_size=4, shuffle=True)

    for gaofen, lidar, mask in trainloader:
        print(gaofen.shape, gaofen.dtype)
        print(lidar.shape, lidar.dtype)
        print(mask.shape, mask.dtype)
        break

