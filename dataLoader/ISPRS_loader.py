import os
import cv2
import torch
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
                 data_name="Vaihingen",
                 is_augmentation=False):
        self.root = root
        self.split = split
        self.classes = classes
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size))
        self.data_name = data_name
        
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
        # img_id = img_name[20:]
        gaofen_path = os.path.join(self.gaofen_data_path, img_name)
        lidar_path = os.path.join(self.lidar_data_path, img_name)
        mask_path = os.path.join(self.mask_data_path, img_name)

        gaofen2np = rasterio.open(gaofen_path).read().transpose(1, 2, 0)
        lidar2np = rasterio.open(lidar_path).read().transpose(1, 2, 0)
        mask2np = rasterio.open(mask_path).read().transpose(1, 2, 0)
        # print(gaofen2np.shape, lidar2np.shape, mask2np.shape)

        real_image_size = gaofen2np.shape[-2]
        if real_image_size != self.img_size[0]:
            print("Warning: image size not equal to setting size, resize it")
            gaofen2np, lidar2np, mask2np = self.scaleNorm(gaofen2np, lidar2np, mask2np)
        
        # 在这里把三维变成了一维！！！ 0 - 5
        if self.classes == 6:
            mask2np = rgb_to_2D_label(mask2np)
        mask2np = mask2np[:, :, 0].astype(np.int64)

        if self.augmentation:
            gaofen, lidar, mask = self.is_aug(gaofen2np, lidar2np, mask2np)
        else:
            gaofen, lidar, mask = self.no_aug(gaofen2np, lidar2np, mask2np)
        gaofen, lidar = self.norm(gaofen, lidar)
        # lidar = lidar.expand(3, -1, -1)
        return gaofen, lidar, mask.long()

    def scaleNorm(self, gaofen2np, lidar2np, mask2np):
        # resize the image
        gaofen2np = cv2.resize(gaofen2np, self.img_size, cv2.INTER_LINEAR)
        lidar2np = np.stack([
                        cv2.resize(lidar2np[:, :, c], \
                        self.img_size, \
                        interpolation=cv2.INTER_LINEAR)
                        for c in range(lidar2np.shape[2])
                        ], axis=2)
        mask2np = cv2.resize(mask2np, self.img_size, cv2.INTER_LINEAR)
        # print(gaofen2np.shape, lidar2np.shape, mask2np.shape)

        return gaofen2np, lidar2np, mask2np
        
    def norm(self, gaofen, lidar):
        "https://github.com/jsten07/CNNvsTransformer/blob/2273b7f72de7aad00d7abc5a5c35f8c81ec62d4d/Notebooks/count_classes.ipynb#L257"
        gaofen = gaofen.float() / 255.0

        if self.data_name == "Vaihingen":
            gaofen = transforms.Normalize(mean=[0.4731, 0.3206, 0.3182], 
                                            std=[0.1970, 0.1306, 0.1276])(gaofen)
        elif self.data_name == "Potsdam":
            gaofen = transforms.Normalize(mean=[0.349, 0.371, 0.347], 
                                            std=[0.1196, 0.1164, 0.1197])(gaofen) 
        else: # imagenet
            gaofen = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])(gaofen)
        # potsdam_irrg
        # gaofen = transforms.Normalize(mean=[0.3823, 0.3625, 0.3364], 
        #                                  std=[0.1172, 0.1167, 0.1203])(gaofen)
        
        # # Min-Max 归一化（缩放到固定区间）
        lidar = (lidar - lidar.min()) / (lidar.max() - lidar.min())  # → [0, 1]

        # Z-score 标准化（标准差归一化）
        # lidar = (lidar - lidar.mean(dim=(1, 2), keepdim=True)) / (lidar.std(dim=(1, 2), keepdim=True) + 1e-6)
        return gaofen, lidar

    def is_aug(self, gaofen2np, lidar2np, mask2np):
        _, _, gaofen_band = gaofen2np.shape
        _, _, lidar_band = lidar2np.shape
        mask2np = np.expand_dims(mask2np, axis=2)

        aug = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.RandomRotation(15)])

        img = torch.cat((torch.from_numpy(gaofen2np), 
                         torch.from_numpy(lidar2np), 
                         torch.from_numpy(mask2np)), dim=2)  # (512,512,*)
        img = aug(img.permute(2, 0, 1))
        gaofen_aug = img[0: gaofen_band, :, :]
        lidar_aug = img[gaofen_band: gaofen_band + lidar_band, :, :]
        mask_aug = img[-1, :, :]

        return gaofen_aug, lidar_aug, mask_aug

    def no_aug(self, gaofen2np, lidar2np, mask2np):
        gaofen = torch.from_numpy(gaofen2np).permute(2, 0, 1)
        lidar = torch.from_numpy(lidar2np).permute(2, 0, 1)
        mask = torch.from_numpy(mask2np)

        return gaofen, lidar, mask

    def __len__(self):
        return len(self.gaofen_imgs)

    # 自定义排序键
    def sort_key(self, filename):
        # 将文件名前缀（数字部分）提取出来并转换为整数
        return int(filename.split('.')[0][20:])


if __name__ == '__main__':
    root = "/home/icclab/Documents/lqw/DatasetMMF/Vaihingen"
    root = "/home/icclab/Documents/lqw/DatasetMMF/Potsdam"
    # dataset = ISPRS_loader(root, split='train', img_size=256, is_augmentation=False)
    dataset = ISPRS_loader(root, split='train', img_size=512, classes=6, data_name="Potsdam", is_augmentation=False)
    trainloader = data.DataLoader(dataset, batch_size=4, shuffle=True)

    for gaofen, lidar, mask in trainloader:
        print(gaofen.shape, gaofen.dtype, gaofen.max(), gaofen.min())
        print(lidar.shape, lidar.dtype, lidar.max(), lidar.min())
        print(mask.shape, mask.dtype, mask.max(), mask.min())
        break

    # dataset = ISPRS_loader(root, split='val', img_size=256, is_augmentation=False)
    # trainloader = data.DataLoader(dataset, batch_size=4, shuffle=True)
    # for gaofen, lidar, mask in trainloader:
    #     print(gaofen.shape, gaofen.dtype, gaofen.max(), gaofen.min())
    #     print(lidar.shape, lidar.dtype, lidar.max(), lidar.min())
    #     print(mask.shape, mask.dtype, mask.max(), mask.min())
    #     break