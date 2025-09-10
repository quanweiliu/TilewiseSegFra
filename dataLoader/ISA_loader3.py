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
    Road = [44, 160, 44]
    nonRoad = [127, 127, 127]

    label_seg = np.zeros(label.shape,dtype=np.uint8)
    label_seg [np.all(label==Road,axis=-1)] = 0
    label_seg [np.all(label==nonRoad,axis=-1)] = 1

    # label_seg = label_seg[:,:,0]  #Just take the first channel, no need for all 3 channels
    
    return label_seg


class ISA_loader3(data.DataLoader):
    def __init__(self,
                 root,
                 txt_name='train.txt',
                 img_size=512,
                 classes=2,
                 is_augmentation=False):
        self.root = root
        self.classes = classes
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size))
        self.augmentation = is_augmentation

        """生成图像文件夹路径与标注(mask)文件夹路径"""
        image_dir = os.path.join(root, 'RGB_1m')
        depth_dir = os.path.join(root, 'Sentinel2')
        mask_dir = os.path.join(root, 'Label_train')
        """读取图像列表-txt文件放在根目录"""
        txt_path = os.path.join(root, txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        """读取文件名-图像images，深度depths，标签masks
        不同文件具有不同的扩展名"""
        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        self.img_dir_train = [os.path.join(image_dir, x + ".tif") for x in file_names]
        self.depth_dir_train = [os.path.join(depth_dir, x + ".tif") for x in file_names]
        self.label_dir_train = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.img_dir_train) == len(self.label_dir_train) and len(self.img_dir_train) == len(
            self.depth_dir_train))

        # self.gaofen_data_path = os.path.join(self.root, self.split, 'images256')
        # # self.gaofen_imgs = sorted(os.listdir(self.gaofen_data_path), key=self.sort_key)
        # self.gaofen_imgs = sorted(os.listdir(self.gaofen_data_path))
        # self.lidar_data_path = os.path.join(self.root, self.split, 'DSM256')
        # # self.lidar_imgs = sorted(os.listdir(self.lidar_data_path), key=self.sort_key)
        # self.lidar_imgs = sorted(os.listdir(self.lidar_data_path))
        # self.mask_data_path = os.path.join(self.root, self.split, 'masks256')
        # # self.masks = sorted(os.listdir(self.mask_data_path), key=self.sort_key)
        # self.masks = sorted(os.listdir(self.mask_data_path))
        
    def __getitem__(self, index):
        # img_name = self.gaofen_imgs[index]
        # # img_id = img_name[20:]
        # gaofen_path = os.path.join(self.gaofen_data_path, img_name)
        # lidar_path = os.path.join(self.lidar_data_path, img_name)
        # mask_path = os.path.join(self.mask_data_path, img_name)

        gaofen2np = rasterio.open(self.img_dir_train[index]).read().transpose(1, 2, 0)
        lidar2np = rasterio.open(self.depth_dir_train[index]).read().transpose(1, 2, 0)[:, :, 0:3].astype(np.float32)
        mask2np = cv2.imread(self.label_dir_train[index], flags=cv2.IMREAD_UNCHANGED)
        # print("gaofen2np", gaofen2np.dtype, lidar2np.dtype, mask2np.dtype)
        # print("gaofen2np", gaofen2np.shape, lidar2np.shape, mask2np.shape)

        # 在这里把三维变成了一维！！！ 0 - 5
        if self.classes == 2 or self.classes == 1:
            mask2np = rgb_to_2D_label(mask2np)
        mask2np = mask2np[:, :, 0].astype(np.int64)

        gaofen2np, lidar2np, mask = self.scaleNorm(gaofen2np, lidar2np, mask2np)
        # print("mask", mask.shape)

        if self.augmentation:
            gaofen, lidar, mask = self.is_aug(gaofen2np, lidar2np, mask)
        # print("mask", mask.shape)
        # else:
        #     gaofen, lidar, mask = self.no_aug(gaofen2np, lidar2np, mask2np)
        gaofen, lidar = self.norm(gaofen2np, lidar2np)
        # mask = mask
        # lidar = lidar.expand(3, -1, -1)
        return gaofen, lidar, mask.long()

    def norm(self, gaofen, lidar):
        gaofen = gaofen.float() / 255.0
        # imagenet
        gaofen = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])(gaofen)
        
        # # vaihingen
        # gaofen = transforms.Normalize(mean=[0.4731, 0.3206, 0.3182], 
        #                                 std=[0.1970, 0.1306, 0.1276])(gaofen)
        # # potsdam
        # gaofen = transforms.Normalize(mean=[0.349, 0.371, 0.347], 
        #                                  std=[0.1196, 0.1164, 0.1197])(gaofen) 

        # potsdam_irrg
        # gaofen = transforms.Normalize(mean=[0.3823, 0.3625, 0.3364], 
        #                                  std=[0.1172, 0.1167, 0.1203])(gaofen)
        lidar = (lidar - lidar.min()) / (lidar.max() - lidar.min())  # → [0, 1]
        return gaofen, lidar


    def scaleNorm(self, gaofen2np, lidar2np, mask2np):
        _, _, gaofen_band = gaofen2np.shape
        _, _, lidar_band = lidar2np.shape
        mask2np = np.expand_dims(mask2np, axis=2)

        scaleF = transforms.Resize(self.img_size)

        gaofen2np = torch.from_numpy(gaofen2np)
        lidar2np = torch.from_numpy(lidar2np)
        mask2np = torch.from_numpy(mask2np)

        gaofen2np = scaleF(gaofen2np.permute(2, 0, 1))
        lidar2np = scaleF(lidar2np.permute(2, 0, 1))
        mask2np = scaleF(mask2np.permute(2, 0, 1))

        return gaofen2np, lidar2np, mask2np
    

    def is_aug(self, gaofen2np, lidar2np, mask2np, gaofen_band=3, lidar_band=3):
        # _, _, gaofen_band = gaofen2np.shape
        # _, _, lidar_band = lidar2np.shape

        aug = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.RandomVerticalFlip(p=0.5),
                                 transforms.RandomRotation(15)])
        
        # print("gaofen2np", gaofen2np.dtype, lidar2np.dtype, mask2np.dtype)
        # print("gaofen2np", gaofen2np.shape, lidar2np.shape, mask2np.shape)

        img = torch.cat((gaofen2np, lidar2np, mask2np), dim=0)  # (512,512,*)
        img = aug(img)
        # print("img", img.dtype, img.shape)  # float32 (7, 256, 256)
        gaofen_aug = img[0: gaofen_band, :, :]
        lidar_aug = img[gaofen_band: gaofen_band + lidar_band, :, :]
        mask_aug = img[-1, :, :].unsqueeze(0)

        return gaofen_aug, lidar_aug, mask_aug


    # def no_aug(self, gaofen2np, lidar2np, mask2np):
    #     gaofen = torch.from_numpy(gaofen2np).permute(2, 0, 1)
    #     lidar = torch.from_numpy(lidar2np).permute(2, 0, 1)
    #     mask = torch.from_numpy(mask2np)

    #     return gaofen, lidar, mask

    def __len__(self):
        return len(self.img_dir_train)

    # 自定义排序键
    def sort_key(self, filename):
        # 将文件名前缀（数字部分）提取出来并转换为整数
        return int(filename.split('.')[0][20:])


if __name__ == '__main__':
    root = "/home/icclab/Documents/lqw/DatasetMMF/ISASeg"
    # dataset = ISA_loader3(root, txt_name='train.txt', img_size=1024, classes=1, is_augmentation=True)
    dataset = ISA_loader3(root, txt_name='train.txt', img_size=1600, classes=1, is_augmentation=False)
    trainloader = data.DataLoader(dataset, batch_size=4, shuffle=True)

    for gaofen, lidar, mask in trainloader:
        print(gaofen.shape, gaofen.dtype, gaofen.max(), gaofen.min())
        print(lidar.shape, lidar.dtype, lidar.max(), lidar.min())
        print(mask.shape, mask.dtype, mask.max(), mask.min(), np.unique(mask))
        break

    # dataset = ISPRS_loader(root, split='val', img_size=256, is_augmentation=False)
    # trainloader = data.DataLoader(dataset, batch_size=4, shuffle=True)
    # for gaofen, lidar, mask in trainloader:
    #     print(gaofen.shape, gaofen.dtype, gaofen.max(), gaofen.min())
    #     print(lidar.shape, lidar.dtype, lidar.max(), lidar.min())
    #     print(mask.shape, mask.dtype, mask.max(), mask.min())
    #     break