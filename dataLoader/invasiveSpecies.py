import os
import cv2
import torch
import rasterio
import numpy as np
import scipy.io as scio
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import v2
import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


def rgb_to_2D_label(label):
    """
    Suply our label masks as input in RGB format. 
    Replace pixels with specific RGB values ...
    """
    # present = [13, 143, 83]
    # present = [13, 83, 143]
    present = [83, 143, 13]
    # present = [83, 13, 143]
    # present = [143, 13, 83]
    # present = [143, 83, 13]
    absent = [0, 0, 0]

    label_seg = np.zeros(label.shape,dtype=np.uint8)
    label_seg [np.all(label==present,axis=-1)] = 1
    label_seg [np.all(label==absent,axis=-1)] = 0

    # label_seg = label_seg[:,:,0]  #Just take the first channel, no need for all 3 channels
    
    return label_seg


class InvasiveSpecies(data.DataLoader):
    def __init__(self,
                 root,
                 split='train',
                 img_size=[512, 512],
                 classes=2,
                 data_name="NYUv2",
                 normalization="minMax",
                 is_augmentation=False):
        self.root = root
        self.split = split
        self.classes = classes
        self.img_size = img_size
        self.data_name = data_name
        self.normalization = normalization
        self.augmentation = is_augmentation


        self.img_data_path = os.path.join(self.root, self.split, 'images')
        # print("self.img_dir: ", self.root, self.split, self.img_dir)
        self.img_dir = sorted(os.listdir(self.img_data_path))
        # print("self.img_dir: ", self.img_dir)
        # self.lidar_data_path = os.path.join(self.root, self.split, 'DSM256')
        # self.lidar_imgs = sorted(os.listdir(self.lidar_data_path))
        self.label_data_path = os.path.join(self.root, self.split, 'labels')
        self.label_dir = sorted(os.listdir(self.label_data_path))

        
    def __getitem__(self, index):
        gaofen_path = os.path.join(self.img_data_path, self.img_dir[index])
        # lidar_path = os.path.join(self.lidar_data_path, self.img_dir[index])
        mask_path = os.path.join(self.label_data_path, self.label_dir[index])

        gaofen2np = cv2.imread(gaofen_path, cv2.COLOR_BGR2RGB).astype(np.float32)
        # lidar2np = cv2.imread(self.depth_dir_train[index], cv2.IMREAD_UNCHANGED).astype(np.float32)
        # lidar2np = np.expand_dims(lidar2np, axis=2)
        lidar2np = np.zeros((gaofen2np.shape[0], gaofen2np.shape[1], 1), dtype=np.float32)  # Create a dummy depth map
        mask2np = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.int16)
        # print("mask2np", mask2np, mask2np.shape, np.unique(mask2np))
        # print("gaofen2np.shape: ", gaofen2np.shape)
        # print("lidar2np.shape: ", lidar2np.shape)
        # print("mask2np.shape: ", mask2np.shape)

        H, W, C = gaofen2np.shape
        if H != self.img_size[0] or W != self.img_size[1]:
            # print("Warning: image size not equal to setting size, resize it")
            gaofen2np, lidar2np, mask2np = self.scaleNorm(gaofen2np, lidar2np, mask2np)
        
        # 在这里把三维变成了一维！！！ 0 - 5
        if self.classes == 1:
            # print("#"*10)
            mask2np = rgb_to_2D_label(mask2np)
            # print("mask2np", np.unique(mask2np))
        # mask2np = mask2np[:, :, 0].astype(np.int64)

        if self.augmentation:
            gaofen, lidar, mask = self.is_aug(gaofen2np, lidar2np, mask2np)
        else:
            gaofen, lidar, mask = self.no_aug(gaofen2np, lidar2np, mask2np)

            # 合成数据
            # lidar = v2.RandomPerspective(distortion_scale=0.1, p=1)(lidar)

        gaofen, lidar = self.norm(gaofen, lidar)
        # lidar = lidar.expand(3, -1, -1)
        # mask -= 1
        # mask -= torch.tensor(1) 
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
        
    def norm(self, gaofen, depth):
        # "https://github.com/jsten07/CNNvsTransformer/blob/2273b7f72de7aad00d7abc5a5c35f8c81ec62d4d/Notebooks/count_classes.ipynb#L257"


        if self.normalization == "minMax":
            gaofen = gaofen.float()
            gao_band, _, _ = gaofen.shape
            # 归一化
            for i in range(gao_band):
                max = torch.max(gaofen[i, :, :])
                min = torch.min(gaofen[i, :, :])
                if max == 0 and min == 0:
                    # print(" ############################## skip ############################## ")
                    continue
                gaofen[i, :, :] = (gaofen[i, :, :] - min) / (max-min)
            depth = (depth - depth.min()) / (depth.max() - depth.min())  # → [0, 1]

        elif self.normalization == "standard":
            gaofen = gaofen.float() / 255.0
            # depth = depth / 1000

            gaofen = v2.Normalize(mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])(gaofen)
            depth = v2.Normalize(mean=[2.8424503515351494],
                                            std=[0.9932836506164299])(depth)
            
        # else:
        #     raise NameError("normalization {} is not implemented".format(self.normalization))

        return gaofen, depth

    def is_aug(self, gaofen2np, lidar2np, mask2np):
        _, _, gaofen_band = gaofen2np.shape
        _, _, lidar_band = lidar2np.shape
        # mask2np = np.expand_dims(mask2np, axis=2)

        aug = v2.Compose([v2.RandomHorizontalFlip(p=0.5),
                                v2.RandomVerticalFlip(p=0.5),
                                v2.RandomRotation(15)])

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
        return len(self.img_dir)

    # 自定义排序键
    def sort_key(self, filename):
        # 将文件名前缀（数字部分）提取出来并转换为整数
        
        return int(filename.split('.')[0][20:])


if __name__ == '__main__':
    # root = "/home/icclab/Documents/lqw/DatasetMMF/NYUv2"
    root = "/home/icclab/Documents/lqw/DatasetSMD/InvasiveSpecies"
    # dataset = ISPRS_loader(root, split='train', img_size=256, is_augmentation=False)
    dataset = InvasiveSpecies(root, 
                           split='train', 
                           img_size=[512, 512], 
                           classes=2, 
                           data_name="InvasiveSpecies", 
                           normalization="standard", 
                           is_augmentation=False)
    # dataset = ISPRS_loader(root, split='train', img_size=256, classes=6, data_name="Vaihingen", is_augmentation=False)
    trainloader = data.DataLoader(dataset, batch_size=2, shuffle=True)
    print(len(dataset))

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