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


class RGBD_loader(data.DataLoader):
    def __init__(self,
                 root,
                 split='train',
                 img_size=[480, 640],
                 classes=6,
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

        image_dir = os.path.join(root, 'images')
        depth_dir = os.path.join(root, 'depths')
        mask_dir = os.path.join(root, 'labels')

        txt_path = os.path.join(root, split + '.txt')
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)

        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]


        self.img_dir_train = [os.path.join(image_dir, x + ".png") for x in file_names]
        self.depth_dir_train = [os.path.join(depth_dir, x + ".png") for x in file_names]
        self.label_dir_train = [os.path.join(mask_dir, x + ".png") for x in file_names]

        
    def __getitem__(self, index):

        gaofen2np = cv2.imread(self.img_dir_train[index], cv2.COLOR_BGR2RGB).astype(np.float32)
        lidar2np = cv2.imread(self.depth_dir_train[index], cv2.IMREAD_UNCHANGED).astype(np.float32)
        lidar2np = np.expand_dims(lidar2np, axis=2)
        mask2np = cv2.imread(self.label_dir_train[index], cv2.IMREAD_UNCHANGED).astype(np.int16)

        H, W, C = gaofen2np.shape
        if H != self.img_size[0] or W != self.img_size[1]:
            print("Warning: image size not equal to setting size, resize it")
            gaofen2np, lidar2np, mask2np = self.scaleNorm(gaofen2np, lidar2np, mask2np)
        
        # (480, 640, 3) (480, 640) (480, 640)
        # print(gaofen2np.shape, lidar2np.shape, mask2np.shape)

        if self.augmentation:
            gaofen, lidar, mask = self.is_aug(gaofen2np, lidar2np, mask2np)
        else:
            gaofen, lidar, mask = self.no_aug(gaofen2np, lidar2np, mask2np)
        gaofen, lidar = self.norm(gaofen, lidar)
        # lidar = lidar.expand(3, -1, -1)
        mask -= 1
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
            depth = depth / 1000

            gaofen = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])(gaofen)
            depth = transforms.Normalize(mean=[2.8424503515351494],
                                            std=[0.9932836506164299])(depth)
            
        # else:
        #     raise NameError("normalization {} is not implemented".format(self.normalization))

        return gaofen, depth

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
        return len(self.img_dir_train)

    # 自定义排序键
    def sort_key(self, filename):
        # 将文件名前缀（数字部分）提取出来并转换为整数
        return int(filename.split('.')[0][20:])


if __name__ == '__main__':
    root = "/home/icclab/Documents/lqw/DatasetMMF/NYUv2"
    # dataset = ISPRS_loader(root, split='train', img_size=256, is_augmentation=False)
    dataset = RGBD_loader(root, 
                           split='train', 
                           img_size=[480, 640], 
                           classes=40, 
                           data_name="NYUv2", 
                           normalization="standard", 
                           is_augmentation=False)
    # dataset = ISPRS_loader(root, split='train', img_size=256, classes=6, data_name="Vaihingen", is_augmentation=False)
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