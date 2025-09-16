import os
import cv2
import torch
import rasterio
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

import matplotlib
import matplotlib.colors
import random
import torchvision


class ISA_loader2_test(Dataset):
    def __init__(self, transform=None, phase_train=True, data_dir=None, txt_name='train.txt'):
        self.phase_train = phase_train
        self.transform = transform

        root = data_dir
        assert os.path.exists(root), "path '{}' does not exist.".format(root)

        """生成图像文件夹路径与标注(mask)文件夹路径"""
        image_dir = os.path.join(root, 'RGB_1m')
        # depth_dir = os.path.join(root, 'Sentinel1')
        depth_dir = os.path.join(root, 'Sentinel2')

        """读取图像列表-txt文件放在根目录"""
        txt_path = os.path.join(root, txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)

        """读取文件名-图像images，深度depths，标签masks
        不同文件具有不同的扩展名"""
        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        self.img_dir_train = [os.path.join(image_dir, x + ".tif") for x in file_names]
        self.depth_dir_train = [os.path.join(depth_dir, x + ".tif") for x in file_names]

        assert len(self.img_dir_train) == len(self.depth_dir_train)

    def __len__(self):
        return len(self.img_dir_train)

    def __getitem__(self, idx):

        img_dir = self.img_dir_train
        depth_dir = self.depth_dir_train
        # print("img_dir", img_dir[idx])
        # print("depth_dir", depth_dir[idx])
        # print("label_dir", label_dir[idx])

        # print("label", label.shape, label.dtype)
        image = self._open_image(img_dir[idx], cv2.COLOR_BGR2RGB)
        
        depth = rasterio.open(depth_dir[idx]).read()
        depth = depth.astype(np.float32).transpose(1, 2, 0)
        # channel_0 = depth[:, :, 0:1] # 使用切片保持维度
        # depth = np.concatenate((depth, channel_0), axis=2)

        # print("image", image.shape, image.dtype)
        # print("depth", depth.shape, depth.dtype)
        # print("label", label.shape, label.dtype)

        # print("image 2", image.shape)
        # print("depth 2", depth.shape)
        # print("label 2", label.shape)

        sample = {'image': image, 'depth': depth}

        if self.transform:
            sample = self.transform(sample)

        return sample

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        img = np.array(cv2.imread(filepath, mode), dtype=dtype)
        return img


class RandomHSV(object):
    """
        Args:
            h_range (float tuple): random ratio of the hue channel,
                new_h range from h_range[0]*old_h to h_range[1]*old_h.
            s_range (float tuple): random ratio of the saturation channel,
                new_s range from s_range[0]*old_s to s_range[1]*old_s.
            v_range (int tuple): random bias of the value channel,
                new_v range from old_v-v_range to old_v+v_range.
        Notice:
            h range: 0-1
            s range: 0-1
            v range: 0-255
        """

    def __init__(self, h_range, s_range, v_range):
        assert isinstance(h_range, (list, tuple)) and \
               isinstance(s_range, (list, tuple)) and \
               isinstance(v_range, (list, tuple))
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range

    def __call__(self, sample):
        img = sample['image']
        img_hsv = matplotlib.colors.rgb_to_hsv(img)
        img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        h_random = np.random.uniform(min(self.h_range), max(self.h_range))
        s_random = np.random.uniform(min(self.s_range), max(self.s_range))
        v_random = np.random.uniform(-min(self.v_range), max(self.v_range))
        img_h = np.clip(img_h * h_random, 0, 1)
        img_s = np.clip(img_s * s_random, 0, 1)
        img_v = np.clip(img_v + v_random, 0, 255)
        img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        img_new = matplotlib.colors.hsv_to_rgb(img_hsv)

        return {'image': img_new, 'depth': sample['depth']}


# # make sure the image, depth, and label have the same shape
# image = cv2.resize(image, (1600, 1600), cv2.INTER_LINEAR)
# depth = np.stack([
#                 cv2.resize(depth[:, :, c], (160, 160), interpolation=cv2.INTER_LINEAR)
#                 for c in range(depth.shape[2])
#                 ], axis=2)
# label = cv2.resize(label, (1600, 1600), cv2.INTER_LINEAR)
# label = np.all(label == [44, 160, 44], axis=-1).astype(np.uint8)
# # print("image 2", image.shape)
# # print("depth 2", depth.shape)
# # print("label 2", label.shape)


class scaleNorm(object):
    def __init__(self, th=1600, tw=1600):
        self.target_height = th
        self.target_width = tw

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        # Bi-linear
        image = cv2.resize(image, (self.target_width, self.target_height), cv2.INTER_LINEAR)
        # Nearest-neighbor
        # depth = cv2.resize(depth, (image_w, image_h), cv2.INTER_NEAREST)
        depth = np.stack([
                        cv2.resize(depth[:, :, c], \
                        (self.target_width, self.target_height), \
                        interpolation=cv2.INTER_LINEAR)
                        for c in range(depth.shape[2])
                        ], axis=2)

        return {'image': image, 'depth': depth}


class RandomScale(object):
    def __init__(self, scale):
        self.scale_low = min(scale)
        self.scale_high = max(scale)

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth']

        target_scale = random.uniform(self.scale_low, self.scale_high)
        # (H, W, C)
        target_height = int(round(target_scale * image.shape[0]))
        target_width = int(round(target_scale * image.shape[1]))

        # Bi-linear
        image = cv2.resize(image, (target_width, target_height), cv2.INTER_LINEAR)
        # Nearest-neighbor
        depth = cv2.resize(depth, (target_width, target_height), cv2.INTER_NEAREST)
        # print("image", image.shape, image.dtype)
        # print("depth", depth.shape, depth.dtype)

        return {'image': image, 'depth': depth}


# class RandomCrop(object):
#     def __init__(self, th, tw):
#         self.target_width = th
#         self.target_height = tw

#     def __call__(self, sample):
#         image, depth, label = sample['image'], sample['depth'], sample['label']
#         h = image.shape[0]
#         w = image.shape[1]
#         i = random.randint(0, h - self.target_height)
#         j = random.randint(0, w - self.target_width)


#         # Bi-linear
#         image = cv2.resize(image, (self.target_width, self.target_height), cv2.INTER_LINEAR)
#         # Nearest-neighbor
#         depth = cv2.resize(depth, (self.target_width, self.target_height), cv2.INTER_NEAREST)
#         label = cv2.resize(label, (self.target_width, self.target_height), cv2.INTER_NEAREST)


#         return {'image': image, 'depth': depth, 'label': label}


class RandomCrop(object):
    def __init__(self, th, tw):
        self.target_height = th
        self.target_width = tw

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        h = image.shape[0]
        w = image.shape[1]
        i = random.randint(0, h - self.target_height)
        j = random.randint(0, w - self.target_width)

        return {'image': image[i:i + self.target_height, j:j + self.target_width, :],
                'depth': depth[i:i + self.target_height, j:j + self.target_width]}


class RandomFlip(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            depth = np.fliplr(depth).copy()

        return {'image': image, 'depth': depth}


# Transforms on torch.*Tensor
class Normalize(object):
    def __call__(self, sample):
        image = sample['image']
        depth = sample['depth']
        image = image / 255
        # depth = depth / 1000
        # image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                          std=[0.229, 0.224, 0.225])(image)
        image = torchvision.transforms.Normalize(mean=[0.4850042694973687, 0.41627756261047333, 0.3981809741523051],
                                                 std=[0.26415541082494515, 0.2728415392982039, 0.2831175140191598])(
            image)
        # depth = torchvision.transforms.Normalize(mean=[2.8424503515351494],
        #                                          std=[0.9932836506164299])(depth)

        depth = (depth - depth.mean(dim=(1, 2), keepdim=True)) / (depth.std(dim=(1, 2), keepdim=True) + 1e-6)

        sample['image'] = image
        sample['depth'] = depth

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        # # Generate different label scales
        # label3 = cv2.resize(label, (w // 4, h // 4), cv2.INTER_NEAREST)
        # label4 = cv2.resize(label, (w // 8, h // 8), cv2.INTER_NEAREST)
        # label5 = cv2.resize(label, (w // 16, h // 16), cv2.INTER_NEAREST)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        depth = depth.transpose((2, 0, 1))
        # depth = np.expand_dims(depth, 0).astype(np.float64)
        return {'image': torch.from_numpy(image).float(),
                'depth': torch.from_numpy(depth).float()}
                # 'label3': torch.from_numpy(label3).float(),
                # 'label4': torch.from_numpy(label4).float(),
                # 'label5': torch.from_numpy(label5).float()}


if __name__ == '__main__':

    image_h = 400
    image_w = 400

    data_dir = "/home/icclab/Documents/lqw/DatasetMMF/ISASeg"
    # data_dir = "/home/icclab/Documents/lqw/DatasetMMF/ISASeg_train"

    train_data = ISA_loader2_test(transform=transforms.Compose([
                                                            scaleNorm(),
                                                            # RandomScale((1.0, 1.4, 2.0)),
                                                            # RandomHSV((0.9, 1.1),
                                                            #         (0.9, 1.1),
                                                            #         (25, 25)),
                                                            RandomCrop(th=image_h, tw=image_w),
                                                            RandomFlip(),
                                                            ToTensor(),
                                                            Normalize()]),
                                                            phase_train=True,
                                                            data_dir = data_dir)
    train_loader = DataLoader(train_data, \
                            batch_size=4, \
                            shuffle=True, \
                            num_workers=0, \
                            pin_memory=False)

    val_data = ISA_loader2_test(transform=transforms.Compose([scaleNorm(),
                                                         ToTensor(),
                                                         Normalize()]),
                                 phase_train=False,
                                 data_dir=data_dir,
                                 txt_name='val.txt'
                                 )
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    for sample in train_loader:
        print(sample["image"].shape, sample["image"].dtype, sample["image"].max(), sample["image"].min())    # torch.Size([2, 3, 480, 640])
        print(sample["depth"].shape, sample["depth"].dtype, sample["depth"].max(), sample["depth"].min())   # torch.Size([2, 1, 480, 640])
        # print(sample["label"].shape, sample["label"].dtype)   # torch.Size([2, 480, 640])
        # print(np.max(sample["image"].numpy()), np.min(sample["image"].numpy()))
        # print(np.max(sample["depth"].numpy()), np.min(sample["depth"].numpy()))
        break 


    # for sample in val_loader:
    #     print(sample["image"].shape, sample["image"].dtype)   # torch.Size([2, 3, 480, 640])
    #     print(sample["depth"].shape, sample["depth"].dtype)   # torch.Size([2, 1, 480, 640])
    #     # print(sample["label"].shape, sample["label"].dtype)   # torch.Size([2, 480, 640])
    #     # print(np.max(sample["image"].numpy()), np.min(sample["image"].numpy()))
    #     # print(np.max(sample["depth"].numpy()), np.min(sample["depth"].numpy()))
    #     break 
