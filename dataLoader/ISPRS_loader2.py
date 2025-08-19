import os
import cv2
import random
import numpy as np
import matplotlib
import matplotlib.colors
import torchvision
from torchvision import transforms
import torch
from torch.utils.data import Dataset


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


# 这个用 txt 选择数据
class ISPRS_loader2(Dataset):
    def __init__(self, transform=None, 
                 data_dir=None, 
                 txt_name='train.txt'):
        self.transform = transform

        root = data_dir
        assert os.path.exists(root), "path '{}' does not exist.".format(root)

        """生成图像文件夹路径与标注(mask)文件夹路径"""
        image_dir = os.path.join(root, 'images256')
        depth_dir = os.path.join(root, 'DSM256')
        mask_dir = os.path.join(root, 'masks256')

        """读取图像列表-txt文件放在根目录"""
        txt_path = os.path.join(root, txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)

        """读取文件名-图像images，深度depths，标签masks
        不同文件具有不同的扩展名"""
        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        self.img_dir_train = [os.path.join(image_dir, x + ".tif") for x in file_names]
        self.depth_dir_train = [os.path.join(depth_dir, x + ".tif") for x in file_names]
        self.label_dir_train = [os.path.join(mask_dir, x + ".tif") for x in file_names]

        assert (len(self.img_dir_train) == len(self.label_dir_train) and len(self.img_dir_train) == len(
            self.depth_dir_train))

    def __len__(self):
        return len(self.img_dir_train)

    def __getitem__(self, idx):

        img_dir = self.img_dir_train
        depth_dir = self.depth_dir_train
        label_dir = self.label_dir_train

        label = cv2.imread(label_dir[idx], flags=cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(depth_dir[idx], flags=cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32)

        image = self._open_image(img_dir[idx], cv2.COLOR_BGR2RGB)

        # image () depth (1024, 1024) label (1024, 1024)
        # print("image", image.shape, "depth", depth.shape, "label", label.shape)

        sample = {'image': image, 'depth': depth, 'label': label}

        if self.transform:
            # print("transform")
            sample = self.transform(sample)

        # sample = {'image': image, 'depth': depth, 'label': label}
        # sample = {'image': image, 'depth': depth, 'label': rgb_to_2D_label(label)}
        sample['label'] = rgb_to_2D_label(label)[:, :, 0].astype(np.int64)

        return sample

    # _open_image() 封装了额外的功能, mode, dtype
    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        # print("filepath", filepath)
        img = np.array(cv2.imread(filepath, mode), dtype=dtype)
        # print("img", img.shape)
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
        # print("img", img.dtype, img.max(), img.min())

        # rgb_to_hsv 对于输入是 [0, 1] 范围的浮点数和 [0, 255] 范围的 uint8 整数，
        # 其输出的 V (明度) 通道范围是不同的。
        img_hsv = matplotlib.colors.rgb_to_hsv(img) 
        # Hue / Saturation / Value
        img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        h_random = np.random.uniform(min(self.h_range), max(self.h_range))
        s_random = np.random.uniform(min(self.s_range), max(self.s_range))
        v_random = np.random.uniform(-min(self.v_range), max(self.v_range))
        img_h = np.clip(img_h * h_random, 0, 1)
        img_s = np.clip(img_s * s_random, 0, 1)
        img_v = np.clip(img_v + v_random, 0, 255)
        img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        img_new = matplotlib.colors.hsv_to_rgb(img_hsv)

        return {'image': img_new, 'depth': sample['depth'], 'label': sample['label']}


class scaleNorm(object):
    def __init__(self, image_w=512, image_h=512):
        self.image_w = image_w
        self.image_h = image_h

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        # image (480, 640, 3) depth (480, 640) label (480, 640)
        # print("image", image.shape, "depth", depth.shape, "label", label.shape)

        # Bi-linear
        image = cv2.resize(image, (self.image_w, self.image_h), cv2.INTER_LINEAR)
        # Nearest-neighbor
        depth = cv2.resize(depth, (self.image_w, self.image_h), cv2.INTER_NEAREST)
        label = cv2.resize(label, (self.image_w, self.image_h), cv2.INTER_NEAREST)

        return {'image': image, 'depth': depth, 'label': label}


class RandomScale(object):
    def __init__(self, scale):
        self.scale_low = min(scale)
        self.scale_high = max(scale)

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        target_scale = random.uniform(self.scale_low, self.scale_high)
        # (H, W, C)
        target_height = int(round(target_scale * image.shape[0]))
        target_width = int(round(target_scale * image.shape[1]))

        # Bi-linear
        image = cv2.resize(image, (target_width, target_height), cv2.INTER_LINEAR)
        # Nearest-neighbor
        depth = cv2.resize(depth, (target_width, target_height), cv2.INTER_NEAREST)
        label = cv2.resize(label, (target_width, target_height), cv2.INTER_NEAREST)

        return {'image': image, 'depth': depth, 'label': label}


class RandomCrop(object):
    def __init__(self, th=512, tw=512):
        self.th = th
        self.tw = tw

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        h = image.shape[0]
        w = image.shape[1]
        i = random.randint(0, h - self.th)
        j = random.randint(0, w - self.tw)

        return {'image': image[i : i + self.th, j : j + self.tw, :],
                'depth': depth[i : i + self.th, j : j + self.tw],
                'label': label[i : i + self.th, j : j + self.tw]}


class RandomFlip(object):
    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            depth = np.fliplr(depth).copy()
            label = np.fliplr(label).copy()

        return {'image': image, 'depth': depth, 'label': label}


# Transforms on torch.*Tensor
class Normalize(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = image / 255
        # depth = depth / 255
        # imagenet
        # image = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
        #                                  std=[0.229, 0.224, 0.225])(image)
        
        # vaihingen
        image = transforms.Normalize(mean=[0.4731, 0.3206, 0.3182], 
                                         std=[0.1970, 0.1306, 0.1276])(image)

        # potsdam
        # image = transforms.Normalize(mean=[0.349, 0.371, 0.347], 
        #                                  std=[0.1196, 0.1164, 0.1197])(image)

        # potsdam_irrg
        # image = transforms.Normalize(mean=[0.3823, 0.3625, 0.3364], 
        #                                  std=[0.1172, 0.1167, 0.1203])(image)  
        # depth = torchvision.transforms.Normalize(mean=[2.8424503515351494],
        #                                          std=[0.9932836506164299])(depth)
        depth = (depth - depth.min()) / (depth.max() - depth.min())  # → [0, 1]
        sample['image'] = image
        sample['depth'] = depth

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        label = label.astype(np.int16)
        # h = label.shape[0]
        # w = label.shape[1]
        # # Generate different label scales
        # label3 = cv2.resize(label, (w // 4, h // 4), cv2.INTER_NEAREST)
        # label4 = cv2.resize(label, (w // 8, h // 8), cv2.INTER_NEAREST)
        # label5 = cv2.resize(label, (w // 16, h // 16), cv2.INTER_NEAREST)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        depth = np.expand_dims(depth, 0).astype(np.float64)

        return {'image': torch.from_numpy(image).float(),
                'depth': torch.from_numpy(depth).float(),
                'label': torch.from_numpy(label).float(),}
                # 'label3': torch.from_numpy(label3).float(),
                # 'label4': torch.from_numpy(label4).float(),
                # 'label5': torch.from_numpy(label5).float()}


if __name__ == '__main__':
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    image_w = 256
    image_h = 256

    data_dir = "/home/icclab/Documents/lqw/DatasetMMF/Vaihingen"


    transformer = transforms.Compose([scaleNorm(image_w, image_h),
                                    RandomScale((1.0, 1.4, 2.0)),
                                    RandomHSV((0.9, 1.1),
                                            (0.9, 1.1),
                                            (25, 25)),
                                    RandomCrop(th=image_h, tw=image_w),
                                    RandomFlip(),
                                    ToTensor(),
                                    Normalize()
                                    ])
    
    transformer = transforms.Compose([scaleNorm(image_w, image_h),
                                    ToTensor(),
                                    Normalize()
                                    ])
    
    train_data = ISPRS_loader2(transform=transformer,
                                data_dir=data_dir)
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True,
                              num_workers=0, pin_memory=False)

    num_train = len(train_data)
    # print(num_train)

    for sample in train_loader:
        print("image", sample["image"].shape, sample["image"].max(), sample["image"].min())   # torch.Size([2, 3, 480, 640])
        print("depth", sample["depth"].shape, sample["depth"].max(), sample["depth"].min())   # torch.Size([2, 1, 480, 640])
        print("label", sample["label"].shape, sample["label"].dtype, np.unique(sample["label"]))   # torch.Size([2, 480, 640])
        break