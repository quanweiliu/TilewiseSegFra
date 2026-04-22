import glob
import os
import scipy.io as sio
import numpy as np
import cv2
from PIL import Image
from scipy.io import loadmat
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import argparse
import torch
import albumentations as albu
from torchvision.transforms import (Pad, ColorJitter, Resize, FiveCrop, RandomCrop,
                                    RandomHorizontalFlip, RandomRotation, RandomVerticalFlip)
import random

SEED = 42


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

Water = np.array([255, 255, 0]) # label 0
oil = np.array([0, 0, 255]) # label 1
Boundary = np.array([0, 0, 0]) # label 2
num_classes = 2


# split huge RS image to small patches
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", default=r"C:\Users\jc962911\Project\datasets\MMF\OSD_H\trainCopy")
    parser.add_argument("--sar-dir", default=r"C:\Users\jc962911\Project\datasets\MMF\OSD_H\trainSAR")
    parser.add_argument("--mask-dir", default=r"C:\Users\jc962911\Project\datasets\MMF\OSD_H\trainCopy")
    parser.add_argument("--output-img-dir", default=r"C:\Users\jc962911\Project\datasets\MMF\OSD_H\train\images_128")
    parser.add_argument("--output-sar-dir", default=r"C:\Users\jc962911\Project\datasets\MMF\OSD_H\train\sar_128")
    parser.add_argument("--output-mask-dir", default=r"C:\Users\jc962911\Project\datasets\MMF\OSD_H\train\masks_128")

    parser.add_argument("--gt", action='store_true')
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--val-scale", type=float, default=1.0)
    parser.add_argument("--split-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    return parser.parse_args()


# 通过 pad 的方式保证数据够用 ！
def get_img_mask_padded(image, sar, mask, patch_size):
    img, sar, mask = np.array(image), np.array(sar), np.array(mask)
    # print("img", img.shape, "mask", mask.shape)
    oh, ow = img.shape[0], img.shape[1]
    rh, rw = oh % patch_size, ow % patch_size
    width_pad = 0 if rw == 0 else patch_size - rw
    height_pad = 0 if rh == 0 else patch_size - rh

    h, w = oh + height_pad, ow + width_pad

    pad_img = albu.PadIfNeeded(min_height=h, min_width=w, position='bottom_right', 
                                border_mode=cv2.BORDER_CONSTANT, value=0)(image=img)
    pad_sar = albu.PadIfNeeded(min_height=h, min_width=w, position='bottom_right', 
                                border_mode=cv2.BORDER_CONSTANT, value=0)(image=sar)
    pad_mask = albu.PadIfNeeded(min_height=h, min_width=w, position='bottom_right', 
                                border_mode=cv2.BORDER_CONSTANT, value=2)(image=mask)
    # print(pad_img.keys(), pad_sar.keys(), pad_mask.keys())

    img_pad, sar_pad, mask_pad = pad_img['image'], pad_sar['image'], pad_mask['image']
    # print("img_pad", img_pad.shape, "mask_pad", mask_pad.shape)
    # img_pad = cv2.cvtColor(np.array(img_pad), cv2.COLOR_RGB2BGR)
    # mask_pad = cv2.cvtColor(np.array(mask_pad), cv2.COLOR_RGB2BGR)
    return img_pad, sar_pad, mask_pad


# pv2rgb将一个二维掩码（mask）转换为三通道的RGB图像
def pv2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [0, 0, 255]
    return mask_rgb


# 将图像中所有颜色为 [0, 255, 255]（黄色）的像素替换为 [0, 204, 255]（淡蓝色）
# 真的是将 车 的颜色替换！！！
def car_color_replace(mask):
    mask = cv2.cvtColor(np.array(mask.clone()), cv2.COLOR_RGB2BGR)
    mask[np.all(mask == [0, 255, 255], axis=-1)] = [0, 204, 255]

    return mask


# pv2rgb 的相反过程
def rgb_to_2D_label(_label):
    _label = _label.transpose(2, 0, 1)
    label_seg = np.zeros(_label.shape[1:], dtype=np.uint8)
    label_seg[np.all(_label.transpose([1, 2, 0]) == Water, axis=-1)] = 0
    label_seg[np.all(_label.transpose([1, 2, 0]) == oil, axis=-1)] = 1
    label_seg[np.all(_label.transpose([1, 2, 0]) == Boundary, axis=-1)] = 2
    if label_seg[np.all(_label.transpose([1, 2, 0]) == Boundary, axis=-1)] == 2:
        print('Boundary')
    return label_seg


# image_augment 则主要关注翻转和缩放操作，增强策略相对简单，但会根据训练或验证模式使用不同的增强方法。
def image_augment(image, sar, mask, patch_size, mode='train', val_scale=1.0):
    image_list = []
    sar_list = []
    mask_list = []
    # image_width, image_height = image.size[1], image.size[0]
    # mask_width, mask_height = mask.size[1], mask.size[0]
    image_width, image_height = image.shape[1], image.shape[0]
    sar_width, sar_height = sar.shape[1], sar.shape[0]
    mask_width, mask_height = mask.shape[1], mask.shape[0]
    
    assert image_height == mask_height and image_width == mask_width and sar_width == sar_width
    
    if mode == 'train':
        # h_vlip = RandomHorizontalFlip(p=1.0)
        # v_vlip = RandomVerticalFlip(p=1.0)
        # image_h_vlip, mask_h_vlip = h_vlip(image.clone()), h_vlip(mask.clone())
        # image_v_vlip, mask_v_vlip = v_vlip(image.clone()), v_vlip(mask.clone())

        # image_list_train = [image, image_h_vlip, image_v_vlip]
        # mask_list_train = [mask, mask_h_vlip, mask_v_vlip]

        image, sar, mask = get_img_mask_padded(image.clone(), sar.clone(), mask.clone(), patch_size)

        # for i in range(len(image_list_train)):
        #     image_tmp, mask_tmp = get_img_mask_padded(image_list_train[i], \
        #                                               mask_list_train[i], \
        #                                               patch_size, mode)
    #         mask_tmp = rgb_to_2D_label(mask_tmp.clone())
    #         image_list.append(image_tmp)
    #         mask_list.append(mask_tmp)
        image_list.append(image)
        sar_list.append(sar)
        mask_list.append(mask)
    else:
        # image <class 'PIL.Image.Image'> mask <class 'PIL.Image.Image'>
        # PIL.Image.Image????? 必须搞成这样吗？？
        # rescale = Resize(size=(int(image_width * val_scale), \
        #                        int(image_height * val_scale)))
        # image = rescale(image.clone())
        # print("rescale image", image.shape)

        # print("rescale mask 1", mask.shape)
        # mask = rescale(mask.clone())
        # print("rescale mask 2", mask.shape)

        image, sar, mask = get_img_mask_padded(image.clone(),sar.clone(), mask.clone(), patch_size)
        # mask = rgb_to_2D_label(mask.clone())
        image_list.append(image)
        sar_list.append(sar)
        mask_list.append(mask)
    return image_list, sar_list, mask_list


def randomsizedcrop(image, mask):
    # assert image.shape[:2] == mask.shape
    h, w = image.shape[0], image.shape[1]
    crop = albu.RandomSizedCrop(min_max_height=(int(3*h//8), int(h//2)), \
                                width=h, height=w)(image=image.clone(), mask=mask.clone())
    img_crop, mask_crop = crop['image'], crop['mask']
    return img_crop, mask_crop


# car_aug 涵盖了更多种类的增强操作，包括缩放、随机裁剪、翻转和旋转，增强的种类更加丰富。
# 这是 针对某几个比较难分的类 的单独数据增强！！！
def car_aug(image, mask):
    assert image.shape[:2] == mask.shape
    v_flip = albu.VerticalFlip(p=1.0)(image=image.clone(), mask=mask.clone())
    h_flip = albu.HorizontalFlip(p=1.0)(image=image.clone(), mask=mask.clone())
    rotate_90 = albu.RandomRotate90(p=1.0)(image=image.clone(), mask=mask.clone())
    # blur = albu.GaussianBlur(p=1.0)(image=image.copy())
    image_vflip, mask_vflip = v_flip['image'], v_flip['mask']
    image_hflip, mask_hflip = h_flip['image'], h_flip['mask']
    image_rotate, mask_rotate = rotate_90['image'], rotate_90['mask']
    # blur_image = blur['image']
    image_list = [image, image_vflip, image_hflip, image_rotate]
    mask_list = [mask, mask_vflip, mask_hflip, mask_rotate]

    return image_list, mask_list


def dwh_format(inp):

    (img_path, sar_path, mask_path, imgs_output_dir, sar_output_dir, \
     masks_output_dir, gt, mode, val_scale, split_size, stride) = inp
    
    img_filename = os.path.splitext(os.path.basename(img_path))[0]
    sar_filename = os.path.splitext(os.path.basename(sar_path))[0]
    mask_filename = os.path.splitext(os.path.basename(mask_path))[0]

    img = loadmat(img_path)['img']
    img = torch.from_numpy(img)
    # print("img  ***", img.shape)               # 1260, 523, 224

    sar = cv2.imread(sar_path)
    sar = torch.from_numpy(sar)
    # print("sar  ***", sar.shape)               # 1260, 523, 224

    mask = loadmat(img_path)['map']
    mask = torch.from_numpy(mask).unsqueeze(-1).repeat(1, 1, 3)
    # print("mask  ***", mask.shape)             # 1260, 523

    # 这个生成的 mask 不是 rgb 格式的。看不了。利用这个函数，生成 RGB 格式的可以看的图像
    # 但是生成的 mask 没有裁剪，这地方虽然没有剪裁，但是后面的裁剪的部分裁剪了。
    if gt:
        mask_ = car_color_replace(mask)
        out_origin_mask_path = os.path.join(masks_output_dir + '/origin/', "{}.tif".format(mask_filename))
        cv2.imwrite(out_origin_mask_path, mask_)
    # print(img_path)
    # print("img", img.size, "mask", mask.size)
    # img and mask shape: WxHxC
    image_list, sar_list, mask_list = image_augment(image=img.clone(), \
                                                    sar=sar.clone(), \
                                                    mask=mask.clone(), \
                                                    patch_size=split_size, \
                                                    mode=mode, \
                                                    val_scale=val_scale)
    assert img_filename == mask_filename and len(image_list) == len(sar_list) and len(image_list) == len(mask_list)
    
    
    for m in range(len(image_list)):
        k = 0
        img = image_list[m]
        sar = sar_list[m]
        mask = mask_list[m]
        # [224, 1260, 523] / [1260, 523, 3]
        # print("img", img.shape, "mask", mask.shape) 
        assert img.shape[0] == mask.shape[0] and img.shape[1] == mask.shape[1]

        if gt:
            mask = pv2rgb(mask)

        # 裁剪图像的过程
        for y in range(0, img.shape[0], stride):
            for x in range(0, img.shape[1], stride):
                img_tile = img[y: y + split_size, x: x + split_size, :]
                sar_tile = sar[y: y + split_size, x: x + split_size, :]
                mask_tile = mask[y: y + split_size, x: x + split_size:, :]
                # print("img_tile", img_tile.shape, "mask_tile", mask_tile.shape)

                # 还有不成立的情况？
                if img_tile.shape[0] == split_size and img_tile.shape[1] == split_size \
                        and mask_tile.shape[0] == split_size and mask_tile.shape[1] == split_size:
                    
                    # pass
                    
                    # image_crop, mask_crop = randomsizedcrop(img_tile, mask_tile)
                    # bins = np.array(range(num_classes + 1))

                    # 计算的是 distrubution，有啥用？
                    # 这段代码通过计算像素分类的分布，决定是否对图像进行数据增强，
                    # 并将增强后的图像和掩码或原始裁剪图像和掩码保存到指定路径，
                    # 以此来丰富训练数据集，提升模型的泛化能力。
                    # 检查类别 4（假设是车类）的像素占比是否大于 0.1（即 10%），并且当前模式是否是训练模式。
                    # class_pixel_counts, _ = np.histogram(mask_crop, bins=bins)
                    # cf = class_pixel_counts / (mask_crop.shape[0] * mask_crop.shape[1])
                    
                    # if cf[1] > 0.1 and mode == 'train':
                    #     car_imgs, car_masks = car_aug(image_crop, mask_crop)
                    #     for i in range(len(car_imgs)):
                    #         out_img_path = os.path.join(imgs_output_dir,
                    #                                     "{}_{}_{}_{}.tif".format(img_filename, m, k, i))
                    #         cv2.imwrite(out_img_path, car_imgs[i])

                    #         out_mask_path = os.path.join(masks_output_dir,
                    #                                      "{}_{}_{}_{}.png".format(mask_filename, m, k, i))
                    #         cv2.imwrite(out_mask_path, car_masks[i])
                    # else:
                        # out_img_path = os.path.join(imgs_output_dir, "{}_{}_{}.tif".format(img_filename, m, k))
                        # cv2.imwrite(out_img_path, img_tile)


                    out_img_path = os.path.join(imgs_output_dir, "{}_{}_{}.mat".format(img_filename, m, k))
                    sio.savemat(out_img_path, {"img": img_tile}, do_compression=True)

                    out_sar_path = os.path.join(sar_output_dir, "{}_{}_{}.mat".format(sar_filename, m, k))
                    sio.savemat(out_sar_path, {"sar": sar_tile}, do_compression=True)
                    # out_sar_path = os.path.join(sar_output_dir, "{}_{}_{}.png".format(sar_filename, m, k))
                    # cv2.imwrite(out_sar_path, sar_tile)

                    out_mask_path = os.path.join(masks_output_dir, "{}_{}_{}.mat".format(mask_filename, m, k))
                    sio.savemat(out_mask_path, {"map": mask_tile}, do_compression=True)

                k += 1


if __name__ == "__main__":
    seed_everything(SEED)
    args = parse_args()
    imgs_dir = args.img_dir
    sar_dir = args.sar_dir
    masks_dir = args.mask_dir

    imgs_output_dir = args.output_img_dir
    sar_output_dir = args.output_sar_dir
    masks_output_dir = args.output_mask_dir

    gt = args.gt
    mode = args.mode
    val_scale = args.val_scale
    split_size = args.split_size
    stride = args.stride

    img_paths = glob.glob(os.path.join(imgs_dir, "*.mat"))
    sar_paths = glob.glob(os.path.join(sar_dir, "*.png"))
    mask_paths = glob.glob(os.path.join(masks_dir, "*.mat"))
    # print("mask_paths_raw", mask_paths_raw)

    img_paths.sort()
    sar_paths.sort()
    mask_paths.sort()

    if not os.path.exists(imgs_output_dir):
        os.makedirs(imgs_output_dir)
    if not os.path.exists(sar_output_dir):
        os.makedirs(sar_output_dir)
    if not os.path.exists(masks_output_dir):
        os.makedirs(masks_output_dir)
    if gt:
        os.makedirs(masks_output_dir + '/origin')

    inp = [(img_path, sar_path, mask_path, imgs_output_dir, sar_output_dir, \
            masks_output_dir, gt, mode, val_scale, split_size, stride)
           for img_path, sar_path, mask_path in zip(img_paths, sar_paths, mask_paths)]
    # print("inp", inp)

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(dwh_format, inp)
    t1 = time.time()
    split_time = t1 - t0
    print('images spliting spends: {} s'.format(split_time))



# python tools/dwh_patch_split.py --img-dir "C:\Users\jc962911\Project\datasets\MMF\OSD_H\valCopy" --mask-dir "C:\Users\jc962911\Project\datasets\MMF\OSD_H\valCopy" --output-img-dir "C:\Users\jc962911\Project\datasets\MMF\OSD_H\val\images_128" --output-mask-dir "C:\Users\jc962911\Project\datasets\MMF\OSD_H\val\masks_128" --mode "val" --split-size 128 --stride 64
# python tools/dwh_patch_split.py --img-dir "C:\Users\jc962911\Project\datasets\MMF\OSD_H\trainCopy" --mask-dir "C:\Users\jc962911\Project\datasets\MMF\OSD_H\trainCopy" --output-img-dir "C:\Users\jc962911\Project\datasets\MMF\OSD_H\train\images_128" --output-mask-dir "C:\Users\jc962911\Project\datasets\MMF\OSD_H\train\masks_128" --mode "train" --split-size 128 --stride 64
# python tools/dwh_patch_split.py --img-dir "C:\Users\jc962911\Project\datasets\MMF\OSD_H\testCopy" --mask-dir "C:\Users\jc962911\Project\datasets\MMF\OSD_H\testCopy" --output-img-dir "C:\Users\jc962911\Project\datasets\MMF\OSD_H\test\images_224" --output-mask-dir "C:\Users\jc962911\Project\datasets\MMF\OSD_H\test\masks_224" --mode "val" --split-size 128 --stride 64



# python tools/dwh_patch_split.py \
# --img-dir "C:\Users\jc962911\Project\datasets\MMF\OSD_H\valCopy" \
# --sar-dir "C:\Users\jc962911\Project\datasets\MMF\OSD_H\valSAR" \
# --mask-dir "C:\Users\jc962911\Project\datasets\MMF\OSD_H\valCopy" \
# --output-img-dir "C:\Users\jc962911\Project\datasets\MMF\OSD_H\val\images_128" \
# --output-sar-dir "C:\Users\jc962911\Project\datasets\MMF\OSD_H\val\sar_128" \
# --output-mask-dir "C:\Users\jc962911\Project\datasets\MMF\OSD_H\val\masks_128" \
# --mode "val" --split-size 128 --stride 64


# python tools/dwh_patch_split.py --img-dir "C:\Users\jc962911\Project\datasets\MMF\OSD_H\trainCopy" --sar-dir "C:\Users\jc962911\Project\datasets\MMF\OSD_H\trainSAR" --mask-dir "C:\Users\jc962911\Project\datasets\MMF\OSD_H\trainCopy" --output-img-dir "C:\Users\jc962911\Project\datasets\MMF\OSD_H\train\sar_128" --output-sar-dir "C:\Users\jc962911\Project\datasets\MMF\OSD_H\train\sar_128" --output-mask-dir "C:\Users\jc962911\Project\datasets\MMF\OSD_H\train\masks_128" --mode "train" --split-size 128 --stride 64
# python tools/dwh_patch_split.py --img-dir "C:\Users\jc962911\Project\datasets\MMF\OSD_H\testCopy" --sar-dir "C:\Users\jc962911\Project\datasets\MMF\OSD_H\testSAR" --mask-dir "C:\Users\jc962911\Project\datasets\MMF\OSD_H\testCopy" --output-img-dir "C:\Users\jc962911\Project\datasets\MMF\OSD_H\test\images_224" --output-sar-dir "C:\Users\jc962911\Project\datasets\MMF\OSD_H\test\sar_128" --output-mask-dir "C:\Users\jc962911\Project\datasets\MMF\OSD_H\test\masks_128" --mode "val" --split-size 128 --stride 64



