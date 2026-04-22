import os
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, [0]))
print('using GPU %s' % ','.join(map(str, [0])))

import cv2
import csv
import yaml
import time
import tifffile
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import scipy.io as scio
from collections import namedtuple
from matplotlib.patches import Patch

import torch
from torch.utils import data
from ptsemseg.logger import Logger
from dataLoader.OSTD_loader import OSTD_loader
from dataLoader.OSTD_loader2 import OSTD_loader2
from dataLoader.ISPRS_loader import ISPRS_loader
from dataLoader.RGBD_loader import RGBD_loader
# from ptsemseg.loss import dice_bce_gScore
from ptsemseg.models import get_model
from schedulers.metrics import runningScore, averageMeter
from tools.utils import plot_training_results

###################################
# FUNCTION TO VISUALIZE MODEL PREDICTIONS
###################################
def train_id_to_color(classes):
    # a tuple with name, train_id, color, more pythonic and more easily readable. 
    Label = namedtuple( "Label", [ "name", "train_id", "color"])
    # print(len(classes))
    if len(classes) == 2:
        drivables = [ 
            Label(classes[0], 0, (255, 255, 0)), 
            Label(classes[1], 1, (0, 0, 255)),
        ]
    elif len(classes) == 3:
        drivables = [ 
            Label(classes[0], 0, (255, 255, 0)), 
            Label(classes[1], 1, (0, 0, 255)),
            Label(classes[2], 2, (255, 0, 0))
        ]
    elif len(classes) == 6:
        drivables = [ 
            Label(classes[0], 0, (255, 255, 255)), 
            Label(classes[1], 1, (0, 0, 255)), 
            Label(classes[2], 2, (0, 255, 255)), 
            Label(classes[3], 3, (0, 255, 0)), 
            Label(classes[4], 4, (255, 255, 0)), 
            Label(classes[5], 5, (255, 0, 0))
        ]
    elif len(classes) == 10:
        drivables = [ 
            Label(classes[0], 0, (0, 0, 0)), 
            Label(classes[1], 1, (255, 0, 0)), 
            Label(classes[2], 2, (200, 90, 90)), 
            Label(classes[3], 3, (130, 130, 0)), 
            Label(classes[4], 4, (150, 150, 150)), 
            Label(classes[5], 5, (0, 255, 255)),
            Label(classes[6], 6, (0, 0, 255)), 
            Label(classes[7], 7, (255, 0, 255)), 
            Label(classes[8], 8, (250, 250, 0)), 
            Label(classes[9], 9, (0, 255, 0)) 
        ]
    elif len(classes) == 41:
        drivables = [
            Label(classes[0], 0, (0, 0, 0)), # 0=background
            Label(classes[1], 1, (148, 65, 137)),
            Label(classes[2], 2, (255, 116, 69)),
            Label(classes[3], 3, (86, 156, 137)),
            Label(classes[4], 4, (202, 179, 158)),
            Label(classes[5], 5, (155, 99, 235)),
            Label(classes[6], 6, (161, 107, 108)),
            Label(classes[7], 7, (133, 160, 103)),
            Label(classes[8], 8, (76, 152, 126)),
            Label(classes[9], 9, (84, 62, 35)),
            Label(classes[10], 10, (44, 80, 130)),
            Label(classes[11], 11, (31, 184, 157)),
            Label(classes[12], 12, (101, 144, 77)),
            Label(classes[13], 13, (23, 197, 62)),
            Label(classes[14], 14, (141, 168, 145)),
            Label(classes[15], 15, (142, 151, 136)),
            Label(classes[16], 16, (115, 201, 77)),
            Label(classes[17], 17, (100, 216, 255)),
            Label(classes[18], 18, (57, 156, 36)),
            Label(classes[19], 19, (88, 108, 129)),
            Label(classes[20], 20, (105, 129, 112)),
            Label(classes[21], 21, (42, 137, 126)),
            Label(classes[22], 22, (155, 108, 249)),
            Label(classes[23], 23, (166, 148, 143)),
            Label(classes[24], 24, (81, 91, 87)),
            Label(classes[25], 25, (100, 124, 51)),
            Label(classes[26], 26, (73, 131, 121)),
            Label(classes[27], 27, (157, 210, 220)),
            Label(classes[28], 28, (134, 181, 60)),
            Label(classes[29], 29, (221, 223, 147)),
            Label(classes[30], 30, (123, 108, 131)),
            Label(classes[31], 31, (161, 66, 179)),
            Label(classes[32], 32, (163, 221, 160)),
            Label(classes[33], 33, (31, 146, 98)),
            Label(classes[34], 34, (99, 121, 30)),
            Label(classes[35], 35, (49, 89, 240)),
            Label(classes[36], 36, (116, 108, 9)),
            Label(classes[37], 37, (161, 176, 169)),
            Label(classes[38], 38, (80, 29, 135)),
            Label(classes[39], 39, (177, 105, 197)),
            Label(classes[40], 40, (139, 110, 246))
            ]
    else:
        raise ValueError("Unsupported number of classes: {}".format(len(classes)))
    # print("drivables", drivables)
    id_to_color = [c.color for c in drivables if (c.train_id != -1 and c.train_id != 255)]
    id_to_color = np.array(id_to_color)
    # print("id_to_color", id_to_color)
    
    legend_elements = []
    for i, c in enumerate(classes):
        # A **Patch** is a 2D artist with a face color and an edge color.If any of edgecolor, facecolor, linewidth, 
        # or antialiased are None, they default to their rc params setting.
        # 个人理解就是这个类是其他2D 图形类的支持类，用以控制各个图形类的公有属性。
        legend_elements.append(Patch(facecolor = id_to_color[i]/255, label=c))
        
    return id_to_color, legend_elements


def printMetrics(submit_path, mask_path, running_metrics, log):
    img_list = os.listdir(submit_path)
    # print("img_list: ", img_list)

    for img in img_list:
        pred = tifffile.imread(os.path.join(submit_path, img))
        # mask = tifffile.imread(os.path.join(mask_path, img))
        # pred = scio.loadmat(os.path.join(mask_path, img[:-4]))['pred']
        mask = scio.loadmat(os.path.join(mask_path, img[:-4]))['map']
        # print("mask", np.unique(mask), "pred", np.unique(pred))
        running_metrics.update(mask, pred)

    score, class_iou = running_metrics.get_scores()
    log.write('************ test_result **********\n')
    log.write('{}: '.format(args.TTA) + '\n')

    for k, v in score.items():
        # print(k, v)
        log.write('{}: {}'.format(k, round(v * 100, 2)) + '\n')
    
    log.flush()
    log.write('Finish!\n')
    log.close()

    running_metrics.reset()

def sort_key(filename, args):
    # 将文件名前缀（数字部分）提取出来并转换为整数
    # return int(filename.split('.')[0][20:])
    if args.data_name == 'OSTD':
        name = filename.split('.')[0][5:]
    elif args.data_name == 'Vaihingen':
        name = filename.split('.')[0][20:]
    return int(name)

def test(args):
    # Setup submits
    if args.out_path == '':
        if args.TTA:
            print("############ we use the test time augmentation ############")
            out_path = os.path.join(os.path.split(args.model_path)[0], 'test_tta')
        else:
            print("############ we don't use the test time augmentation ############")
            out_path = os.path.join(os.path.split(args.model_path)[0], 'test')
    else:
        out_path = args.out_path

    if not os.path.exists(out_path):
        os.mkdir(out_path)


    # Setup Dataloader
    if args.data_name == "OSTD":
        print("############ we use the OSTD dataset ############")
        imgname_list = sorted(os.listdir(os.path.join(args.imgs_path, 'test', 'image128')))
        classes = ['Oil', 'Water'] # 其中 Clutter # 是指 background
        test_dataset = OSTD_loader(args.imgs_path, args.split, args.img_size, is_augmentation=False)
        running_metrics_test = runningScore(args.classes+1)

    elif args.data_name == "Vaihingen" or args.data_name == "Potsdam":
        print("############ we use the ISPRS dataset ############")
        imgname_list = sorted(os.listdir(os.path.join(args.imgs_path, 'test', 'images256')))
        classes = ['ImpSurf', 'Building', 'Car', 'Tree', 'LowVeg', 'Clutter'] # 其中 Clutter # 是指 background
        test_dataset = ISPRS_loader(args.imgs_path, args.split, \
                                    args.img_size, args.classes, \
                                    args.data_name, args.normalization, \
                                    is_augmentation=False)
        running_metrics_test = runningScore(args.classes)

    elif args.data_name == "NYUv2":
        txt_path = os.path.join(args.imgs_path, args.split + '.txt')
        with open(os.path.join(txt_path), "r") as f:
            imgname_list = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        classes = ['Background', 'Wall', 'Floor', 'Cabinet', 'Bed', 'Chair', 'Sofa', 
                   'Table', 'Door', 'Window',  'Bookshelf', 'Picture', 'Counter', 
                   'Blind', 'Desk', 'Shelf', 'Curtain', 'Dresser', 'Pillow', 'Mirror', 
                   'Mat', 'Cloths', 'Ceiling', 'Books', 'Refridg', 'TV', 'Paper', 
                   'Towel', 'Shower', 'Box', 'Board', 'Person', 'Stand', 'Toilet', 
                   'Sink', 'Lamp', 'Bathtub', 'Bag', 'Othstr', 'Othfurn', 'Otherprop'] 
        test_dataset = RGBD_loader(args.imgs_path, args.split, args.img_size, args.classes, 
                               args.data_name, args.normalization, is_augmentation=False)
        running_metrics_test = runningScore(args.classes)

    testloader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)


    id_to_color, legend_elements = train_id_to_color(classes)
    model = get_model({"arch":args.model}, args.bands1, args.bands2, args.classes, args.classification, args.img_size).to(args.device)

    # state = convert_state_dict(torch.load(args.model_path)["model_state"])    # multi-gpus
    checkpoint = torch.load(args.model_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    epoch = checkpoint["epoch"]
    print("successfully load model from {} at {}".format(args.model_path, epoch))
    
    # plot_training_results
    # savefig_path = os.path.split(args.model_path)[0]
    # results_train = pd.DataFrame(checkpoint['results_train'])
    # results_val = pd.DataFrame(checkpoint['results_val'])
    
    # plot_training_results(results_train, results_val, args.model, savefig_path)

    model.to(args.device)

    test_log = Logger(os.path.join(os.path.split(args.model_path)[0], 'test_result.log'))

    ########################### test ####################################
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        print("TTA is running...", args.TTA)
        for ind, (gaofen, lidar, mask) in tqdm(enumerate(testloader)):
            img_id = imgname_list[ind]
            gaofen = gaofen.to(args.device)  # (B, C, H, W)
            lidar = lidar.to(args.device)

            if args.TTA:
                # 原图 + 旋转90°
                gaofen_90 = torch.rot90(gaofen, k=1, dims=[2, 3])
                lidar_90 = torch.rot90(lidar, k=1, dims=[2, 3])
                # print("gaofen_90 shape: ", gaofen_90.shape) # B, C, H, W

                # 拼接 batch：原图 和 旋转图
                gaofen_batch = torch.cat([gaofen, gaofen_90], dim=0)
                lidar_batch = torch.cat([lidar, lidar_90], dim=0)
                # print("gaofen_batch shape: ", gaofen_batch.shape) # 2B, C, H, W

                # 第二组：左右翻转
                gaofen_flip = torch.flip(gaofen_batch, dims=[3])  # 翻转 W
                lidar_flip = torch.flip(lidar_batch, dims=[3])

                # 模型推理
                pred_a = model(gaofen_batch, lidar_batch)         # 原图 + 旋转图
                pred_b = model(gaofen_flip, lidar_flip)           # 翻转后预测
                pred_a = pred_a[0]                                 # 取出预测结果
                pred_b = pred_b[0]
                pred_b = torch.flip(pred_b, dims=[3])             # 翻转回来

                # 融合两个方向的预测（上面只是把镜像图复原了，旋转图还没有复原）
                pred = (pred_a + pred_b) / 2                      # shape: (2B, C, H, W)

                # 拆分原图和旋转图的结果（复原旋转图）
                B = gaofen.shape[0]
                pred1 = pred[:B]                                  # 原图预测
                pred2 = torch.rot90(pred[B:], k=-1, dims=[2, 3])  # 旋转回原角度
                pred = (pred1 + pred2) / 2                        # 最终融合 (B, C, H, W)

                # 获取预测类别（最终结果）
                pred = pred.argmax(dim=1).cpu().numpy().astype(np.uint8)  # (B, H, W)
                running_metrics_test.update(mask.numpy(), pred)

            else:
                outputs = model(gaofen, lidar)
                if args.classification == "Multi":
                    outputs = outputs[0]
                    pred = outputs.argmax(dim=1).cpu().numpy().astype(np.uint8)  # [B, H, W]

                elif args.classification == "Binary":
                    outputs = outputs[0]
                    outputs[outputs > args.threshold] = 1
                    outputs[outputs <= args.threshold] = 0
                    pred = outputs.data.cpu().numpy().astype(np.uint8)
                running_metrics_test.update(mask.numpy(), pred)

        ############################### save pred image ###############################
            if args.save_img:
                pred = pred.reshape(args.img_size, args.img_size)
                cv2.imwrite(os.path.join(out_path, str(img_id) + '.png'), id_to_color[pred])
                # cv2.imwrite(os.path.join(out_path, str(img_id) + '.png'), pred)
                # tifffile.imwrite(os.path.join(out_path, str(img_id) + '.tif'), pred.astype(np.uint8))
                # if ind == 10:
                #     break

        # print and save metrics result
        score, class_iou = running_metrics_test.get_scores(ignore_index=args.ignore_index)
        test_log.write('************test_result**********\n')
        test_log.write('TTA: {} Epoch: {}: '.format(args.TTA, epoch) + '\n')

        for k, v in score.items():
            test_log.write('{}: {}'.format(k, round(v * 100, 2)) + '\n')
        
        t1 = time.time()
        img_write_time = t1 - t0
        test_log.write('{}    \t: {}'.format("time", round(img_write_time, 2)) + '\n')
        
        test_log.flush()
        test_log.write('Finish!\n')
        test_log.close()
        
        running_metrics_test.reset()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument('--model',
                        choices=['ACNet', \
                                 'CANet50', \
                                 'CMANet', \
                                 'CMGFNet18', \
                                 'CMGFNet34'],
                        default='CMGFNet34', help="the model architecture that should be trained")    
    parser.add_argument("--device", nargs = "?", type = str, default = "cuda:0", help="CPU or GPU")
    parser.add_argument("--split", type = str, default = "test", help="Dataset to use ['train, val, test']")
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold for binary classification')
    parser.add_argument('--n_workers', type=int, default=4, help='number of workers for validation data')
    parser.add_argument("--TTA", nargs="?", type=bool, default=False, help="default use TTA",) # default=False / True
    parser.add_argument("--out_path", nargs = "?", type = str, default = '', help="Path of the output segmap")
    parser.add_argument("--save_img", type=bool, default=False, help="whether save pred image or not")

    parser.add_argument("--file_path", nargs = "?", type = str, \
                        # default = os.path.join("/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/run_Pot_st/0929-2319-ACNet"),
                        # default = os.path.join("/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/run_Pot_st/0928-1759-CANet50"),
                        # default = os.path.join("/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/run_Vai_st/0928-0844-CMANet"),
                        # default = os.path.join("/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/run_Pot_st/1001-0103-CMGFNet18"),
                        default = os.path.join("/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/run_Vai_st/1015-1045-CMGFNet34"),
                        help="Path to the saved model")
    args = parser.parse_args(args=[])

    with open(os.path.join(args.file_path, args.model + '.yml')) as fp:
        cfg = yaml.safe_load(fp)

    args.model_path = os.path.join(args.file_path, "best.pt")
    args.data_name = cfg['data']['name']
    args.imgs_path = cfg['data']['path']
    args.bands1 = cfg['data']['bands1']
    args.bands2 = cfg['data']['bands2']
    args.classes = cfg['data']['classes']
    args.normalization = cfg['data']['normalization']
    args.classification = cfg['data']['classification']
    args.img_size = cfg['data']['img_size']
    args.split = cfg['data']['test_split']
    args.batch_size = cfg['training']['test_batch_size']
    args.ignore_index = cfg['data']['ignore_index']
    args.threshold = cfg['threshold']
    print("args", args.img_size, args.classes, args.ignore_index, args.threshold, args.normalization)
    test(args)

