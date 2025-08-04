import os
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, [1]))
print('using GPU %s' % ','.join(map(str, [1])))

import cv2
import yaml
import time
import tifffile
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.io as scio
from matplotlib.patches import Patch
from collections import namedtuple
from tools.utils import plot_training_results

import torch
from torch.utils import data
from ptsemseg.logger import Logger
from dataLoader.dataloader_ISPRS import ISPRS_loader
# from ptsemseg.loss import dice_bce_gScore
from ptsemseg.models import get_model

from schedulers.metrics import runningScore, averageMeter

###################################
# FUNCTION TO VISUALIZE MODEL PREDICTIONS
###################################
def train_id_to_color(classes):
    # a tuple with name, train_id, color, more pythonic and more easily readable. 
    Label = namedtuple( "Label", [ "name", "train_id", "color"])
    # print(len(classes))
    if len(classes) == 2:
        # print("into here")
        drivables = [ 
            Label(classes[0], 0, (255, 255, 0)), 
            Label(classes[1], 1, (0, 0, 255)),
            # Label(classes[2], 2, (255, 0, 0))
        ]
    elif len(classes) == 3:
        # print("into here")
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
    else:
        return
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

def sort_key(filename):
    # 将文件名前缀（数字部分）提取出来并转换为整数
    return int(filename.split('.')[0][20:])


def test(args):

    # Setup image
    images_dir = "/home/icclab/Documents/lqw/DatasetMMF/Vaihingen/test/masks256"
    imgname_list = sorted(os.listdir(images_dir), key=sort_key)

    classes = ['ImpSurf', 'Building', 'Car', 'Tree', 'LowVeg', 'Clutter'] # 其中 Clutter # 是指 background
    id_to_color, legend_elements = train_id_to_color(classes)

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


    test_loader = ISPRS_loader(args.imgs_path, args.split, args.img_size, is_augmentation=False)
    testloader = data.DataLoader(test_loader, batch_size = args.batch_size, shuffle=False, num_workers=args.n_workers)

    model = get_model({"arch":args.model}, len(classes)).to(args.device)

    # state = convert_state_dict(torch.load(args.model_path)["model_state"])    # multi-gpus
    checkpoint = torch.load(args.model_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    epoch = checkpoint["epoch"]

    print("successfully load model from {} at {}".format(args.model_path, epoch))

    # plot_training_results
    savefig_path = os.path.split(args.model_path)[0]
    results_train = pd.DataFrame(checkpoint['results_train'])
    results_val = pd.DataFrame(checkpoint['results_val'])
    
    plot_training_results(results_train, results_val, args.model, savefig_path)

    model.eval()
    model.to(args.device)

    # Setup Metrics
    running_metrics_test = runningScore(len(classes))
    test_log = Logger(os.path.join(os.path.split(args.model_path)[0], 'test_result.log'))

    ########################### test ####################################
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for ind, (gaofen, lidar, mask) in tqdm(enumerate(testloader)):
            img_id = imgname_list[ind]

            if args.TTA:
                gaofen = gaofen.to(args.device)  # (B, C, H, W)
                lidar = lidar.to(args.device)

                # 原图 + 旋转90°
                gaofen_90 = torch.rot90(gaofen, k=1, dims=[2, 3])
                lidar_90 = torch.rot90(lidar, k=1, dims=[2, 3])
                print("gaofen_90 shape: ", gaofen_90.shape, "lidar_90 shape: ", lidar_90.shape )

                # 拼接 batch：原图 和 旋转图
                gaofen_batch = torch.cat([gaofen, gaofen_90], dim=0)
                lidar_batch = torch.cat([lidar, lidar_90], dim=0)
                print("gaofen_batch shape: ", gaofen_batch.shape, "lidar_batch shape: ", lidar_batch.shape )

                # 第二组：左右翻转
                gaofen_flip = torch.flip(gaofen_batch, dims=[3])  # 翻转 W
                lidar_flip = torch.flip(lidar_batch, dims=[3])

                # 模型推理
                pred_a = model(gaofen_batch, lidar_batch)         # 原图 + 旋转图
                pred_b = model(gaofen_flip, lidar_flip)           # 翻转后预测
                pred_b = torch.flip(pred_b, dims=[3])             # 翻转回来

                # 融合两个方向的预测
                pred = (pred_a + pred_b) / 2                      # shape: (2B, C, H, W)

                # 拆分原图和旋转图的结果，再合并
                B = gaofen.shape[0]
                pred1 = pred[:B]                                  # 原图预测
                pred2 = torch.rot90(pred[B:], k=-1, dims=[2, 3])  # 旋转回原角度
                pred = (pred1 + pred2) / 2                        # 最终融合 (B, C, H, W)

                # 获取预测类别
                pred = pred.argmax(dim=1).cpu().numpy().astype(np.uint8)  # (B, H, W)
                running_metrics_test.update(mask.cpu().numpy(), pred)

            else:
                gaofen = gaofen.to(args.device)
                lidar = lidar.to(args.device)

                pred = model(gaofen, lidar)
                pred = pred.argmax(dim=1).cpu().numpy().astype(np.uint8)  # [B, H, W]
                running_metrics_test.update(mask.cpu().numpy(), pred)

        ############################### save pred image ###############################
            if args.save_img:
                pred = pred.reshape(args.img_size, args.img_size)
                cv2.imwrite(os.path.join(out_path, str(img_id) + '.png'), id_to_color[pred])
                # cv2.imwrite(os.path.join(out_path, str(img_id) + '.png'), pred.astype(np.uint8))
                # tifffile.imwrite(os.path.join(out_path, str(img_id) + '.tif'), pred.astype(np.uint8))

        # print and save metrics result
        score, class_iou = running_metrics_test.get_scores()
        test_log.write('************test_result**********\n')
        test_log.write('{}: '.format(args.TTA) + '\n')

        for k, v in score.items():
            test_log.write('{}: {}'.format(k, round(np.nanmean(v) * 100, 2)) + '\n')
        
        t1 = time.time()
        img_write_time = t1 - t0
        test_log.write('{}    \t: {}'.format("time", round(img_write_time, 2)) + '\n')
        
        test_log.flush()
        test_log.write('Finish!\n')
        test_log.close()
        
        running_metrics_test.reset()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument('--model', choices=['DE_CCFNet_18', 'DE_CCFNet_34', 'DE_DCGCN', 'Zhiyang', \
                                            "AsymFormer", "SFAFMA", "MCANet", 'MGFNet2_50', "PCGNet34", \
                                            "SFAFMA50", 'SOLC', 'DE_DCGCN', 'PACSCNet50', 'FAFNet'], \
                        default='DE_CCFNet_34', help="the model architecture that should be trained")
    parser.add_argument("--device", nargs = "?", type = str, default = "cuda:0", help="CPU or GPU")
    parser.add_argument("--split", type = str, default = "test", help="Dataset to use ['train, val, test']")
    parser.add_argument("--imgs_path", nargs = "?", type = str, default = '/home/icclab/Documents/lqw/DatasetMMF/Vaihingen', \
                        help="Path of the input image")
    parser.add_argument('--img_size', type=int, default=256, help='size of the image patches the model should be trained on')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size for training data')
    parser.add_argument('--n_workers', type=int, default=4, help='number of workers for validation data')
    parser.add_argument("--TTA", nargs="?", type=bool, default=True, help="default use TTA",) # default=False / True
    parser.add_argument("--out_path", nargs = "?", type = str, default = '', help="Path of the output segmap")


    # parser.add_argument("--config", nargs = "?", type = str,
                        # default = "/home/leo/Semantic_Segmentation/multiRoadHSI/config/extraction_epoch_DE_CCFNet18.yml",
                        # help="Configuration file to use")
    parser.add_argument("--model_path", nargs = "?", type = str, \
                        default = os.path.join("/home/icclab/Documents/lqw/Multimodal_Segmentation/multiISPRS/run/0708-2253-DE_CCFNet_34", "best.pt"),
                        help="Path to the saved model")
    parser.add_argument("--save_img", type=bool, default=False, help="whether save pred image or not")

    args = parser.parse_args(args=[])

    # if os.path.split(args.model_path)[1] == "":
    #     pass 

    test(args)

