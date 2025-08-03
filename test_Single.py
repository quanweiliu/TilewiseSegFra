import os
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
from dataLoader.dataloader import Road_loader
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
    
    # for k, v in class_iou.items():
    #     # print(k, v)
    #     log.write('{}: {}'.format(k, v) + '\n')
    
    log.flush()
    log.write('Finish!\n')
    log.close()

    running_metrics.reset()

def sort_key(filename):
    # 将文件名前缀（数字部分）提取出来并转换为整数
    return int(filename.split('.')[0][5:])
    

def test(args):
    # device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    # Setup image
    images_dir = "/home/leo/DatasetMMF/OSDT/test/image128"
    imgname_list = sorted(os.listdir(images_dir), key=sort_key)

    classes = ['Water', 'oil']
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

    # print("out_path", out_path)

    ## Set threshold
    threshold = args.threshold

    # Setup model
    data_path = cfg['data']['path']
    test_split = cfg['data']['test_split']
    # test_split = cfg['data']['val_split']
    img_size = cfg['data']['img_size']
    batchsize = cfg['training']['batch_size']
    test_batch_size = cfg['training']['test_batch_size']
    n_workers = cfg['training']['n_workers']

    t_loader = Road_loader(data_path, test_split, img_size, is_augmentation=False)
    testloader = data.DataLoader(t_loader, batch_size = test_batch_size, shuffle=False, num_workers=n_workers)

    # model_file_name = os.path.split(args.model_path)[1]
    # model_name = model_file_name[: model_file_name.find("_")]
    # model = CANet(num_class = 1, backbone = 'ResNet-50', pretrained=True, pcca5=True)
    # model = CMGFNet(n_classes=2, pretrained=True)
    # if model == "DE_CCFNet_34_multi":
    #     model = DE_CCFNet_34_multi(n_classes=2, is_pretrained="ResNet34_Weights.IMAGENET1K_V1")
    model_name = cfg['model']
    model = get_model(model_name, len(classes)).to(args.device)

    # state = convert_state_dict(torch.load(args.model_path)["model_state"])    # multi-gpus
    record = torch.load(args.model_path)
    state = record["model_state"]     # single-gpu
    model.load_state_dict(state)
    print("successfully load model from {}".format(args.model_path))

    # plot_training_results
    savefig_path = os.path.split(args.model_path)[0]
    results_train = pd.DataFrame(record['results_train'])
    results_val = pd.DataFrame(record['results_val'])
    
    plot_training_results(results_train, results_val, model_name, savefig_path)


    model.eval()
    model.to(args.device)

    # Setup Metrics
    running_metrics_test = runningScore(len(classes))
    test_log = Logger(os.path.join(os.path.split(args.model_path)[0], \
                                   'test_result.log'))

    ########################### test ####################################
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for ind, (gaofen, lidar, mask) in tqdm(enumerate(testloader)):
            img_id = imgname_list[ind]

            if args.TTA:
                # # use TTA
                # open image

                # gaofen
                gaofen = np.array(gaofen, np.float32)
                # gaofen = norm2(gaofen, gaofen_mean, gaofen_std)
                gaofen90 = np.array(np.rot90(gaofen))
                gaofen1 = np.concatenate([gaofen[None], gaofen90[None]])
                gaofen2 = np.array(gaofen1)[:,::-1].copy()
                gaofen3 = np.concatenate([gaofen1, gaofen2])
                gaofen4 = np.array(gaofen3)[:,:,::-1].copy()
                gaofen5 = gaofen3.transpose(0, 3, 1, 2)
                gaofen5 = torch.Tensor(gaofen5).to(args.device)

                gaofen6 = gaofen4.transpose(0, 3, 1, 2)
                gaofen6 = torch.Tensor(gaofen6).to(args.device)

                # lidar
                lidar = np.array(lidar, np.float32)
                lidar90 = np.array(np.rot90(lidar))
                lidar1 = np.concatenate([lidar[None],lidar90[None]])
                lidar2 = np.array(lidar1)[:,::-1].copy()
                lidar3 = np.concatenate([lidar1,lidar2])
                lidar4 = np.array(lidar3)[:,:,::-1].copy()
                lidar5=lidar3.transpose(0, 3, 1, 2)
                # lidar5 = norm(lidar5)
                lidar5 = torch.Tensor(lidar5).to(args.device)

                lidar6=lidar4.transpose(0, 3, 1, 2)
                # lidar6 = norm(lidar6)
                lidar6 = torch.Tensor(lidar6).to(args.device)

                # multi source
                preda = model(gaofen5, lidar5)
                predb = model(gaofen6, lidar6)

                maska = preda.squeeze().cpu().data.numpy()  # .squeeze(1)
                maskb = predb.squeeze().cpu().data.numpy()

                ## single source
                # maska = model(gaofen5).squeeze().cpu().data.numpy()  # .squeeze(1)
                # maskb = model(gaofen6).squeeze().cpu().data.numpy()

                mask1 = maska + maskb[:, :, ::-1]
                mask2 = mask1[:2] + mask1[2:, ::-1]
                mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]
                pred = mask3
                pred = np.expand_dims(pred, 0)
                pred = np.expand_dims(pred, 0)

                pred[pred <= threshold * 8] = 0
                pred[pred > threshold * 8] = 1
                running_metrics_test.update(mask.cpu().numpy(), pred)

            else:
                gaofen = gaofen.to(args.device)
                lidar = lidar.to(args.device)

                pred = model(gaofen, lidar)
                # print("pred 1: ", pred.shape)

                pred[pred <= threshold] = 0
                pred[pred > threshold] = 1
                pred = pred.data.cpu().numpy().astype(np.uint8)
                # print(type(mask), type(pred))
                running_metrics_test.update(mask.cpu().numpy(), pred)

        ############################### save pred image ###############################
            pred = pred.reshape(128, 128)
            # print("pred 2: ", pred.shape)
            # print("pred", np.unique(pred))
            cv2.imwrite(os.path.join(out_path, str(img_id) + '.png'), id_to_color[pred])
            # tifffile.imwrite(os.path.join(out_path, str(img_id) + '.tif'), pred.astype(np.uint8))

        # print and save metrics result
        score, class_iou = running_metrics_test.get_scores()
        test_log.write('************test_result**********\n')
        test_log.write('{}: '.format(args.TTA) + '\n')

        # for k, v in class_iou.items():
        #     # print(k, v)
        #     test_log.write('{}  \t\t\t: {}'.format(k, round(v * 100, 2)) + '\n')
        for k, v in score.items():
            # print(k, v)
            test_log.write('{}: {}'.format(k, round(np.nanmean(v) * 100, 2)) + '\n')
            # test_log.write('{}: {}'.format(k, round(v * 100, 2)) + '\n')
        
        t1 = time.time()
        img_write_time = t1 - t0
        test_log.write('{}    \t: {}'.format("time", round(img_write_time, 2)) + '\n')
        
        test_log.flush()
        test_log.write('Finish!\n')
        test_log.close()
        
        running_metrics_test.reset()

        # print("out_path", out_path)
        ############################### save pred image ###############################

        # printMetrics(submit_path = out_path, mask_path = masks_path,
        #              running_metrics = running_metrics_test, log = test_log)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument("--device", nargs = "?", type = str, default = "cuda:0", help="CPU or GPU")
    parser.add_argument("--imgs_path", nargs = "?", type = str, \
                        default = '/home/leo/MMF/OSD_H', \
                        help="Path of the input image")
    parser.add_argument("--TTA", nargs="?", type=bool, default=False) # default=False, help="default use TTA",
    parser.add_argument("--threshold", nargs = "?", type = float, default = 0.5, \
                        help="Dataset to use ['pascal, camvid, ade20k etc']")
    parser.add_argument("--out_path", nargs = "?", type = str, default = '', help="Path of the output segmap")


    parser.add_argument("--config", nargs = "?", type = str,
                        # default = "/home/leo/Semantic_Segmentation/multiRoadHSI/config/extraction_epoch_DE_CCFNet18.yml",
                        # default = "/home/leo/Semantic_Segmentation/multiRoadHSI/config/extraction_epoch_DE_CCFNet34.yml",
                        # default = "/home/leo/Semantic_Segmentation/multiRoadHSI/config/extraction_epoch_DE_DCGCN.yml",
                        # default = "/home/leo/Semantic_Segmentation/multiRoadHSI/config/extraction_epoch_HAFNetE.yml",
                        default = "/home/leo/Semantic_Segmentation/multiRoadHSI/config/extraction_epoch_AsymFormer.yml",
                        # default = "/home/leo/Semantic_Segmentation/multiRoadHSI/config/extraction_epoch_PCG.yml",
                        # default = "/home/leo/Semantic_Segmentation/multiRoadHSI/config/extraction_epoch_SFAFMA.yml",
                        # default = "/home/leo/Semantic_Segmentation/multiRoadHSI/config/extraction_epoch_DE_CCFNet18_att.yml",
                        help="Configuration file to use")
    parser.add_argument("--model_path", nargs = "?", type = str, \
                        # default = os.path.join("/home/leo/Semantic_Segmentation/multiRoadHSI/FinalResults/0723-0927-DE_CCFNet_18", "best_model_temp.pkl"),
                        # default = os.path.join("/home/leo/Semantic_Segmentation/multiRoadHSI/runs/0919-1402-HAFNetE", "best_model_temp.pkl"),
                        default = os.path.join("/home/leo/Semantic_Segmentation/multiRoadHSI/runs/0302-0039-AsymFormer", "best_model_temp.pkl"),
                        # default = os.path.join("/home/leo/Semantic_Segmentation/multiRoadHSI/FinalResults/0728-1001-DE_DCGCN", "best_model_temp.pkl"),
                        # default = os.path.join("/home/leo/Semantic_Segmentation/multiRoadHSI/runs/0919-0101-HAFNetE", "best_model_temp.pkl"),
                        # default = os.path.join("/home/leo/Semantic_Segmentation/multiRoadHSI/FinalResults/0803-1407-PCGNet", "best_model_temp.pkl"),
                        # default = os.path.join("/home/leo/Semantic_Segmentation/multiRoadHSI/FinalResults/0803-1202-SFAFMA50", "best_model_temp.pkl"),
                        help="Path to the saved model")

    args = parser.parse_args(args=[])

    # if os.path.split(args.model_path)[1] == "":
    #     pass 

    test(args)

