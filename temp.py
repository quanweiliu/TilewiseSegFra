import os
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, [1]))
print('using GPU %s' % ','.join(map(str, [1])))

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
from matplotlib.patches import Patch
from collections import namedtuple

import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
from dataLoader import ISA_loader2
# import dataLoader.dataloader_ISA_test as Data
from ptsemseg.logger import Logger
from ptsemseg.models import get_model
# from ptsemseg.schedulers2 import get_scheduler, WarmupLR
# from ptsemseg.schedulers2.warmuplr import WarmupCosineLR
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


def sort_key(filename):
    # 将文件名前缀（数字部分）提取出来并转换为整数
    return int(filename.split('.')[0][5:])
    

def test(args):
    # with open(args.config) as fp:
    #     cfg = yaml.safe_load(fp)

    # Setup image
    with open(os.path.join(args.imgs_path, "val.txt"), "r") as f:
        imgname_list = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
    classes = ['NonISA', 'ISA']
    id_to_color, legend_elements = train_id_to_color(classes)
    

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

    ## Set threshold
    threshold = args.threshold

    val_dataset = ISA_loader2.ISA_loader2(transforms.Compose([ISA_loader2.scaleNorm(),
                                                    ISA_loader2.ToTensor(),
                                                    ISA_loader2.Normalize()]),
                                 phase_train=False,
                                 data_dir=args.imgs_path,
                                 txt_name='val.txt')
    testloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, \
                            num_workers=args.n_workers, prefetch_factor=2, pin_memory=True)

    model = get_model({"arch":args.model}, 3, 2, 1, "Binary").to(args.device)

    # state = convert_state_dict(torch.load(args.model_path)["model_state"])    # multi-gpus
    checkpoint = torch.load(args.model_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    print("successfully load model from {}".format(args.model_path))

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
    t0 = time.time()
    with torch.no_grad():
        for ind, sample in enumerate(tqdm(testloader)):
            img_id = imgname_list[ind]
            gaofen = sample["image"]
            lidar = sample["depth"]
            mask = sample["label"]

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
                gaofen = sample["image"].to(args.device)
                lidar = sample["depth"].to(args.device)
                mask = sample["label"].to(args.device)

                pred = model(gaofen, lidar)
                # print("pred 1: ", pred.shape)

                pred[pred <= threshold] = 0
                pred[pred > threshold] = 1
                pred = pred.data.cpu().numpy().astype(np.uint8)
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
            # print(k, v)
            test_log.write('{}: {}'.format(k, round(np.nanmean(v) * 100, 2)) + '\n')
        
        t1 = time.time()
        img_write_time = t1 - t0
        test_log.write('{}    \t: {}'.format("time", round(img_write_time, 2)) + '\n')
        
        test_log.flush()
        test_log.write('Finish!\n')
        test_log.close()
        
        running_metrics_test.reset()


def mask2rle(img):
    '''
    Convert mask to RLE.
    img: numpy array,
    1 - mask,
    0 - background

    Returns run length as string formatted
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument('--model', choices=['DE_CCFNet18', 'DE_CCFNet34', 'DE_DCGCN', 'RDFNet', "AsymFormer", "PCG", "SFAFMA"], \
                        default='DE_CCFNet18', help="the model architecture that should be trained")
    parser.add_argument("--device", nargs = "?", type = str, default = "cuda:0", help="CPU or GPU")
    parser.add_argument("--imgs_path", nargs = "?", type = str, default = '/home/icclab/Documents/lqw/DatasetMMF/ISASeg', \
                        help="Path of the input image")
    parser.add_argument('--img_size', type=int, default=1600, help='size of the image patches the model should be trained on')
    parser.add_argument('--lr', type=float, default=3e-3, help='maximum learning rate')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size for training data')

    parser.add_argument('--n_workers', type=int, default=4, help='number of workers for validation data')
    parser.add_argument("--TTA", nargs="?", type=bool, default=False) # default=False, help="default use TTA",
    parser.add_argument("--threshold", nargs = "?", type = float, default = 0.5, \
                        help="Dataset to use ['pascal, camvid, ade20k etc']")
    parser.add_argument("--out_path", nargs = "?", type = str, default = '', help="Path of the output segmap")
    parser.add_argument("--save_img", type=bool, default=True, help="whether save pred image or not")


    # parser.add_argument("--model_path", nargs = "?", type = str, \
    #                     default = os.path.join("/home/icclab/Documents/lqw/Multimodal_Segmentation/multiISA/run/0613-1507-DE_CCFNet_18", "best.pt"),
    #                     help="Path to the saved model")
    parser.add_argument("--model_path", nargs = "?", type = str, \
                        default = os.path.join("/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/run_ISA/0908-1805-DE_CCFNet18", "best.pt"),
                        help="Path to the saved model")    
    args = parser.parse_args(args=[])

    test(args)

    csvfile = os.path.join(os.path.split(args.model_path)[0], "submit.csv")
    with open(csvfile, 'w', newline='') as f:
        csv_write = csv.writer(f, dialect='unix')
        csv_head = ["ID", "Result","Usage"]
        csv_write.writerow(csv_head)
        for id in tqdm(os.listdir(os.path.join(os.path.split(args.model_path)[0], "test"))):
            img_path = os.path.join(os.path.split(args.model_path)[0], "test", id)
            if not os.path.isfile(img_path):
                continue  # Skip if not a file
            img = Image.open(img_path)
            img = np.array(img)
            result = mask2rle(img)
            tmp = [id[0:-4], result,"Public"]
            csv_write.writerow(tmp)