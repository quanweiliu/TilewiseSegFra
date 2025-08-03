import os
import yaml
import time
import tifffile
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from tools.utils import plot_training_results

import torch
from torch.utils import data
from ptsemseg.logger import Logger
from dataLoader.dataloader import Road_loader
# from ptsemseg.loss import dice_bce_gScore
from ptsemseg.models import get_model

from schedulers.metrics import runningScore, averageMeter
import scipy.io as scio


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


def test(args):
    # device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    # # Setup image
    # gaofens_path = os.path.join(args.imgs_path, 'test2', 'images_128')
    # lidars_path = os.path.join(args.imgs_path, 'test2', 'sar_128')
    # masks_path = os.path.join(args.imgs_path, 'test2', 'masks_128')
    # print("Read Input Image from : {}".format(gaofens_path))

    # imgname_list = os.listdir(gaofens_path)

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
    n_classes = 2
    data_path = cfg['data']['path']
    test_split = cfg['data']['test_split']
    # test_split = cfg['data']['val_split']
    img_size = cfg['data']['img_size']
    batchsize = cfg['training']['batch_size']
    test_batch_size = cfg['training']['test_batch_size']

    t_loader = Road_loader(data_path, test_split, img_size, is_augmentation=False)
    testloader = data.DataLoader(t_loader, batch_size = test_batch_size, shuffle=False)

    # model_file_name = os.path.split(args.model_path)[1]
    # model_name = model_file_name[: model_file_name.find("_")]
    # model = CANet(num_class = 1, backbone = 'ResNet-50', pretrained=True, pcca5=True)
    # model = CMGFNet(n_classes=2, pretrained=True)
    # if model == "DE_CCFNet_34_multi":
    #     model = DE_CCFNet_34_multi(n_classes=2, is_pretrained="ResNet34_Weights.IMAGENET1K_V1")
    model_name = cfg['model']
    model = get_model(model_name, n_classes).to(args.device)

    # state = convert_state_dict(torch.load(args.model_path)["model_state"])    # multi-gpus
    state = torch.load(args.model_path)["model_state"]     # single-gpu
    model.load_state_dict(state)
    print("successfully load model from {}".format(args.model_path))

    # plot_training_results
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
    running_metrics_test = runningScore(n_classes)
    test_log = Logger(os.path.join(os.path.split(args.model_path)[0], \
                                   'test_result.log'))

    ########################### test ####################################
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for ind, (gaofen, lidar, mask) in tqdm(enumerate(testloader)):
            # img_id = imgname_list[ind][:-4]

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
                # preda = model(gaofen5)
                # predb = model(gaofen6)
                # preda = model(lidar5)
                # predb = model(lidar6)
                # preda = model(gaofen5,lidar5)[0]
                # predb = model(gaofen6, lidar6)[0]

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

                # pred = model(gaofen, lidar)
                # print("pred: ", pred.shape)
                # pred = model(gaofen)
                pred = model(lidar)
                # pred = model(gaofen, lidar)[0]
                pred[pred <= threshold] = 0
                pred[pred > threshold] = 1
                pred = pred.data.cpu().numpy()
                # print(type(mask), type(pred))
                running_metrics_test.update(mask.cpu().numpy(), pred)

        ############################### save pred image ###############################

            pred = pred.reshape((128, 128))
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
                        default = "/home/leo/Semantic_Segmentation/multiRoadHSI/config/extraction_epoch_baseline18.yml",
                        # default = "/home/leo/Semantic_Segmentation/multiRoadHSI/config/extraction_epoch_baseline34.yml",
                        help="Configuration file to use")
    parser.add_argument("--model_path", nargs = "?", type = str, \
                        # default = os.path.join("/home/leo/Semantic_Segmentation/multiRoadHSI/runs/0721-0850-baseline18", "best_model_temp.pkl"),
                        default = os.path.join("/home/leo/Semantic_Segmentation/multiRoadHSI/runs/0722-2234-baseline18", "best_model_temp.pkl"),
                        help="Path to the saved model")

    args = parser.parse_args(args=[])

    # if os.path.split(args.model_path)[1] == "":
    #     pass 

    test(args)

