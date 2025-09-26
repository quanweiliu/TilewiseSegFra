import os
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, [1]))
print('using GPU %s' % ','.join(map(str, [1])))

import logging
import random
import shutil
import time
import yaml
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from thop import profile, clever_format

import torch
from torch.utils import data
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from dataLoader.OSTD_loader import OSTD_loader
from dataLoader.OSTD_loader2 import OSTD_loader2
from dataLoader.ISPRS_loader import ISPRS_loader
from dataLoader.ISPRS_loader3 import ISPRS_loader3
from dataLoader import ISPRS_loader2
from ptsemseg import get_logger
from ptsemseg.loss import get_loss_function
from ptsemseg.models import get_model
from ptsemseg.optimizers import get_optimizer
# from ptsemseg.schedulers2 import get_scheduler, WarmupLR
# from ptsemseg.schedulers2.warmuplr import WarmupCosineLR
from schedulers.metrics import runningScore, averageMeter
from tools.utils import plot_training_results

def train(cfg, rundir):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda")
    torch.set_num_threads(1)

    # seed=1337
    # torch.manual_seed(cfg.get('seed', seed))
    # torch.cuda.manual_seed(cfg.get('seed', seed))
    # np.random.seed(cfg.get('seed', seed))
    # random.seed(cfg.get('seed', seed))

    # cfg parameters
    data_name = cfg['data']['name']
    data_path = cfg['data']['path']
    train_split = cfg['data']['train_split']
    val_split = cfg['data']['val_split']
    img_size = cfg['data']['img_size']
    bands1 = cfg['data']['bands1']
    bands2 = cfg['data']['bands2']
    classes = cfg['data']['classes']
    batchsize = cfg['training']['batch_size']
    epoch = cfg['training']['train_epoch']
    n_workers = cfg['training']['n_workers']
    classification = cfg["data"]["classification"]
    print("img_size", img_size)



    # Setup Dataloader
    if data_name == "OSTD":
        t_loader = OSTD_loader(data_path, train_split, img_size, is_augmentation=True)
        v_loader = OSTD_loader(data_path, val_split, img_size, is_augmentation=False)

        # t_loader = OSTD_loader2(data_path, train_split, img_size, is_augmentation=True)
        # v_loader = OSTD_loader2(data_path, val_split, img_size, is_augmentation=False)
        running_metrics_train = runningScore(classes+1)
        running_metrics_val = runningScore(classes+1)

    elif data_name == "Vaihingen" or data_name == "Potsdam":
        t_loader = ISPRS_loader(data_path, train_split, img_size, classes, data_name, is_augmentation=True)
        v_loader = ISPRS_loader(data_path, val_split, img_size, classes, data_name, is_augmentation=False)
        # t_loader = ISPRS_loader3(data_path, 'train.txt', img_size, is_augmentation=True)
        # v_loader = ISPRS_loader3(data_path, 'val.txt', img_size, is_augmentation=False)
        running_metrics_train = runningScore(classes)
        running_metrics_val = runningScore(classes)

    trainloader = data.DataLoader(t_loader, batch_size=batchsize, shuffle=True,
                                num_workers=n_workers, prefetch_factor=4, pin_memory=True)
    valloader = data.DataLoader(v_loader, batch_size=batchsize, shuffle=False,
                                num_workers=n_workers, prefetch_factor=4, pin_memory=True)

    # for gaofens, lidars, labels in trainloader:
    #     print("train gaofens", gaofens.shape)
    #     print("train lidars", lidars.shape)
    #     print("train labels", labels.shape)
    #     break

    # for gaofens, lidars, labels in valloader:
    #     print("val gaofens", gaofens.shape)
    #     print("val lidars", lidars.shape)
    #     print("val labels", labels.shape)
    #     break

    # Set Model
    model = get_model(cfg['model'], bands1, bands2, classes, classification, img_size).to(device)

    ## Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)

    # 单一 学习率 更新 (?)
    optimizer_params = {k:v for k, v in cfg['training']['optimizer'].items() if k != 'name'}
    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    # optimizer=torch.optim.Adam(model.parameters(), lr=cfg['training']['optimizer']['lr'],
    #                            betas=[cfg['training']['optimizer']['momentum'], 0.999],
    #                            weight_decay=cfg['training']['optimizer']['weight_decay'])
    # logger.info("Using optimizer {}".format(optimizer))

    ## scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.6, patience=7, min_lr=0.0000001)
    lr  = scheduler.get_last_lr()
    print("epoch: ", epoch, lr)

    # loss_function
    loss_fn = get_loss_function(cfg)

    ################################# FLOPs and Params ###################################################
    # # input = torch.randn(2, 1, hsi_bands+sar_bands, args.patch_size, args.patch_size).to(args.device)
    # # input = torch.randn(2, 3, 128, 128).to(device)
    # input=(torch.randn(2, 193, 128, 128).to(device), torch.randn(2, 3, 128, 128).to(device))
    # # print(input.shape)

    # flops, params = profile(model, inputs=input)
    # # flops, params = profile(model, inputs=(torch.randn(2, hsi_bands).to(args.device), torch.randn(2, sar_bands).to(args.device)))

    # flops, params = clever_format([flops, params])
    # print('# Model FLOPs: {}'.format(flops))
    # print('# Model Params: {}'.format(params))


    # 初始化 log-dir 用于后续画图
    logger2 = initLogger(cfg['model']['arch'], rundir)

################################# retrain ###################################################
    if args.model_path is not None:
        resume = torch.load(args.model_path, weights_only=False)
        start_epoch = resume["epoch"]
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        scheduler.load_state_dict(resume["scheduler_state"])
        best_iou = resume["best_iou"]
        results_train = resume["results_train"]
        results_val = resume["results_val"]
        print("successfully load model from {}, Epoch {}".format(args.model_path, start_epoch))
    else:
        start_epoch = 0
        print("start from scratch, no model loaded")
################################# train ###################################################
    results_train = []
    results_val = []

    train_loss_meter = averageMeter()
    val_loss_meter = averageMeter()  #val_loss
    time_meter = averageMeter()
    best_iou2 = -100.0
    best_iou = -100.0
    i = start_epoch
    flag = True 

    while i < cfg['training']['train_epoch'] and flag:      #  Number of total training iterations
        ## every epoch
        i += 1
        model.train()
        print('current lr: ', optimizer.state_dict()['param_groups'][0]['lr'])
        start_ts = time.time()
        for (gaofens, lidars, labels) in tqdm(trainloader):
            gaofens = gaofens.to(device)
            lidars = lidars.to(device)
            labels = labels.to(device)
            # print("gaofens", gaofens.shape)
            # print("lidars", lidars.shape)
            # print("labels", labels.shape, labels.dtype)

            outputs = model(gaofens, lidars)
            loss, loss1, loss2 = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if classification == "Multi":
                pred = outputs.argmax(dim=1).cpu().numpy()  # [B, H, W]
            elif classification == "Binary":
                outputs[outputs > cfg['threshold']] = 1
                outputs[outputs <= cfg['threshold']] = 0
                pred = outputs.data.cpu().numpy()
            gt = labels.data.cpu().numpy()
            # update each train batchsize metric and loss
            running_metrics_train.update(gt, pred)  # update confusion_matrix
            train_loss_meter.update(loss.item())  # update sum_loss

        time_meter.update(time.time() - start_ts)

        ############## print result for each train epoch ############################
        print("Epoch [{:d}/{:d}]  Loss: {:.4f} Time/Image: {:.4f}".format(
            i, cfg['training']['train_epoch'], loss.item(), time_meter.avg))
        train_score, train_class_iou = running_metrics_train.get_scores()

        # store results
        results_train.append({'epoch': i, 
                        'trainLoss': train_loss_meter.avg, 
                        'F1': np.nanmean(train_score["F1  \t\t"]),
                        # "Kappa": np.nanmean(train_score["Kappa: \t\t"]),
                        "mIoU": np.nanmean(train_score["mIoU  \t\t"]),
                    })
        
        train_loss_meter.reset()
        running_metrics_train.reset()

        ############################## val ####################################
        # evaluate for each epoch
        if i % 1 == 0:
            model.eval()
            with torch.no_grad():
                for gaofens_val, lidars_val, labels_val in tqdm(valloader):
                    gaofens_val = gaofens_val.to(device)
                    lidars_val = lidars_val.to(device)
                    labels_val = labels_val.to(device)

                    outputs = model(gaofens_val, lidars_val)
                    val_loss, val_loss1, val_loss2 = loss_fn(outputs, labels_val)
                    if classification == "Multi":
                        pred = outputs.argmax(dim=1).cpu().numpy()  # [B, H, W]

                    elif classification == "Binary":
                        outputs[outputs > cfg['threshold']] = 1
                        outputs[outputs <= cfg['threshold']] = 0
                        pred = outputs.data.cpu().numpy()
                    gt = labels_val.data.cpu().numpy()

                    # update each val batchsize metric and loss
                    running_metrics_val.update(gt, pred)     # update confusion_matrix
                    val_loss_meter.update(val_loss.item())  # update sum_loss

            score, class_iou = running_metrics_val.get_scores()

            # store results
            results_val.append({'epoch': i, 
                            'valLoss': val_loss_meter.avg, 
                            'F1': np.nanmean(score["F1  \t\t"]),
                            "mIOU": np.nanmean(score["mIoU  \t\t"]),
                        })

            logger2.info('Epoch ({}) | Loss: {:.4f} | Tra_F1 {:.2f} Tra_IOU {:.2f} Val_F1 {:.2f} Val_IOU {:.2f} Tra_Time {:.2f}'.format(
                i,
                val_loss_meter.avg,
                np.nanmean(train_score["F1  \t\t"]).round(4)*100,
                np.nanmean(train_score["mIoU  \t\t"]).round(4)*100,
                np.nanmean(score["F1  \t\t"]).round(4)*100,
                np.nanmean(score["mIoU  \t\t"]).round(4)*100,
                time_meter.avg
            ))
            val_loss_meter.reset()
            running_metrics_val.reset()

            # save best model by mIoU
            val_IoU = np.nanmean(score["mIoU  \t\t"])
            scheduler.step(val_IoU)
            # if i <= warm_up:
            #     scheduler.step()
            # else:
            #     scheduler2.step(val_IoU)
            if val_IoU > best_iou:
                best_iou = val_IoU
                torch.save({
                    "epoch": i,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_iou": best_iou,
                    "results_train": results_train,
                    'results_val': results_val,
                }, f"{rundir}/best.pt")

                if (i) == cfg['training']['train_epoch']:
                    flag=False
                    break

    # plot results
    results_train = pd.DataFrame(results_train)
    results_val = pd.DataFrame(results_val)
    plot_training_results(results_train, results_val, cfg['model'])


def initLogger(model_name, run_dir):
    # 初始化log
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path = os.path.join(run_dir,'logs')
    os.mkdir(log_path)
    log_file_name = os.path.join(log_path, model_name + '_multi_road_metrics_' + rq + '.log')
    # logfile = log_name
    fh = logging.FileHandler(log_file_name, mode='w')
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


if __name__ ==  "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs = "?",
        type = str,
        # default = "/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/config/baseline18_double.yml",
        # default = "/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/config/baseline34_double.yml",
        # default = "/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/config/AsymFormer_b0.yml",
        # default = "/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/config/CMFNet.yml",
        # default = "/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/config/DE_CCFNet18.yml",
        # default = "/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/config/DE_CCFNet34.yml",
        # default = "/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/config/DE_DCGCN.yml",
        # default = "/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/config/FAFNet50.yml",
        # default = "/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/config/FAFNet101.yml",
        default = "/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/config/MCANet.yml",
        # default = "/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/config/MGFNet_Wei50.yml",
        # default = "/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/config/MGFNet_Wu34.yml",
        # default = "/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/config/MGFNet_Wu50.yml",
        # default = "/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/config/PACSCNet50.yml",
        # default = "/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/config/PCGNet18.yml",
        # default = "/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/config/SOLC.yml",
        # default = "/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/config/SFAFMA50.yml",
        # default = "/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/config/SFAFMA502.yml",
        help="Configuration file to use")

    parser.add_argument(
        "--results",
        type = str,
        # default = os.path.join("/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/run"),
        default = os.path.join("/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/run_Vai"),
        # default = os.path.join("/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/run_OSTD"),
        # default = os.path.join("/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/run_Potsdam"),
        help="Path to the saved model")
    
    parser.add_argument(
        "--model_path",
        type = str,
        # default = os.path.join("/home/icclab/Documents/lqw/Multimodal_Segmentation/multiISA/run/0703-0034-ACNet", "best.pt"),
        default = None,
        help="Path to the saved model")
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)
    
    run_id = datetime.now().strftime("%m%d-%H%M-") + cfg['model']['arch']
    rundir = os.path.join(args.results, str(run_id))
    os.makedirs(rundir, exist_ok=True)

    shutil.copy(args.config, rundir)   # copy config file to rundir

    train(cfg, rundir)
    time.sleep(30)