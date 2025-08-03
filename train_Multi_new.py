import os
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

from dataLoader.dataloader import Road_loader

# from dataLoader.dataloader_ISPRS import ISPRS_loader
from ptsemseg import get_logger
from ptsemseg.loss import get_loss_function
from ptsemseg.models import get_model
from ptsemseg.optimizers import get_optimizer
# from ptsemseg.schedulers2 import get_scheduler, WarmupLR
# from ptsemseg.schedulers2.warmuplr import WarmupCosineLR
from schedulers.metrics import runningScore, averageMeter
from tools.utils import plot_training_results


def train(cfg, rundir):

    # # Setup device
    device_num = cfg['training']['loss']['device']
    # device = torch.device("cuda:"+str(device_num) if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda")
    torch.set_num_threads(1)

    # seed=1337
    # torch.manual_seed(cfg.get('seed', seed))
    # torch.cuda.manual_seed(cfg.get('seed', seed))
    # np.random.seed(cfg.get('seed', seed))
    # random.seed(cfg.get('seed', seed))

    # cfg parameters
    data_path = cfg['data']['path']
    train_split = cfg['data']['train_split']
    val_split = cfg['data']['val_split']
    img_size = cfg['data']['img_size']
    batchsize = cfg['training']['batch_size']
    epoch = cfg['training']['train_epoch']
    n_workers = cfg['training']['n_workers']

    # print("data_path", data_path)

    # Setup Dataloader
    t_loader = Road_loader(data_path, train_split, img_size, is_augmentation=True)
    v_loader = Road_loader(data_path, val_split, img_size, is_augmentation=False)
    trainloader = data.DataLoader(t_loader, batch_size=batchsize, shuffle=True, num_workers=n_workers)
    valloader = data.DataLoader(v_loader, batch_size=batchsize, shuffle=False, num_workers=n_workers)

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

    # Setup Metrics
    n_classes = t_loader.n_classes
    running_metrics_val = runningScore(n_classes)
    running_metrics_train = runningScore(n_classes)

    # Set Model
    model_name = cfg['model']
    print("model_name", model_name)
    model = get_model(model_name, n_classes).to(device)

    # state = torch.load(args.model_path)["model_state"]  # single-gpu
    # model.load_state_dict(state)

    ## Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)

    # 单一 学习率 更新 (?)
    optimizer_params = {k:v for k, v in cfg['training']['optimizer'].items() if k != 'name'}
    # print("optimizer_params", optimizer_params)    # {'lr': 0.001}

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
    # logger.info("Using loss {}".format(loss_fn))

    ####################### FLOPs and Params #########################################
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

################################# train ###################################################
    start_epoch = 0
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
        print('current lr: ', optimizer.state_dict()['param_groups'][0]['lr'])
        start_ts = time.time()
        # logger.info('current lr: %.6f', optimizer.state_dict()['param_groups'][0]['lr'])
        for (gaofens, lidars, labels) in tqdm(trainloader):
            model.train()
            gaofens = gaofens.to(device)
            lidars = lidars.to(device)
            labels = labels.to(device)
            # print("gaofens", gaofens.shape)
            # print("lidars", lidars.shape)
            # print("labels", labels.shape)

            model = model.to(device)
            # print("gaofens", gaofens.shape, "lidars", lidars.shape, "labels", labels.shape)

            # ################ output ################
            multi_outputs = model(gaofens, lidars)    # 这是指大于 3 个输出吗？
            # print("multi_outputs", multi_outputs[0].shape, multi_outputs[1].shape, multi_outputs[2].shape)

            # ################ loss ################
            loss, loss1, loss2 = loss_fn(multi_outputs, labels)
            # print("loss", loss)


            # outputs = F.softmax(multi_outputs[0], dim=1)
            outputs = multi_outputs[0]
            # outputs = outputs[0]
            outputs[outputs > cfg['threshold']] = 1
            outputs[outputs <= cfg['threshold']] = 0
            # pred = outputs.data.max(1)[1].cpu().numpy()
            pred = outputs.data.cpu().numpy()
            gt = labels.data.cpu().numpy()
            # update each train batchsize metric and loss
            running_metrics_train.update(gt, pred)  # update confusion_matrix
            train_loss_meter.update(loss.item())  # update sum_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            # time_meter.update(time.time() - start_ts)
        time_meter = time.time() - start_ts

        print("Epoch [{:d}/{:d}]  Loss: {:.4f}".format(i,cfg['training']['train_epoch'],loss.item()))
        writer.add_scalar('loss/train_loss', loss.item(), i)
        writer.add_scalar('loss/train_loss1', loss1.item(), i)
        writer.add_scalar('loss/train_loss2', loss2.item(), i)
        # writer.add_scalar('loss/train_loss3', loss3.item(), i)

        train_score, train_class_iou = running_metrics_train.get_scores()

        # store results
        results_train.append({'epoch': i, 
                        'trainLoss': train_loss_meter.avg, 
                        'F1': np.nanmean(train_score["F1: \t\t"]),
                        "Kappa": np.nanmean(train_score["Kappa: \t\t"]),
                        "mIOU": np.nanmean(train_score["mIoU : \t\t"]),
                    })
        

        # logger2.info('TRAIN ({}) | Loss: {:.4f} | classAcc 0 {:.2f} classAcc 1 {:.2f} OA {:.2f} P {:.2f} R {:.2f} F1 {:.2f} Kappa {:.2f} IOU {:.2f} Time {:.4f}'.format(
        #     i,
        #     train_loss_meter.avg,
        #     np.nanmean(train_score["classAcc 0 :"]).round(4)*100,
        #     np.nanmean(train_score["classAcc 1 :"]).round(4)*100,
        #     np.nanmean(train_score["OA: \t\t"]).round(4)*100,
        #     np.nanmean(train_score["Precision : "]).round(4)*100,
        #     np.nanmean(train_score["Recall : \t"]).round(4)*100,
        #     np.nanmean(train_score["F1: \t\t"]).round(4)*100,
        #     np.nanmean(train_score["Kappa: \t\t"]).round(4)*100,
        #     np.nanmean(train_score["mIoU : \t\t"]).round(4)*100,
        #     time_meter
        # ))
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

                    # ################ output ################
                    multi_outputs = model(gaofens_val, lidars_val)

                    # outputs = model(gaofens_val, lidars_val)
                    # outputs, _, _ = model(gaofens_val, lidars_val)
                    # outputs = model(gaofens_val)
                    # outputs = model(lidars_val)

                    # ################ loss ################
                    val_loss, val_loss1, val_loss2 = loss_fn(multi_outputs, labels_val)
                    
                    # val_loss, val_loss1, val_loss2 = loss_fn(outputs, labels_val)
                    # val_loss, val_loss1, val_loss2, val_loss3 = loss_fn(multi_outputs, labels_val)
                    # val_loss, val_loss1, val_loss2 = loss_fn(outputs, outputs2, outputs3, labels_val)
                    # val_loss = loss_fn(outputs, refine_outputs, labels_val)
                    # val_loss = loss_fn(outputs, labels_val)

                    # outputs = F.softmax(multi_outputs[0], dim=1)
                    outputs = multi_outputs[0]
                    # outputs = outputs[0]
                    outputs[outputs > cfg['threshold']] = 1
                    outputs[outputs <= cfg['threshold']] = 0
                    pred = outputs.data.cpu().numpy()
                    # pred = outputs.data.max(1)[1].cpu().numpy()
                    gt = labels_val.data.cpu().numpy()

                    # update each val batchsize metric and loss
                    running_metrics_val.update(gt, pred)     # update confusion_matrix
                    val_loss_meter.update(val_loss.item())  # update sum_loss

            # logger.info("Epoch %d Loss: %.4f" % (i, val_loss_meter.avg))

            score, class_iou = running_metrics_val.get_scores()

            # store results
            results_val.append({'epoch': i, 
                            'valLoss': val_loss_meter.avg, 
                            'F1': np.nanmean(score["F1: \t\t"]),
                            "Kappa": np.nanmean(score["Kappa: \t\t"]),
                            "mIOU": np.nanmean(score["mIoU : \t\t"]),
                        })

            # logger2.info('VAL ({}) | Loss: {:.4f} | classAcc 0 {:.2f} classAcc 1 {:.2f} OA {:.2f} P {:.2f} R {:.2f} F1 {:.2f} Kappa {:.2f} IOU {:.2f}'.format(
            #     i,
            #     val_loss_meter.avg,
            #     np.nanmean(score["classAcc 0 :"]).round(4)*100,
            #     np.nanmean(score["classAcc 1 :"]).round(4)*100,
            #     np.nanmean(score["OA: \t\t"]).round(4)*100,
            #     np.nanmean(score["Precision : "]).round(4)*100,
            #     np.nanmean(score["Recall : \t"]).round(4)*100,
            #     np.nanmean(score["F1: \t\t"]).round(4)*100,
            #     np.nanmean(score["Kappa: \t\t"]).round(4)*100,
            #     np.nanmean(score["mIoU : \t\t"]).round(4)*100,
            # ))
            logger2.info('Epoch ({}) | Loss: {:.4f} | Tra_F1 {:.2f} Tra_Kappa {:.2f} Tra_IOU {:.2f} Val_F1 {:.2f} Val_Kappa {:.2f} Val_IOU {:.2f}'.format(
                i,
                val_loss_meter.avg,
                np.nanmean(train_score["F1: \t\t"]).round(4)*100,
                np.nanmean(train_score["Kappa: \t\t"]).round(4)*100,
                np.nanmean(train_score["mIoU : \t\t"]).round(4)*100,
                np.nanmean(score["F1: \t\t"]).round(4)*100,
                np.nanmean(score["Kappa: \t\t"]).round(4)*100,
                np.nanmean(score["mIoU : \t\t"]).round(4)*100,
            ))
            val_loss_meter.reset()
            running_metrics_val.reset()

            # save best model by mIoU
            val_IoU = np.nanmean(score["mIoU : \t\t"])
            # gaofen_scheduler.step(val_IoU)
            # scheduler.step()
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
    plot_training_results(results_train, results_val, model_name)


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
        default = "/home/icclab/Documents/lqw/Multimodal_Segmentation/multiRoad/config/extraction_epoch_ACNet.yml",
        # default = "/home/leo/Semantic_Segmentation/multiRoadHSI/config/extraction_epoch_ACNet2.yml",
        # default = "/home/leo/Semantic_Segmentation/multiRoadHSI/config/extraction_epoch_CANet.yml",
        # default = "/home/leo/Semantic_Segmentation/multiRoadHSI/config/extraction_epoch_CMANet.yml",
        # default = "/home/leo/Semantic_Segmentation/multiRoadHSI/config/extraction_epoch_CMGF.yml",
        # default = "/home/leo/Semantic_Segmentation/multiRoadHSI/config/extraction_epoch_CMGFNet_U.yml",
        help = "Configuration file to use")

    # for i in range(1):
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    run_id = datetime.now().strftime("%m%d-%H%M-") + cfg['model']['arch']
    # run_id = random.randint(1, 100000)
    rundir = os.path.join(cfg['results']['path'], str(run_id))
    os.makedirs(rundir, exist_ok=True)

    print("args.config", args.config)
    # print("basename", os.path.basename(args.config))
    # print("basename[-4]", os.path.basename(args.config)[:-4])
    # print('RUNDIR: {}'.format(rundir))

    writer = SummaryWriter(log_dir = rundir)
    shutil.copy(args.config, rundir)   # copy config file to rundir

    train(cfg, rundir)
    time.sleep(30)

