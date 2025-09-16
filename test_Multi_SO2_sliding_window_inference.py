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
from collections import namedtuple
from matplotlib.patches import Patch

import torch
from torch.utils import data
from torch.nn import functional as F
from torchvision import transforms
from ptsemseg.logger import Logger
from dataLoader.OSTD_loader import OSTD_loader
from dataLoader import ISPRS_loader2
from dataLoader import ISA_loader2
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

def _get_patch_coords(length, patch_size, stride):
    """返回滑窗左上角坐标列表，保证最后一个 patch 覆盖图像右/下边缘"""
    if length <= patch_size:
        return [0]
    coords = list(range(0, length - patch_size + 1, stride))
    if coords[-1] != length - patch_size:
        coords.append(length - patch_size)
    return coords

def sliding_window_predict(model, img, lidar, patch_size=400, stride=200, device="cuda"):
    """
    Args:
        model: torch model, 返回 (B, C, h, w) 或 (B,1,h,w)
        img: Tensor, (B,C,H,W)
        lidar: Tensor, (B,C,H,W)
        patch_size, stride: int
    Returns:
        outputs: Tensor (B, num_classes, H, W) 或 (B,1,H,W) （已经裁回原始 H,W 大小）
    """
    B, C, H, W = img.shape
    outputs_all = []

    # 逐样本处理（避免一次性把整个大图/批次放满显存）
    for b in range(B):
        single_gaofen = img[b:b+1]  # (1,C,H,W)
        single_lidar = lidar[b:b+1]  # (1,C,H,W)
        # 若图像任一边小于 patch_size，则 pad 到最小尺寸
        pad_h = max(0, patch_size - H)
        pad_w = max(0, patch_size - W)
        if pad_h > 0 or pad_w > 0:
            # pad = (left, right, top, bottom)
            single_padded_gaofen = F.pad(single_gaofen, (0, pad_w, 0, pad_h), mode="reflect")
            single_padded_lidar = F.pad(single_lidar, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            single_padded_gaofen = single_gaofen
            single_padded_lidar = single_lidar
        _, _, Hp, Wp = single_padded_gaofen.shape

        y_coords = _get_patch_coords(H, patch_size, stride)
        x_coords = _get_patch_coords(W, patch_size, stride)
        # print(f"b={b}, Hp={Hp}, Wp={Wp}, y_coords={y_coords}, x_coords={x_coords}")

        output_sum = None
        # count map 用于重叠区域求平均
        count_map = torch.zeros((1, 1, Hp, Wp), device=device, dtype=torch.float32)

        for y in y_coords:
            for x in x_coords:
                patch_gaofen = single_padded_gaofen[:, :, y:y+patch_size, x:x+patch_size]  # (1,C,ps,ps)
                patch_lidar = single_padded_lidar[:, :, y:y+patch_size, x:x+patch_size]  # (1,C,ps,ps)
                # print("patch", patch.shape)  # patch torch.Size([1, 3, 384, 384])
                pred = model(patch_gaofen, patch_lidar)  # 期望 (1, num_classes, ps, ps) 或 (1,1,ps,ps)
                # print("pred", pred.shape)  # pred torch.Size([1, 1, 384, 384])
                # 如果 model 返回 tuple/list（有些模型返回 (pred, aux)），取第一个
                if isinstance(pred, (tuple, list)):
                    pred = pred[0]
                pred = pred.detach()  # (1, nc, ps, ps)
                if output_sum is None:
                    nc = pred.shape[1]
                    output_sum = torch.zeros((1, nc, Hp, Wp), device=device, dtype=pred.dtype)
                output_sum[:, :, y:y+patch_size, x:x+patch_size] += pred
                count_map[:, :, y:y+patch_size, x:x+patch_size] += 1.0

        # 防止除0（理论上 count_map >0）
        output_avg = output_sum / count_map
        # 裁回原始大小（去掉前面的 padding）
        output_cropped = output_avg[:, :, :H, :W].cpu()  # 放回 CPU，便于后续处理
        outputs_all.append(output_cropped)

    outputs = torch.cat(outputs_all, dim=0)  # (B, nc, H, W)
    return outputs

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
        with open(os.path.join(args.imgs_path, 'test.txt'), "r") as f:
            imgname_list = sorted([x.strip() for x in f.readlines() if len(x.strip()) > 0])
        classes = ['ImpSurf', 'Building', 'Car', 'Tree', 'LowVeg', 'Clutter'] # 其中 Clutter # 是指 background
        val_transform = transforms.Compose([ISPRS_loader2.scaleNorm(args.img_size, args.img_size),
                                            ISPRS_loader2.ToTensor(),
                                            ISPRS_loader2.Normalize()])
        test_dataset = ISPRS_loader2.ISPRS_loader2(transform=val_transform, data_dir=args.imgs_path)
        running_metrics_test = runningScore(args.classes)

    elif args.data_name == "ISA":
        print("############ we use the ISA dataset ############")
        txt_path = os.path.join(args.imgs_path, 'val.txt')
        with open(os.path.join(txt_path), "r") as f:
            imgname_list = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        classes = ['NonISA', 'ISA'] # 其中 Clutter # 是指 background
        test_dataset = ISA_loader2.ISA_loader2(transforms.Compose([ISA_loader2.scaleNorm(args.img_size, args.img_size),
                                                        ISA_loader2.ToTensor(),
                                                        ISA_loader2.Normalize()]),
                                    phase_train=False,
                                    data_dir=args.imgs_path,
                                    txt_name='val.txt')
        running_metrics_test = runningScore(args.classes+1)

    testloader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)


    id_to_color, legend_elements = train_id_to_color(classes)
    model = get_model({"arch":args.model}, args.bands1, args.bands2, args.classes, args.classification).to(args.device)

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

    test_log = Logger(os.path.join(os.path.split(args.model_path)[0], 'test_result.log'))

    ########################### test ####################################
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for ind, sample in enumerate(tqdm(testloader)):
            img_id = imgname_list[ind]
            gaofen = sample["image"].to(args.device)
            lidar = sample["depth"].to(args.device)
            mask = sample["label"]

            outputs = sliding_window_predict(model, gaofen, lidar, args.patch_size, args.stride, args.device)

            if args.classification == "Multi":
                pred = outputs.argmax(dim=1).cpu().numpy().astype(np.uint8)  # [B, H, W]

            elif args.classification == "Binary":
                outputs = (outputs > args.threshold).int()
                pred = outputs.data.cpu().numpy().astype(np.uint8)
            running_metrics_test.update(mask.numpy(), pred)

        ############################### save pred image ###############################
            if args.save_img:
                pred = pred.reshape(args.img_size, args.img_size)
                cv2.imwrite(os.path.join(out_path, str(img_id) + '.png'), id_to_color[pred])
                if ind == 10:
                    break

        # print and save metrics result
        score, class_iou = running_metrics_test.get_scores(ignore_index=args.ignore_index)
        test_log.write('************test_result**********\n')
        test_log.write('{}: '.format(args.TTA) + '\n')

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
                         choices=["baseline18_double", "AsymFormer_b0", "baseline34_double", 'DE_CCFNet18', 'DE_CCFNet34', \
                                'DE_DCGCN', 'Zhiyang', "SFAFMA", "MCANet", "MGFNet50", 'MGFNet_Wei50', \
                                "MGFNet_Wu34", "MGFNet_Wu50", "PCGNet18", "PCGNet34", 'RDFNet50', \
                                "SFAFMA50", 'SOLC', 'PACSCNet50', 'FAFNet'], \
                        default="DE_CCFNet34", help="the model architecture that should be trained")
    parser.add_argument("--device", nargs = "?", type = str, default = "cuda:0", help="CPU or GPU")
    parser.add_argument("--split", type = str, default = "test", help="Dataset to use ['train, val, test']")
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold for binary classification')
    parser.add_argument('--n_workers', type=int, default=4, help='number of workers for validation data')
    parser.add_argument("--TTA", nargs="?", type=bool, default=False, help="default use TTA",) # default=False / True
    parser.add_argument("--out_path", nargs = "?", type = str, default = '', help="Path of the output segmap")
    parser.add_argument("--save_img", type=bool, default=False, help="whether save pred image or not")

    parser.add_argument("--file_path", nargs = "?", type = str,
                        # default = os.path.join("/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/run/0818-0951-baseline18_double"),
                        # default = os.path.join("/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/run/0903-2315-AsymFormer_b0"),
                        default = os.path.join("/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/run_ISA/0914-1048-DE_CCFNet34"),
                        # default = os.path.join("/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/run/0904-1053-DE_DCGCN"),
                        # default = os.path.join("/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/run/0904-1625-MGFNet_Wei50"),
                        # default = os.path.join("/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/run/0818-1039-SOLC"),
                        # default = os.path.join("/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/run/0905-2028-MGFNet_Wu34"),
                        # default = os.path.join("/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/run/0812-1954-PCGNet18"),
                        # default = os.path.join("/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/run/0812-2010-SFAFMA50"),
                        # default = os.path.join("/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/run/0810-2232-DE_CCFNet34"),
                        # default = os.path.join("/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/run/0813-1449-baseline18_double"),
                        # default = os.path.join("/home/icclab/Documents/lqw/Multimodal_Segmentation/TilewiseSegFra/run/0813-1449-baseline34_double"),
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
    args.classification = cfg['data']['classification']
    args.img_size = cfg['data']['img_size']
    args.img_size = 1600
    args.split = cfg['data']['test_split']
    args.batch_size = cfg['training']['test_batch_size']
    args.ignore_index = cfg['data']['ignore_index']
    args.threshold = cfg['threshold']
    args.patch_size = 1200
    args.stride = 400
    print("args", args.img_size, args.classes, args.ignore_index, args.threshold)
    test(args)