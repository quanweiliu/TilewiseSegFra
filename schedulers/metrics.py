# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np

class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.eps = 1e-8
        # self.eps = 0

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask].astype(int),
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes
            )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


    def get_scores(self, ignore_index=None):
        """
        Returns segmentation evaluation metrics:
            - Overall Accuracy (OA)
            - Per-class Accuracy (excluding ignore class)
            - Per-class IoU
            - Mean IoU (excluding ignore class)
            - Precision, Recall, F1 (macro average, excluding ignore class)
        """
        hist = self.confusion_matrix
        eps = 1e-6
        n_class = self.n_classes

        # --- 1. 确定需要计算的有效类别 ---
        if ignore_index is not None:
            # 将 ignore_index 统一处理为列表
            if isinstance(ignore_index, int):
                ignore_index = [ignore_index]
            
            # 创建一个布尔掩码，标记哪些类别是有效的
            valid_mask = np.ones(n_class, dtype=bool)
            for idx in ignore_index:
                if 0 <= idx < n_class:
                    valid_mask[idx] = False
            valid_classes = np.where(valid_mask)[0]
        else:
            # 如果不忽略任何类，则所有类都有效
            valid_classes = np.arange(n_class)
        
        # 如果所有类别都被忽略了，返回空结果
        if len(valid_classes) == 0:
            return {}, {}
        
        # --- 2. 计算 OA (只在有效类别上计算) ---
        # 使用 np.ix_ 从原始混淆矩阵中提取有效类别的子矩阵
        valid_hist = hist[np.ix_(valid_classes, valid_classes)]
        OA_valid = np.diag(valid_hist).sum() / (valid_hist.sum() + eps)

        # 计算所有类别的基础指标 TP, FP, FN
        TP = np.diag(hist)
        FP = hist.sum(axis=0) - TP
        FN = hist.sum(axis=1) - TP
        TN = np.diag(self.confusion_matrix).sum() - np.diag(self.confusion_matrix)


        # po = (TP + TN) / (TP + TN + FP + FN)
        # pe = ((TP + FP) * (TP + FN) + (FP + TN) * (FN + TN)) / (TP + TN + FP + FN) ** 2
        # Kappa = (po - pe) / (1 - pe + eps)

        IoU = TP / (TP + FP + FN + eps)
        Precision = TP / (TP + FP + eps)
        Recall = TP / (TP + FN + eps)                       # 在机器学习中通常被称为 召回率 (Recall)，在语义分割任务中也常被称为 类别准确率 (Per-Class Accuracy)
        acc_cls = np.diag(hist) / (hist.sum(axis=1) + eps)  # 和上面的计算方式得到的结果一样
        F1 = 2 * Precision * Recall / (Precision + Recall + eps)

        # 只保留非背景类指标用于平均
        IoU_valid = IoU[valid_classes].mean()
        F1_valid = F1[valid_classes].mean()
        Precision_valid = Precision[valid_classes].mean()
        Recall_valid = Recall[valid_classes].mean()
        # acc_cls_valid = acc_cls[valid_classes]  # 和上面的 recall 一样
        # Kappa_valid = Kappa[valid_classes].mean()
        # print("Kappa", Kappa, "Kappa_valid", Kappa_valid)

        results = {
            "OA \t\t\t": OA_valid,
            "mIoU  \t\t": IoU_valid,
            "F1  \t\t": F1_valid,
            "Precision\t": Precision_valid,
            "Recall  \t": Recall_valid,
        }

        # 输出每类指标（包括背景类也一起打印出来）
        for i in range(n_class):
            # results[f"Recall {i} Acc"] = Recall[i]
            results[f"Class {i} Acc "] = acc_cls[i]
            # results[f"Class {i} IoU"] = IoU[i]
            # results[f"Class {i} F1"] = F1[i]

        return results, Recall[valid_classes]


class averageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':

    gt = np.array([[0, 2, 1],
                   [1, 2, 1],
                   [1, 0, 1]])

    pre = np.array([[0, 1, 1],
                   [2, 0, 1],
                   [1, 1, 1]])
    
    # gt = np.array([0, 0, 1, 1, 1, 1, 1, 0])
    # pre = np.array([0, 1, 0, 1, 1, 1, 1, 0])
    # gt = np.array([0, 0, 1, 1, 1, 1])
    # pre = np.array([0, 1, 0, 1, 1, 1])
    # print(gt.shape, pre.shape)

    running_metrics_test = runningScore(n_classes=3)
    # eval.add_batch(gt, pre)
    running_metrics_test.update(gt, pre)
    score, class_iou = running_metrics_test.get_scores()
    # print(score.items(), class_iou)
    # print("score", score.keys())

    # print("classAcc 0", score["Class 0 Acc"])
    # print("classAcc 1", score["Class 1 Acc"])
    # print("OA       ", score["OA \t\t\t"])
    # print("Precision", score["Precision\t"])
    # print("Recall   ", score["Recall  \t"])
    # print("F1       ", score["F1  \t\t"])
    # # print("Kappa    ", score["Kappa: \t\t"])
    # print("mIoU      ", score["mIoU  \t\t"])
    # print("Class IoU", class_iou)
    print(score)