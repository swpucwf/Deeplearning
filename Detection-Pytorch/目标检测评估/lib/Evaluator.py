import os
import sys
from collections import Counter
import time
import matplotlib.pyplot as plt
import numpy as np


class Evaluator:


    def GetPascalVOCMetrics(self,
                            cfg,
                            classes,
                            gt_boxes,
                            num_pos,
                            det_boxes):
        ret = []
        groundTruths = []
        detections = []
        #
        for c in classes:
            # 遍历每个类别
            dects = det_boxes[c]
            gt_class = gt_boxes[c]
            npos = num_pos[c]

            # 根据置信度排序
            dects = sorted(dects, key=lambda conf: conf[4], reverse=True)

            TP = np.zeros(len(dects))
            FP = np.zeros(len(dects))

            for d in range(len(dects)):
                iouMax = sys.float_info.min
                if dects[d][-1] in gt_class:
                    for j in range(len(gt_class[dects[d][-1]])):
                        iou = Evaluator.iou(dects[d][:4], gt_class[dects[d][-1]][j][:4])
                        if iou > iouMax:
                            iouMax = iou
                            jmax = j

                    if iouMax >= cfg['iouThreshold']:
                        '''
                        大于阈值
                        '''
                        if gt_class[dects[d][-1]][jmax][4] == 0:
                            '''
                            从未匹配过，标志位1
                            '''
                            TP[d] = 1
                            gt_class[dects[d][-1]][jmax][4] == 1
                        else:
                            FP[d] = 1
                    else:
                        FP[d] = 1
                else:
                    FP[d] = 1

            acc_FP = np.cumsum(FP)
            acc_TP = np.cumsum(TP)
            rec = acc_TP / npos
            prec = np.divide(acc_TP, (acc_FP + acc_TP))
            # print(' ')
            [ap, mpre, mrec, ii] = Evaluator.CalculateAveragePrecision(rec, prec)
            r = {
                'class': c,
                'precision': prec,
                'recall': rec,
                'AP': ap,
                'interpolated precision': mpre,
                'interpolated recall': mrec,
                'total positives': npos,
                'total TP': np.sum(TP),
                'total FP': np.sum(FP),
            }
            ret.append(r)
        return ret, classes

    @staticmethod
    def CalculateAveragePrecision(rec, prec):

        '''
        rec : recall 召回率
        prec: 准确度
        '''
        # 添加 0和1

        mrec = []
        mrec.append(0)
        [mrec.append(e) for e in rec]
        mrec.append(1)
        mpre = []
        mpre.append(0)
        [mpre.append(e) for e in prec]
        mpre.append(0)


       # print("mpre",mpre)
        for i in range( 0, len(mpre) - 1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])

        ii = []
        for i in range(len(mrec) - 1):
            if mrec[i + 1] != mrec[i]:
                # 不等于就+1
                ii.append(i + 1)
        ap = 0

        for i in ii:
            # 积分
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
        return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]

    @staticmethod
    def iou(boxA, boxB):
        '''
        计算iou
        '''
        # if boxes dont intersect
        if Evaluator._boxesIntersect(boxA, boxB) is False:
            return 0
        interArea = Evaluator._getIntersectionArea(boxA, boxB)
        union = Evaluator._getUnionAreas(boxA, boxB, interArea=interArea)
        # intersection over union
        iou = interArea / union
        if iou < 0:
            import pdb
            pdb.set_trace()
        assert iou >= 0
        return iou

    # boxA = (Ax1,Ay1,Ax2,Ay2)
    # boxB = (Bx1,By1,Bx2,By2)
    @staticmethod
    def _boxesIntersect(boxA, boxB):
        '''
        判断框是否正确
        '''
        if boxA[0] > boxB[2]:
            return False  # boxA is right of boxB
        if boxB[0] > boxA[2]:
            return False  # boxA is left of boxB
        if boxA[3] < boxB[1]:
            return False  # boxA is above boxB
        if boxA[1] > boxB[3]:
            return False  # boxA is below boxB
        return True

    @staticmethod
    def _getIntersectionArea(boxA, boxB):

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # intersection area
        return (xB - xA + 1) * (yB - yA + 1)

    @staticmethod
    def _getUnionAreas(boxA, boxB, interArea=None):
        '''联合面积'''
        area_A = Evaluator._getArea(boxA)
        area_B = Evaluator._getArea(boxB)
        if interArea is None:
            interArea = Evaluator._getIntersectionArea(boxA, boxB)
        return float(area_A + area_B - interArea)

    @staticmethod
    def _getArea(box):
        '''
        单个框的面积
        '''
        return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
