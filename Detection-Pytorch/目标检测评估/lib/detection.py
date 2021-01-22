import os
from Evaluator import *
import pdb


def getGTBoxes(cfg, GTFolder):
    # 获取真实标签
    files = os.listdir(GTFolder)
    # print(files)
    files.sort()
    # 类别
    classes = []
    # 数量
    num_pos = {}
    # 真实框
    gt_boxes = {}
    for f in files:
        nameOfImage = f.replace(".txt", "")
        # 打开
        fh1 = open(os.path.join(GTFolder, f), "r")

        for line in fh1:
            line = line.replace("\n", "")
            # 空行省略
            if line.replace(' ', '') == '':
                continue
            # 空格分割
            splitLine = line.split(" ")
            # 获取对应的
            cls = (splitLine[0])  # class
            left = float(splitLine[1])
            top = float(splitLine[2])
            right = float(splitLine[3])
            bottom = float(splitLine[4])
            one_box = [left, top, right, bottom, 0]
            # 如果不在此类，则添加类别
            if cls not in classes:
                classes.append(cls)
                gt_boxes[cls] = {}
                num_pos[cls] = 0
            # 数量
            num_pos[cls] += 1

            if nameOfImage not in gt_boxes[cls]:
                gt_boxes[cls][nameOfImage] = []
            gt_boxes[cls][nameOfImage].append(one_box)

        fh1.close()
    #     框，类别，类别信息

    # print(gt_boxes,classes,num_pos)
    return gt_boxes, classes, num_pos


def getDetBoxes(cfg, DetFolder):
    files = os.listdir(DetFolder)
    files.sort()
    det_boxes = {}
    for f in files:
        nameOfImage = f.replace(".txt", "")
        fh1 = open(os.path.join(DetFolder, f), "r")

        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")

            cls = (splitLine[0])  # class
            left = float(splitLine[1])
            top = float(splitLine[2])
            right = float(splitLine[3])
            bottom = float(splitLine[4])
            score = float(splitLine[5])
            # 左，上，右，底，置信度得分，哪张图片
            one_box = [left, top, right, bottom, score, nameOfImage]

            if cls not in det_boxes:
                det_boxes[cls] = []
            det_boxes[cls].append(one_box)

        fh1.close()
    # print(det_boxes)
    return det_boxes


def detections(cfg,
               gtFolder,
               detFolder,
               savePath,
               show_process=True):

    gt_boxes, classes, num_pos = getGTBoxes(cfg, gtFolder)
    det_boxes = getDetBoxes(cfg, detFolder)

    evaluator = Evaluator()
    # 传入配置，类别

    return evaluator.GetPascalVOCMetrics(cfg, classes, gt_boxes, num_pos, det_boxes)


def plot_save_result(cfg, results, classes, savePath):
    plt.rcParams['savefig.dpi'] = 80
    plt.rcParams['figure.dpi'] = 130

    acc_AP = 0
    validClasses = 0
    fig_index = 0
    for cls_index, result in enumerate(results):
        if result is None:
            raise IOError('Error: Class %d could not be found.' % result)
        cls = result['class']
        precision = result['precision']
        recall = result['recall']
        average_precision = result['AP']
        acc_AP = acc_AP + average_precision
        mpre = result['interpolated precision']
        mrec = result['interpolated recall']
        npos = result['total positives']
        total_tp = result['total TP']
        total_fp = result['total FP']

        fig_index += 1

        plt.figure(fig_index)
        plt.plot(recall, precision, cfg['colors'][cls_index], label='Precision')
        plt.xlabel('recall')
        plt.ylabel('precision')
        ap_str = "{0:.2f}%".format(average_precision * 100)
        plt.title('Precision x Recall curve \nClass: %s, AP: %s' % (str(cls), ap_str))
        plt.legend(shadow=True)
        plt.grid()
        plt.savefig(os.path.join(savePath, cls + '.png'))
        plt.show()
        plt.pause(0.05)

    mAP = acc_AP / fig_index
    mAP_str = "{0:.2f}%".format(mAP * 100)
    print('mAP: %s' % mAP_str)

