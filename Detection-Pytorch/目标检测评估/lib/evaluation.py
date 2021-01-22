import os
import sys
import yaml
# 加入当前目录
# sys.path.insert(0, os.path.join(os.getcwd(), 'lib'))
from detection import detections, plot_save_result

class ComputationMap(object):

    def __init__(self,conf_path = '../conf/conf.yaml',
                 gtFolder='../data/groundtruths',
                 detFolder='../data/detections',
                 savePath='../data/results',
                 ):

        # 配置目录
        self.conf_path = conf_path
        # 读取配置
        with open(self.conf_path, 'r', encoding='utf-8') as f:
            data=f.read()
        # 加载相关配置,字典型
        self.cfg = yaml.load(data,Loader=yaml.FullLoader)

        # 真实标签位置，类别,四个坐标
        self.gtFolder =gtFolder
        # 侦测目标目录 类别，四个坐标，置信度
        self.detFolder = detFolder
        # 存储的目录
        self.savePath = savePath
        # 结果
        results, classes = detections(self.cfg, self.gtFolder, self.detFolder, self.savePath)
        # print(results)
        # 成图
        plot_save_result(self.cfg, results, classes, self.savePath)

if __name__ == '__main__':
    comp = ComputationMap()
