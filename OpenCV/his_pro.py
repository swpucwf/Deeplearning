import cv2
import numpy as np

if __name__ == '__main__':
    # 直方图反投影 需要找出来的图片
    roi = cv2.imread("./images/10.jpg")
    hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    # cv2.imshow("--",hsv)
    # cv2.waitKey(0)
    target = cv2.imread("./images/9.jpg")
    hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)
    # cv2.imshow("--",hsvt)
    # cv2.waitKey(0)
    #
    roihist = cv2.calcHist([hsv],[0,1],None,[180,256],[0,180,0,256])
    #
    cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
    # 反投影
    dst = cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)
    # cv2.imshow("2",dst)
    # cv2.waitKey(0)
    # exit()
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dst = cv2.filter2D(dst,-1,disc)
    # cv2.imshow("2",dst)
    # cv2.waitKey(0)
    # exit()
    ret,thresh = cv2.threshold(dst,20,255,0)
    # cv2.imshow("2", ret)
    # cv2.waitKey(0)
    # 合并
    thresh = cv2.merge((thresh,thresh,thresh))
    res = cv2.bitwise_and(target,thresh)
    # 合并
    res = np.hstack((target,thresh,res))
    cv2.imshow("img",res)
    cv2.waitKey(0)