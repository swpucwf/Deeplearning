import cv2

img = cv2.imread("./images/1.jpg")

# 更改大小
# img2 = cv2.resize(img,(300,300))

# 镜像
img2 = cv2.transpose(img)
# img2 = cv2.flip(img,cv2.IMREAD_GRAYSCALE)




cv2.imshow("--",img2)
cv2.waitKey(0)