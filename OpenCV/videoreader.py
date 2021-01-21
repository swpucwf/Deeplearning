import cv2

cap = cv2.VideoCapture(0)

while True:
    # 判断ret是否读取成功
    ret,frame = cap.read()
    if ret:
        cv2.imshow("frame",frame)
        if cv2.waitKey(30) & 0xFF ==ord('q'):
            break
cap.release()
cv2.destroyAllWindows()