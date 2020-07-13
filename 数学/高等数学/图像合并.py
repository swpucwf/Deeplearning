from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    image = Image.open("pic.jpg")
    r,g,b= image.split()
    # 合并图像,通道操作
    img0 = Image.merge("RGB",
                       (r.point(lambda i:i==0),
                        g,
                        b.point(lambda i:i==0)))
    img_data = np.array(img0)
    print(img_data.shape)
    # resize 会改变图片尺寸
    # img0 = img0.resize((375,375))
    plt.imshow(img0)
    plt.show()