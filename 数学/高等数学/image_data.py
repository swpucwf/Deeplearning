from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open("pic.jpg")
r,g,b = img.split()
# 合并通道
img0 = Image.merge('RGB',(r.point(lambda i:i==i*0),
                          g.point(lambda i:i==i*0),b.point(lambda  i:i==i*0.5)))
img_data = np.array(img)
# print(img_data)
print(img_data.shape)
# img.show()
plt.imshow(img0)
plt.show()