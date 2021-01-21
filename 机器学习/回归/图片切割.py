import numpy as np
from PIL import Image

img = Image.open("1.jpg")
# print(img.size)
# exit()
# img_data = np.array(img)
# print(img_data.shape)
img = img.resize((240,150))
img_data = np.array(img)
print(img_data.shape)
# img.show()
img_data = img_data.reshape(2,75,2,120,3)
img_data = img_data.transpose(0,2,1,3,4)
# print(img_data.shape)
img_data = img_data.reshape(-1,75,120,3)
print(img_data.shape)
imgs = np.split(img_data,4,axis=0)
for i,img_d in enumerate(imgs):
    # print(img_d.shape)
    img = Image.fromarray(img_d[0])
    img.show()
    img.save(f"s{i}.jpg")