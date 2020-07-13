from PIL import Image
import numpy as np

if __name__ == '__main__':
    img = Image.open("pic.jpg")
    img.show()
    img_data = np.array(img)
    # nd array数据
    print(img_data)
    print(img_data.shape)
    img_data = img_data.reshape(-1)
    print(img_data)
    print(img_data.shape)
    img_data = img_data.reshape(375,499,3)
    print(img_data.shape)
    img = Image.fromarray(img_data)
    img.show()
    