from PIL import Image
import numpy as np
if __name__ == '__main__':

    img = Image.open("pic.jpg")
    img_data = np.array(img)
    print(img_data.shape)
    img.show()