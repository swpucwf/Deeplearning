from PIL import Image
from sklearn.datasets import load_sample_image
china = load_sample_image("flower.jpg")
print(china.dtype)
print(china.shape)

if __name__ == '__main__':
    img= Image.fromarray(china)
    img.show()