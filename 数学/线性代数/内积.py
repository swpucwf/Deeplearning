import numpy as np

# 内积 相当于一个向量在另外一个向量上的投影
if __name__ == '__main__':
    a = np.array([1,2],dtype=np.float32)
    b = np.array([3,4],dtype=np.float32)
    c = np.sum(a*b)
    print(c)
