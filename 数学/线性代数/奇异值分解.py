import numpy as np

if __name__ == '__main__':
    data = np.array([[1,2],[3,4],[5,6]],dtype=np.float32)
    print(data.shape)
    (u,f,v) = np.linalg.svd(data)
    # 用户打分 ， 分数和种类表  书的偏向种类 svd是中间矩阵是直接给的最大特征
    print(u.shape,f.shape,v.shape)
