import numpy as np
#
# a = np.array([[1.,2.,3.],[3.,4.,3.]],dtype=np.float32)
#
# print(a.ndim,a.shape,a.dtype,a.size)
#
# b = np.array(7)
# print(b)
#
# a = np.random.randn(2,3)
# print(a)
# b = np.random.uniform(0,1,(3,4))
# print(b)
#
#
# a = np.arange(0,12).reshape(3,2,2)
# print(a)
#
# import matplotlib.pyplot as plt
# x = np.linspace(0,2,10)
# print(x)
#
# y = np.sin(x)
#
# plt.plot(x,y)
#
# plt.show()
#
#
# a  = np.arange(0,12)
# print(a)
#
# print(a**2)
# print(a<6)
# print(a[a<6])
#
#
# a = np.arange(0,12)
# print(a.sum())
# print(a.mean())
#
#
# a = np.arange(0,12)
# print(a.sum())
# print(a.mean())
#
# a = np.arange(0,12).reshape(3,4)
# print(a)
# print(a.sum(axis=1))
#
#
# img = np.random.rand(2,64,64,3)
# img.mean(axis=3)
# img.mean(axis=0)
#
# print(img.argmax(axis=3).shape)
# print(img.argmax(axis=3))
#
# a = np.arange(0,12).reshape(3,4)
# print(a.shape)
# print(a[:,None,:].shape)
# print(a)
# print(a.flatten())
# print(a.reshape(-1))

a = np.array([[1,2,3],[3,4,5]])
b = np.array([[5, 6, 4], [7, 8, 4]])
# c = np.stack([a,b],axis=0)
# # print(c)
# print(c.shape)
# c = np.stack([a,b],axis=1)
# c = np.stack([a,b],axis=1)
# print(c)
# print(c.shape)
d = np.concatenate([a,b],axis=1)
print(d.shape)
print(d)

a = np.arange(10)

print(a[-2])
print(a[::2])
rgb = np.random.randn(2,3,64,64)
bgr = rgb[:,::-1,...]
# print(bgr.shape)
a = np.arange(10)
print(a)
print(a[:-7])
print(a[-1:-7:-1])
print(a[np.array([3,3,-3,8])])

a = np.arange(10)+1
print(a)
print(a[a>6])
print(np.where(a>6))

# if __name__ == '__main__':
#     x = np.array([[1,2,3,4,32],[4,312,3,1231,3]])
#     print(x.shape)
#     print(x[:,x.argmax(axis=1)])

