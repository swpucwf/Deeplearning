import numpy as np
a = np.array([[1,2],[3,4]],dtype=np.float32)
b = np.array([[3],[4]],dtype=np.float32)

print(a+b)
a = np.array([5])
print(np.tile(a,[2,3]))
b = np.array([5,6])
print(np.tile(b,[3,1]))
print(np.tile(b,[3]))