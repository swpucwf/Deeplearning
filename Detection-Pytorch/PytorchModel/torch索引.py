import torch

a = torch.Tensor([[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]]])



print(a.shape)
print(a[:,::2])
print(a>0)
print(a[0][1])
print(a>0)
print(a[a>2])

indexes =  torch.nonzero(a,as_tuple=True)
print(indexes)
exit()

print(indexes.shape)
print(indexes)
#
for index in indexes:
    # 获取到了索引
    print(a[index[0]][index[1]][index[2]])
print(torch.full_like(a,1))

print(a.shape)

i,j,k = torch.where(a>2)
print(a[i][j][k])
