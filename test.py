import torch
import torch.nn.functional as F

a=torch.tensor([[[1,2,3],
               [3,4,3]],
               [[1, 2,3],
                [3, 4,3]]],dtype=torch.float32)
b=torch.tensor([[[1,2,3],[3,4,3]],[[1,2,3],[3,3,3]]],dtype=torch.float32)

b=b.transpose(1,2)

x=torch.matmul(a,b)


x1=torch.softmax(x,dim=1)#dim=1表示对第二维度进行softmax，行，row
x2=torch.softmax(x,dim=2)#dim=2表示对第三维度进行softmax，列，cloumn

x3=F.avg_pool1d(x,x.size(2))


a1=torch.tensor([[[1,2]],[[2,2]],[[3,2]]])
a2=torch.tensor([[[1,2]],[[2,2]],[[3,2]]])
a3=torch.tensor([[[1,2]],[[2,2]],[[3,2]]])
a4=torch.tensor([[[1,2]],[[2,2]],[[3,2]]])
print(a1.shape)

m=torch.cat((a1,a2,a3,a4),dim=2)
print(a.shape)
print(m.shape)


