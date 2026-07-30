import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#生成数据
inputs=torch.rand(100,3)
weight=torch.tensor([[1.1],[2.2],[3.3]])
bias=torch.tensor(4.4)
targets=inputs@weight+bias+0.1*torch.randn(100,1)

#初始化参数
#需要对特征做归一化处理：（x-mean）/std
w=torch.rand((3,1),requires_grad=True,device=device)
b=torch.rand((1,),requires_grad=True,device=device)

#将其他参数也存放进cuda
inputs=inputs.to(device)
targets=targets.to(device)
#设置超参数
epoch=10000
lr=0.001

for i in range(epoch):
    outputs=inputs@w+b
    loss=torch.mean(torch.square(outputs-targets))
    print("loss:",loss.item())
    loss.backward()

    with torch.no_grad():
        w -= lr*w.grad
        b -= lr*b.grad

    w.grad.zero_()
    b.grad.zero_()

print("训练后权重w:",w)
print("训练后偏置b:",b)
