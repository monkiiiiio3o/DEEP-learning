import torch
from torch.utils.tensorboard import SummaryWriter

# 确保cuda可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#生成数据
inputs=torch.rand(100,3)#随机生成shape为（100,3）[100条数据，3个feature]的tensor，里面的每个元素在0-1之间
weight=torch.tensor([[1.1],[2.2],[3.3]])
bias=torch.tensor(4.4)
targets=inputs@weight+bias+0.1*torch.randn(100,1)#加入随机误差，模拟真实情况

#创建一个SummaryWriter，记录文件位置
writer = SummaryWriter(log_dir=r"C:\Users\Monki\Desktop\deep-learing")

#初始化线性回归参数并放在cuda上
w=torch.rand((3,1),requires_grad=True,device=device)
b=torch.rand((1,),requires_grad=True,device=device)

#将数据都移到cuda上
inputs=inputs.to(device)
targets=targets.to(device)
#设置超参数
epoch=10000
lr=0.003

for i in range(epoch):
    outputs=inputs@w+b#预测值
    loss=torch.mean(torch.square(outputs-targets))
    print("loss:",loss.item())
    #记录loss，三个参数分别为：tag，loss，第几步
    writer.add_scalar("loss/train",loss.item(),i)
    loss.backward()
    #梯度下降
    with torch.no_grad():
        w -=lr*w.grad
        b -=lr*b.grad
    
    #清零梯度
    w.grad.zero_()
    b.grad.zero_()

print("训练后的权重w:",w)
print("训练后的偏置b:",b)