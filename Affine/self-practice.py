import torch
import torch.nn as nn

class CompareNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)  
        self.fc2 = nn.Linear(5, 2)   

    def forward(self, x):
        mid_logits = self.fc1(x)  # 注意：这里的数值是随机的，且没有经过任何处理
        
        # 我们先分别保存两种激活后的结果，方便对比
        mid_relu = torch.relu(mid_logits)      # 这是你原来的
        mid_sigmoid = torch.sigmoid(mid_logits) # 这是你现在想看的
        
        # 各自走完后续流程
        final_logits_relu = self.fc2(mid_relu)
        final_logits_sigmoid = self.fc2(mid_sigmoid)
        
        return mid_logits, mid_relu, mid_sigmoid, final_logits_relu, final_logits_sigmoid

# 制造同样的输入（固定随机种子，让每次结果一样，方便观察）
torch.manual_seed(42) 
dummy_input = torch.randn(1, 10)

net = CompareNet()
mid_l, mid_r, mid_s, final_r, final_s = net(dummy_input)

print("=" * 50)
print("【原始 Logits（没激活前的裸分数）】")
print(mid_l)  # 注意看：这里有正数，也有负数
print("-" * 50)

print("【经过 ReLU 后】（负数全部变0，正数不变）")#神经元死亡
print(mid_r)
print("-" * 50)

print("【经过 Sigmoid 后】（所有数都被压缩到 0~1 之间）")
print(mid_s)
print("-" * 50)

print("【最终输出的 Logits 对比】")
print(f"ReLU分支进入最后一层的结果: {final_r}")
print(f"Sigmoid分支进入最后一层的结果: {final_s}")



