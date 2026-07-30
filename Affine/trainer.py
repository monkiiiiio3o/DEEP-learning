import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# 自定义数据集
#加载数据：dataloader
#读取数据&预处理:dataset
class MNISTDataset(Dataset):
    def __init__(self, file_path):
        self.images, self.labels = self._read_file(file_path)

    def _read_file(self, file_path):
        images = []
        labels = []
        with open(file_path, 'r') as f:
            next(f)  # 跳过标题行
            for line in f:
                items = line.strip().split(",")
                images.append([float(x) for x in items[1:]])
                labels.append(int(items[0]))
        return images, labels

    def __getitem__(self, index):#根据索引返回具体样本
        image = torch.tensor(self.images[index], dtype=torch.float32).view(-1)#将张量拉成一维
        image = image / 255.0  # 归一化:将数据调整至[-1,1]or[0,1]
        image = (image - 0.1307) / 0.3081  # 标准化:(x-mean)/std,注意在分训练集和测试集之后进行（保证测试集的纯净,防止数据泄露）
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image, label

    def __len__(self):#返回数据集样本数目
        return len(self.images)


# 模型定义
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(#把模块串起来
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)


# 参数设置
batch_size = 64
learning_rate = 0.1
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据加载
train_dataset = MNISTDataset(r'C:\Users\Monki\Desktop\deep-learing\mnist\mnist_train.csv\mnist_train.csv')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = MNISTDataset(r"C:\Users\Monki\Desktop\deep-learing\mnist\mnist_test.csv\mnist_test.csv")
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 模型、损失函数、优化器
model = NeuralNetwork().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 训练过程
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)#计算每个batch的平均loss值
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# 测试过程
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")