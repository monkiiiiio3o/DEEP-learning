from torch.utils.data import Dataset,DataLoader,random_split
import torch
import torch.nn as nn
import pandas as pd

#读取&预处理数据
class TitanicNN(nn.Module):
    """增加了隐藏层的神经网络，比逻辑回归表达能力更强"""
    def __init__(self,input_dim):
        super().__init__()
        self.fc1=nn.Linear(input_dim,64)   # 第1层: 10 → 64
        self.relu=nn.ReLU()                # 激活函数: 引入非线性
        self.fc2=nn.Linear(64,32)          # 第2层: 64 → 32
        self.relu2=nn.ReLU()
        self.fc3=nn.Linear(32,1)           # 第3层: 32 → 1
        self.sigmoid=nn.Sigmoid()          # 输出概率

    def forward(self,x):
        x=self.relu(self.fc1(x))           # 10→64→激活
        x=self.relu2(self.fc2(x))          # 64→32→激活
        x=self.fc3(x)                      # 32→1
        return self.sigmoid(x)             # 压缩到(0,1)
    
class TitanicDataset(Dataset):
    def __init__(self,file_path):
        self.file_path=file_path
        self.mean={
            "Pclass":2.236695,
            "Age":29.699118,
            "SibSp":0.512605,
            "Parch":0.431373,
            "Fare":34.694514,
            "Sex_female":0.365546,
            "Sex_male":0.634454,
            "Embarked_C":0.182073,
            "Embarked_Q":0.039216,
            "Embarked_S":0.775910,
        }

        self.std={
            "Pclass":0.838250,
            "Age":14.526497,
            "SibSp":0.929783,
            "Parch":0.853289,
            "Fare":52.918930,
            "Sex_female":0.481921,
            "Sex_male":0.481921,
            "Embarked_C":0.386175,
            "Embarked_Q":0.194244,
            "Embarked_S":0.417274,
        }

        self.data=self._load_data()
        self.feature_size=len(self.data.columns)-1  # column → columns

    #数据清洗
    def _load_data(self):
        df=pd.read_csv(self.file_path)
        df=df.drop(columns=["PassengerId","Name","Ticket","Cabin"])
        df=df.dropna(subset=["Age"])#删除age有缺失的行
        df=pd.get_dummies(df,columns=["Sex","Embarked"],dtype=int)#进行独热编码
        # 标准化：让所有特征处于同一尺度，(x-均值)/标准差
        for col in self.mean:
            df[col]=(df[col]-self.mean[col])/self.std[col]
        return df  # ← 必须返回！不然self.data是None

    def __len__(self):
        return len(self.data)#了解数据个数以便于分batch
    
    def __getitem__(self,idx):#根据索引返回值
        features=self.data.drop(columns=["Survived"]).iloc[idx].values
        label=self.data["Survived"].iloc[idx]
        return torch.tensor(features,dtype=torch.float32),torch.tensor(label,dtype=torch.float32)

train_dataset=TitanicDataset(r"C:\Users\Monki\Desktop\deep-learing\titanic\train.csv")
validation_dataset=TitanicDataset(r"C:\Users\Monki\Desktop\deep-learing\titanic\validation.csv")

#将模型移动至gpu
model=TitanicNN(train_dataset.feature_size)
model.to("cuda")
model.train()

optimizer=torch.optim.Adam(model.parameters(),lr=0.001)  # Adam + 更低学习率

epochs=200
for epoch in range(epochs):
    correct=0
    total_loss=0#loss初始化
    step=0  # ← Bug4修复：step必须在这里初始化为0
    for features, labels in DataLoader(train_dataset,batch_size=256,shuffle=True):
        step +=1
        features=features.to("cuda")
        labels=labels.to("cuda")  # ← Bug5修复：labels也要移到GPU！
        optimizer.zero_grad()
        outputs=model(features).squeeze()
        correct+= torch.sum(((outputs>=0.5)==labels))
        loss=torch.nn.functional.binary_cross_entropy(outputs,labels)  # ← Bug3修复：拼写
        total_loss+=loss.item()
        loss.backward()
        optimizer.step()
    print(f'Epoch{epoch+1},Loss:{total_loss/step:.4f}')
    print(f'Training Accuracy:{correct/len(train_dataset)}')

#切换验证模式
model.eval()
with torch.no_grad():
    correct=0
    for features,labels in DataLoader(validation_dataset,batch_size=256):
        features=features.to("cuda")
        labels=labels.to("cuda")
        outputs=model(features).squeeze()
        correct+=torch.sum(((outputs>=0.5)==labels))
    print(f'Validation Accuracy:{correct/len(validation_dataset)}')