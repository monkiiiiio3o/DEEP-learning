#准备数据：验证图片
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import torch
from torchvision import transforms
import torch.nn as nn
import os

def verify_images(image_folder):
    classes=["Cat","Dog"]
    class_to_idx={"Cat":0,"Dog":1}
    samples=[]
    for cls_name in classes:
        cls_dir=os.path.join(image_folder,cls_name)
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith(('.jpg','.jpeg','.png')):#不是以图片格式结尾的
                continue#跳过
            path=os.path.join(cls_dir,fname)
            try:
                with Image.open(path)as img:
                    #尝试读取图片
                    img.verify()#验证图片
                samples.append((path,class_to_idx[cls_name]))#存入图片位置及对应类别
            except Exception:
                print(f'Warning:Skipping corrupted image{path}')
    return samples#返回处理好的列表


#数据处理
class ImageDataset(Dataset):
    def __init__(self,samples,transform=None):
        self.samples=samples
        self.transform=transform#对对象的一系列操作

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):#读取图片路径和label
        path,label=self.samples[idx]
        with Image.open(path) as img:
            img=img.convert('RGB')#转化为rgb模式
            if self.transform:
                img=self.transform(img)
        return img,label

#定义卷积神经网络
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            #省去全连接层
            nn.Conv2d(in_channels=128,out_channels=1,kernel_size=1),#1*1卷积
            nn.AdaptiveAvgPool2d((1,1)),#全局平均池化
            nn.Flatten(),
            nn.Sigmoid()#二分类
        )

    def forward(self,x):
        return self.model(x)

#验证集检验正确率
def evaluate(model,test_dataloader):
    model.eval()
    val_correct=0
    val_total=0

    with torch.no_grad():
        for inputs,labels in test_dataloader:
            inputs=inputs.to(DEVICE)
            labels=labels.float().unsqueeze(1).to(DEVICE)

            outputs=model(inputs)
            preds=(outputs>0.5).float()#将大于0.5的预测值转化为1
            val_correct += (preds == labels).sum().item()#和labels进行对比
            val_total+=labels.size(0)

    val_acc=val_correct/val_total
    return val_acc#得到准确率


#读取和划分数据
if __name__=="__main__":
    DATA_DIR=r"C:\Users\Monki\Desktop\deep-learing\kagglecatsanddogs_3367a\PetImages"
    BATCH_SIZE=64
    IMG_SIZE=128
    EPOCHS=20
    LR=0.001
    PRINT_STEP=100

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_samples=verify_images(DATA_DIR)
    random.seed(42)
    random.shuffle(all_samples)
    train_size=int(len(all_samples)*0.8)
    train_samples=all_samples[:train_size]#训练集
    valid_samples=all_samples[train_size:]#验证集
    #定义对数据进行一系列操作的transform
    train_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomCrop(size=(IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.15,
        saturation=0.1,
        hue=0.1
    ),
    transforms.RandomRotation(degrees=30),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    valid_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset=ImageDataset(train_samples,train_transform)
    valid_dataset=ImageDataset(valid_samples,valid_transform)



    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model=CNNModel().to(DEVICE)#模型
    criterion=nn.BCELoss()#loss
    optimizer=torch.optim.Adam(model.parameters(),lr=LR)#优化器

#训练循环
    for epoch in range(EPOCHS):
        print(f"\nEpoch{epoch+1}/{EPOCHS}")
        model.train()
        running_loss=0.0
        for step, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            optimizer.zero_grad()#梯度清零
            outputs=model(inputs)
            loss=criterion(outputs,labels)#计算loss
            loss.backward()#反向传播
            optimizer.step()#更新参数

            running_loss+=loss.item()

            if (step+1)%PRINT_STEP==0:
                avg_loss=running_loss/PRINT_STEP
                print(f"step[{step+1}-Loss:{avg_loss:.4f}]")
                running_loss=0.0

    val_acc=evaluate(model,valid_dataloader)
    print(f"Validation Accuracy after epoch{epoch+1}:{val_acc:.4f}")#监控模型训练状态
                        





    

