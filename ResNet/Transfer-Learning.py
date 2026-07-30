import torchvision.models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import torch
from torchvision import transforms, models
import torch.nn as nn
import os

torchvision.models.resnet18()

#将图片格式的数据保留
def verify_images(image_folder):
    classes = ["Cat", "Dog"]
    class_to_idx = {"Cat": 0, "Dog": 1}
    samples = []
    for cls_name in classes:
        cls_dir = os.path.join(image_folder, cls_name)
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            path = os.path.join(cls_dir, fname)
            try:
                with Image.open(path) as img:
                    img.verify()
                samples.append((path, class_to_idx[cls_name]))
            except Exception:
                print(f"Warning: Skipping corrupted image {path}")
    return samples

#加载数据
class ImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert('RGB')
            if self.transform:
                img = self.transform(img)
        return img, label

def evaluate(model, test_dataloader):
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            outputs = torch.sigmoid(model(inputs))
            preds = (outputs > 0.5).float()
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    return val_acc


if __name__ == "__main__":
    DATA_DIR = r"C:\Users\Monki\Desktop\deep-learing\kagglecatsanddogs_3367a\PetImages"
    BATCH_SIZE = 64
    IMG_SIZE = 128
    EPOCHS = 20
    LR = 0.001
    PRINT_STEP = 100

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #划分训练集和测试集
    all_samples = verify_images(DATA_DIR)
    random.seed(42)
    random.shuffle(all_samples)
    train_size = int(len(all_samples) * 0.8)
    train_samples = all_samples[:train_size]
    valid_samples = all_samples[train_size:]

    #对图片进行图像增强，增加样本丰富度
    train_transform = transforms.Compose([
        transforms.Resize((150, 150)),#统一大小
        transforms.RandomCrop(size=(IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            # saturation=0.2,
            # hue=0.1
        ),
        #transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#标准化
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = ImageDataset(train_samples, train_transform)
    valid_dataset = ImageDataset(valid_samples, valid_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    #迁移学习：1）加载预训练模型（保留通用特征），冻结所有层。
    model = models.resnet18(pretrained=True)  # 加载 ImageNet 预训练权重
    for param in model.parameters():
        param.requires_grad = False  # 冻结所有层
    #2）更新分类头（因为分类需求不同）
    # 解冻第四阶段的参数（最后一个残差模块）
    for param in model.layer4.parameters():
        param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)  # 二分类
    model = model.to(DEVICE)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    #训练循环
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        model.train()
        running_loss = 0.0

        for step, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            optimizer.zero_grad()
            outputs = torch.sigmoid(model(inputs))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (step + 1) % PRINT_STEP == 0:
                avg_loss = running_loss / PRINT_STEP
                print(f"  Step [{step + 1}] - Loss: {avg_loss:.4f}")
                running_loss = 0.0

        val_acc = evaluate(model, valid_dataloader)
        print(f"Validation Accuracy after epoch {epoch + 1}: {val_acc:.4f}")