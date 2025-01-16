import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models import CNN_Encoder  # attention here
from WHU_RS19_Dataset import WHU_RS19_Dataset

# 超参数设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置训练参数
batch_size = 32
learning_rate = 1e-4
epochs = 20
num_classes = 19  # 数据集中有19个类别
encoder_dim = 768  # ViT 输出维度
dropout = 0.5

# 数据变换：图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载数据集
train_data = WHU_RS19_Dataset(root_dir='/home/xwc/PSNet/WHU_RS19/WHU-RS19/RSDataset', transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# 使用CNN_Encoder提取图像特征（使用ViT）
encoder = CNN_Encoder('vit_b_16', encoded_image_size=14).to(device)

# 修改DecoderWithAttention以适应分类任务
class ClassifierWithAttention(nn.Module):
    def __init__(self, encoder, num_classes, dropout=0.5):
        super(ClassifierWithAttention, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(encoder_dim, num_classes)  # 确保全连接层输入是encoder_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, images):
        # 获取图像特征
        encoder_out = self.encoder(images)  # 提取图像特征

        # 打印 encoder_out 的形状（仅用于调试）
        # print(encoder_out.shape)  # 输出应该是 [batch_size, encoder_dim, height, width]，如 [32, 768, 14, 14]

        # 对 encoder_out 进行全局平均池化，将其转换为 [batch_size, encoder_dim]
        # 使用 torch.mean 来对 height 和 width 维度进行池化
        encoder_out = torch.mean(encoder_out, dim=[2, 3])  # 在 height 和 width 维度上求均值

        # 确保 encoder_out 的形状为 [batch_size, encoder_dim]
        assert encoder_out.dim() == 2, f"Expected encoder_out shape to be [batch_size, encoder_dim], but got {encoder_out.shape}"

        # 通过全连接层进行分类
        out = self.fc(self.dropout(encoder_out))  # 全连接层分类
        return out

# 创建分类模型
model = ClassifierWithAttention(encoder, num_classes=num_classes, dropout=dropout).to(device)

# 优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# 模型训练
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 打印训练损失
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    # 每个 epoch 保存一次模型
    if (epoch + 1) % 5 == 0:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f'checkpoint_{epoch+1}.pth')

# 完成训练后保存最终模型
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': running_loss / len(train_loader),
}, 'final_model.pth')

print("Training complete. Model saved as final_model.pth.")
