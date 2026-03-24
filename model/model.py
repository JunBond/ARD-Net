import torch
from torch import nn
from torchsummary import summary

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, use_shortcut=False):
        super(ResidualBlock1D, self).__init__()
        # 使用 padding='same' 保证卷积后长度不变
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        # 如果通道数发生变化，使用 1x1 卷积对齐残差路径的通道数
        if use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = None

    def forward(self, x):
        identity = x
        
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        
        if self.shortcut is not None:
            identity = self.shortcut(identity)
            
        y += identity
        y = self.relu(y)
        return y

class ResNet1DCNN(nn.Module):
    def __init__(self, input_channels=1, input_len=8, output_dim=83):
        super(ResNet1DCNN, self).__init__()
        
        self.head = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.layer1 = ResidualBlock1D(64, 64, use_shortcut=False)
        self.layer2 = ResidualBlock1D(64, 128, use_shortcut=True)
        self.layer3 = ResidualBlock1D(128, 256, use_shortcut=True)

        self.flatten = nn.Flatten()
        flatten_dim = 256 * input_len 

        self.fc_out = nn.Sequential(
            nn.Linear(flatten_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.head(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(x)
        x = self.fc_out(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet1DCNN(input_channels=1, input_len=8, output_dim=83).to(device)
    
    summary(model, (1, 8))