import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection if channel dimensions change
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class ShobuResidualCNN(nn.Module):
    def __init__(self, num_boards=4, board_size=4, action_size=512, dropout_rate=0.3):
        super(ShobuResidualCNN, self).__init__()
        
        # Input shape: (batch_size, num_boards, board_size, board_size)
        # We'll treat each board as a separate channel initially
        self.input_channels = num_boards
        
        # Initial convolution
        self.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces 4x4 to 2x2
        
        # Residual blocks
        self.residual1 = ResidualBlock2D(64, 64)
        
        # Second convolution
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces 2x2 to 1x1
        
        self.residual2 = ResidualBlock2D(128, 128)
        
        # Final convolution
        self.conv3 = nn.Conv2d(128, 256, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        
        # Calculate FC input size dynamically
        self.fc_input_size = self._calculate_fc_input_size(num_boards, board_size)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_size)
        self.dropout = nn.Dropout(p=dropout_rate)

    def _calculate_fc_input_size(self, num_boards, board_size):
        # Create a dummy input to calculate the flattened size
        x = torch.randn(1, num_boards, board_size, board_size)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.residual1(x)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.residual2(x)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        return x.view(1, -1).size(1)

    def forward(self, x):
        # Input shape: (batch_size, num_boards, board_size, board_size)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.residual1(x)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.residual2(x)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten for FC layers
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        q_values = self.fc3(x)
        
        return q_values