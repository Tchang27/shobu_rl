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
    

class MLP_Head(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, dropout_rate=0.3):
        super(MLP_Head, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    
class Shobu_PPO(nn.Module):
    def __init__(self, num_boards=1, board_size=8, dropout_rate=0.3):
        super(Shobu_PPO, self).__init__()
        
        # Input shape: (batch_size, num_boards, board_size, board_size)
        self.input_channels = num_boards
        
        self.conv1 = nn.Conv2d(self.input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        
        self.residual1 = ResidualBlock2D(16, 16)
        
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)
        
        self.residual2 = ResidualBlock2D(64, 64)
        
        self.conv3 = nn.Conv2d(64, 256, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        
        self.fc_input_size = self._calculate_fc_input_size(num_boards, board_size)
        
        # passive head
        self.passive_pos_head = MLP_Head(self.fc_input_size, 64)
        self.passive_dir_head = MLP_Head(self.fc_input_size, 8)
        self.passive_dist_head = MLP_Head(self.fc_input_size, 2)
        # aggressive head (conditioned on first move)
        self.aggressive_pos_head = MLP_Head(self.fc_input_size+74, 64)
        
        # critic head
        self.critic = MLP_Head(self.fc_input_size, 1) 
        

    def _calculate_fc_input_size(self, num_boards, board_size):
        x = torch.randn(1, num_boards, board_size, board_size)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.residual1(x)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.residual2(x)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        return x.view(1, -1).size(1)
    
    
    def get_features(self, x):
        # Input shape: (batch_size, num_boards, board_size, board_size)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.residual1(x)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.residual2(x)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        
        return x
    
    def get_policy(self, x):
        '''
        Get policy
        '''
        x = self.get_features(x)
        # passive moves
        passive_position_probs = self.passive_pos_head(x)
        passive_dir_probs = self.passive_dir_head(x)
        passive_dist_probs = self.passive_dist_head(x)
        
        # aggressive moves, condition on first move
        x = torch.concat((x,passive_position_probs,passive_dir_probs,passive_dist_probs), dim=1)
        aggressive_position_probs = self.aggressive_pos_head(x)
        
        return {
            "passive": {
                "position": passive_position_probs,
                "direction": passive_dir_probs,
                "distance": passive_dist_probs
            },
            "aggressive": {
                "position": aggressive_position_probs
            }
        }
    
    
    def value_function(self, x): 
        '''
        Get q value
        '''
        x = self.get_features(x)
        # critic
        q_value = self.critic(x)
        return q_value

    
    def forward(self, x):
        '''
        Get policy and q value
        '''
        x = self.get_features(x)
        
        # passive moves
        passive_position_probs = self.passive_pos_head(x)
        passive_dir_probs = self.passive_dir_head(x)
        passive_dist_probs = self.passive_dist_head(x)
        
        # aggressive moves, condition of passive moves
        x = torch.concat((x,passive_position_probs,passive_dir_probs,passive_dist_probs), dim=1)
        aggressive_position_probs = self.aggressive_pos_head(x)
        
        # critic
        q_value = self.critic(x)
        
        return {
            "passive": {
                "position": passive_position_probs,
                "direction": passive_dir_probs,
                "distance": passive_dist_probs
            },
            "aggressive": {
                "position": aggressive_position_probs
            },
            "q_value" : q_value
        }