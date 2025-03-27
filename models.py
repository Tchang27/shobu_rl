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
    def __init__(self, in_channels, out_channels, hidden_channels=256, dropout_rate=0.3):
        super(MLP_Head, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x):
        return self.net(x)

    
class Critic(nn.Module):
    def __init__(self, in_channels, hidden_channels=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels//4),
            nn.ReLU(),
            nn.Linear(hidden_channels//4, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.net(x) 
    
    
class Shobu_PPO(nn.Module):
    def __init__(self, num_boards=1, board_size=8, dropout_rate=0.3):
        super(Shobu_PPO, self).__init__()
        
        # Input shape: (batch_size, num_boards, board_size, board_size)
        self.input_channels = num_boards
        
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(self.input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            #ResidualBlock2D(16, 16),

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #ResidualBlock2D(32, 32),

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_input_size = self._calculate_fc_input_size(num_boards, board_size)
        
        # passive head
        self.passive_head = MLP_Head(self.fc_input_size, 1024, hidden_channels=1024)
        # aggressive head (conditioned on first move)
        self.aggressive_pos_head = MLP_Head(self.fc_input_size+1024, 64, hidden_channels=1024)
        
        # critic head
        self.critic = Critic(self.fc_input_size) 
        

    def _calculate_fc_input_size(self, num_boards, board_size):
        x = torch.randn(1, num_boards, board_size, board_size)
        x = self.backbone(x)
        return x.view(1, -1).size(1)
    
    
    def get_features(self, x):
        # Input shape: (batch_size, num_boards, board_size, board_size)
        x = self.backbone(x)      
        return x.view(x.size(0), -1)
    
    
    def get_policy(self, x):
        '''
        Get policy
        '''
        x = self.get_features(x)
        
        # passive moves
        passive_probs = self.passive_head(x)
        
        # aggressive moves, condition on first move
        x = torch.concat((x,passive_probs), dim=1)
        aggressive_position_probs = self.aggressive_pos_head(x)
        
        return {
            "passive": {
                "position": passive_probs,
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
        passive_probs = self.passive_head(x)
        
        # aggressive moves, condition on first move
        x = torch.concat((x,passive_probs), dim=1)
        aggressive_position_probs = self.aggressive_pos_head(x)
        
        # critic
        q_value = self.critic(x)
        
        return {
            "passive": {
                "position": passive_probs,

            },
            "aggressive": {
                "position": aggressive_position_probs
            },
            "q_value" : q_value
        }