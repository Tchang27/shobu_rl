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
    def __init__(self, in_channels, out_channels, hidden_channels=128, dropout_rate=0.3):
        super(MLP_Head, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x):
        return self.net(x)

    
class Critic(nn.Module):
    def __init__(self, in_channels, hidden_channels=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )
        
    
    def forward(self, x):
        return self.net(x) 
    
    
class Shobu_PPO(nn.Module):
    def __init__(self, device, num_boards=1, board_size=8, dropout_rate=0.3):
        super(Shobu_PPO, self).__init__()
        
        # Input shape: (batch_size, num_boards, board_size, board_size)
        self.input_channels = num_boards
        self.device = device
        
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(self.input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Block 2
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Residual blocks
            ResidualBlock2D(256, 256),
            ResidualBlock2D(256, 256),
            #ResidualBlock2D(256, 256),
            #ResidualBlock2D(256, 256),
            #ResidualBlock2D(256, 256),
            #ResidualBlock2D(256, 256),
        )
        
        # passive head
        self.passive_filter = nn.Sequential(
            nn.Conv2d(256, 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_input_size = self._calculate_fc_input_size(num_boards, board_size)
        self.passive_pos_head = MLP_Head(self.fc_input_size, 64)
        self.passive_dir_head = MLP_Head(self.fc_input_size+2, 8)
        self.passive_dist_head = MLP_Head(self.fc_input_size+10, 2)
        # aggressive head (conditioned on first move)
        self.aggressive_pos_head = MLP_Head(self.fc_input_size+12, 64, hidden_channels=128)
        
        # critic head
        self.critic = Critic(self.fc_input_size//2) 
        

    def _calculate_fc_input_size(self, num_boards, board_size):
        x = torch.randn(1, num_boards, board_size, board_size)
        x = self.backbone(x)
        x = self.passive_filter(x)
        return x.view(1, -1).size(1)
    
    
    def get_positional_encoding(self, piece_probs):
        coords = torch.stack(torch.meshgrid(torch.arange(8, device=self.device), torch.arange(8, device=self.device)), dim=-1)
        coords = coords.float().view(64, 2)

        # Normalize coordinates to [0, 1]
        coords = coords / 7.0

        # Weight by piece_probs and return
        return torch.matmul(piece_probs, coords)
    
    
    def get_features(self, x):
        # Input shape: (batch_size, num_boards, board_size, board_size)
        x = self.backbone(x)      
        return x
    
    
    def get_policy(self, x):
        '''
        Get policy
        '''
        x = self.get_features(x)
        x = self.passive_filter(x)
        
        # passive moves
        passive_position_logits = self.passive_pos_head(x)
        piece_probs = torch.softmax(passive_position_logits, dim=-1)
        piece_pos_encoding = self.get_positional_encoding(piece_probs)
        x = torch.concat((x,piece_pos_encoding), dim=1)
        passive_dir_logits = self.passive_dir_head(x)
        dir_onehot = F.one_hot(torch.argmax(passive_dir_logits, dim=-1), num_classes=8).float()
        x = torch.concat((x,dir_onehot), dim=1)
        passive_dist_logits = self.passive_dist_head(x)
        
        # aggressive moves, condition on first move
        dist_onehot = F.one_hot(torch.argmax(passive_dist_logits, dim=-1), num_classes=2).float()
        x = torch.concat((x,dist_onehot), dim=1)
        aggressive_position_logits = self.aggressive_pos_head(x)
        
        return {
            "passive": {
                "position": passive_position_logits,
                "direction": passive_dir_logits,
                "distance": passive_dist_logits,
            },
            "aggressive": {
                "position": aggressive_position_logits
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
        passive_position_logits = self.passive_pos_head(x)
        piece_probs = torch.softmax(passive_position_logits, dim=-1)
        piece_pos_encoding = self.get_positional_encoding(piece_probs)
        x = torch.concat((x,piece_pos_encoding), dim=1)
        passive_dir_logits = self.passive_dir_head(x)
        dir_onehot = F.one_hot(torch.argmax(passive_dir_logits, dim=-1), num_classes=8).float()
        x = torch.concat((x,dir_onehot), dim=1)
        passive_dist_logits = self.passive_dist_head(x)
        
        # aggressive moves, condition on first move
        dist_onehot = F.one_hot(torch.argmax(passive_dist_logits, dim=-1), num_classes=2).float()
        x = torch.concat((x,dist_onehot), dim=1)
        aggressive_position_logits = self.aggressive_pos_head(x)
        
        
        # critic
        q_value = self.critic(x)
        
        return {
            "passive": {
                "position": passive_position_logits,
                "direction": passive_dir_logits,
                "distance": passive_dist_logits,
            },
            "aggressive": {
                "position": aggressive_position_logits
            },
            "q_value" : q_value
        }