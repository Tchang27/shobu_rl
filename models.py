import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, bias=False, padding='same')
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, bias=False, padding='same')
        self.b1 = nn.BatchNorm2d(out_channels)
        self.b2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.b1(self.conv1(x)))
        out = self.b2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out
    
    
class KataGoGlobalPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: [batch, channels, H, W]
        B, C, H, W = x.shape
        
        #Mean per channel
        mean = x.mean(dim=[2, 3])

        #Max per channel
        max_val, _ = x.view(B, C, -1).max(dim=2)  # [B, C]

        # Concatenate all
        pooled = torch.cat([mean, max_val], dim=1)  # [B, 2C]
        return pooled

    
class MLP_Head(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=512):
        super(MLP_Head, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x):
        return self.net(x)

    
class Critic(nn.Module):
    def __init__(self, in_channels, hidden_channels=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding='same'),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )
        
    
    def forward(self, x):
        return self.net(x) 
    

class Critic_MCTS(nn.Module):
    def __init__(self, in_channels=64, hidden_channels=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=1, stride=1, bias=True, padding='same'), # b x 32 x 4 x 4
            KataGoGlobalPooling(), # 2 x 32
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Tanh()
        )
        

    def forward(self, x):
        return self.net(x) 
    

class Policy_Head(nn.Module):
    def __init__(self, in_channels=96, out_channels=68, hidden_channels=256):
        super(Policy_Head, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False, padding='same'),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding='same'),
        )

    def forward(self, x):
        return self.net(x)
    
    
class Shobu_PPO(nn.Module):
    def __init__(self, device, num_boards=64, board_size=4):
        super(Shobu_PPO, self).__init__()
        
        # Input shape: (batch_size, num_boards, board_size, board_size)
        self.input_channels = num_boards
        self.device = device
        
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(self.input_channels, 256, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            
            # Residual blocks
            ResidualBlock2D(256, 256),
            ResidualBlock2D(256, 256),
            ResidualBlock2D(256, 256),
            ResidualBlock2D(256, 256),
            ResidualBlock2D(256, 256),
            ResidualBlock2D(256, 256),
            ResidualBlock2D(256, 256),
            ResidualBlock2D(256, 256),
            ResidualBlock2D(256, 256),
            ResidualBlock2D(256, 256),
            ResidualBlock2D(256, 256),
            ResidualBlock2D(256, 256),
            ResidualBlock2D(256, 256),
            ResidualBlock2D(256, 256),
            ResidualBlock2D(256, 256),
            ResidualBlock2D(256, 256),
        )
        
        # passive head
        self.passive_filter = nn.Sequential(
            # Block 1
            nn.Conv2d(256, 2, kernel_size=1, stride=1, padding='same'),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_input_size = self._calculate_fc_input_size(num_boards, board_size)
        
        self.passive_pos_head = MLP_Head(self.fc_input_size, 64)
        self.passive_dir_head = MLP_Head(self.fc_input_size+64, 8)
        self.passive_dist_head = MLP_Head(self.fc_input_size+72, 2)
        # aggressive head (conditioned on first move)
        self.aggressive_pos_head = MLP_Head(self.fc_input_size+74, 64)
        
        # critic head
        self.critic = Critic(self.fc_input_size//2, hidden_channels=256) 
        

    def _calculate_fc_input_size(self, num_boards, board_size):
        x = torch.randn(1, num_boards, board_size, board_size)
        x = self.backbone(x)
        x = self.passive_filter(x)
        return x.view(1, -1).size(1)
    
    
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
        x = torch.concat((x,passive_position_logits), dim=1)
        passive_dir_logits = self.passive_dir_head(x)
        x = torch.concat((x,passive_dir_logits), dim=1)
        passive_dist_logits = self.passive_dist_head(x)
        
        # aggressive moves, condition on first move
        x = torch.concat((x,passive_dist_logits), dim=1)
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
        feat = self.get_features(x)
        x = self.passive_filter(feat)
        
        # passive moves
        passive_position_logits = self.passive_pos_head(x)
        x = torch.concat((x,passive_position_logits), dim=1)
        passive_dir_logits = self.passive_dir_head(x)
        x = torch.concat((x,passive_dir_logits), dim=1)
        passive_dist_logits = self.passive_dist_head(x)
        
        # aggressive moves, condition on first move
        x = torch.concat((x,passive_dist_logits), dim=1)
        aggressive_position_logits = self.aggressive_pos_head(x)
        
        
        # critic
        q_value = self.critic(feat)
        
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

HISTORY_SIZE = 1

class Shobu_MCTS(nn.Module):
    def __init__(self, device, num_boards=HISTORY_SIZE*8, board_size=4):
        super(Shobu_MCTS, self).__init__()
        
        # Input shape: (batch_size, num_boards, board_size, board_size)
        self.input_channels = num_boards
        self.device = device
        
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(self.input_channels, 96, kernel_size=3, stride=1, bias=False, padding='same'),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            
            # Residual blocks
            ResidualBlock2D(96, 96),
            ResidualBlock2D(96, 96),
            ResidualBlock2D(96, 96),
            ResidualBlock2D(96, 96),
            ResidualBlock2D(96, 96),
            ResidualBlock2D(96, 96),
        )
        
        # passive head
        self.passive_filter = nn.Sequential(
            # Block 1
            nn.Conv2d(96, 32, kernel_size=1, stride=1, bias=False, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_input_size = self._calculate_fc_input_size(num_boards, board_size)
        
        self.passive_pos_head = MLP_Head(self.fc_input_size, 64, hidden_channels=256)
        self.passive_dir_head = MLP_Head(self.fc_input_size+64, 8, hidden_channels=256)
        self.passive_dist_head = MLP_Head(self.fc_input_size+72, 2, hidden_channels=256)
        # aggressive head (conditioned on first move)
        self.aggressive_pos_head = MLP_Head(self.fc_input_size+74, 64, hidden_channels=256)
        
        # critic head
        self.critic = Critic_MCTS(hidden_channels=48)
        

    def _calculate_fc_input_size(self, num_boards, board_size):
        x = torch.randn(1, num_boards, board_size, board_size)
        x = self.backbone(x)
        x = self.passive_filter(x)
        return x.view(1, -1).size(1)
    
    
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
        x = torch.concat((x,passive_position_logits), dim=1)
        passive_dir_logits = self.passive_dir_head(x)
        x = torch.concat((x,passive_dir_logits), dim=1)
        passive_dist_logits = self.passive_dist_head(x)
        
        # aggressive moves, condition on first move
        x = torch.concat((x,passive_dist_logits), dim=1)
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
    
    
    def get_value(self, x): 
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
        feat = self.get_features(x)
        x = self.passive_filter(feat)
        
        # passive moves
        passive_position_logits = self.passive_pos_head(x)
        x = torch.concat((x,passive_position_logits), dim=1)
        passive_dir_logits = self.passive_dir_head(x)
        x = torch.concat((x,passive_dir_logits), dim=1)
        passive_dist_logits = self.passive_dist_head(x)
        
        # aggressive moves, condition on first move
        x = torch.concat((x,passive_dist_logits), dim=1)
        aggressive_position_logits = self.aggressive_pos_head(x)
        
        
        # critic
        q_value = self.critic(feat)
        
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
    
    
class Shobu_MCTS_Conv(nn.Module):
    def __init__(self, device, num_boards=HISTORY_SIZE*8, board_size=4):
        super(Shobu_MCTS_Conv, self).__init__()
        
        # Input shape: (batch_size, num_boards, board_size, board_size)
        self.input_channels = num_boards
        self.device = device
        
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(self.input_channels, 96, kernel_size=3, stride=1, bias=False, padding='same'),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            
            # Residual blocks
            ResidualBlock2D(96, 96),
            ResidualBlock2D(96, 96),
            ResidualBlock2D(96, 96),
            ResidualBlock2D(96, 96),
            ResidualBlock2D(96, 96),
            ResidualBlock2D(96, 96),
        )
        
        # policy head
        self.policy = Policy_Head()
        
        # critic head
        self.critic = Critic_MCTS(hidden_channels=48)
    
    
    def get_features(self, x):
        # Input shape: (batch_size, num_boards, board_size, board_size)
        x = self.backbone(x)      
        return x
    
    
    def get_policy(self, x):
        '''
        Get policy
        '''
        x = self.get_features(x)
        x = self.policy(x)
        
        # separate passive and aggressive
        passive = x[:, :64, :, :].view(x.size(0), -1)
        aggressive = x[:, 64:, :, :].view(x.size(0), -1)
        
        
        return {
            "passive": passive,
            "aggressive": aggressive
        }
    
    
    def get_value(self, x): 
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
        feat = self.get_features(x)
        
        # policy
        x = self.policy(feat)
        passive = x[:, :64, :, :].view(x.size(0), -1)
        aggressive = x[:, 64:, :, :].view(x.size(0), -1)
        
        # critic
        q_value = self.critic(feat)
        
        return {
            "passive": passive,
            "aggressive": aggressive,
            "q_value" : q_value
        }