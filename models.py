import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalPooling_BiasStructure(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GlobalPooling_BiasStructure, self).__init__()
        self.b1 = nn.BatchNorm2d(in_channels)
        self.pool = KataGoGlobalPooling()
        self.linear = nn.Linear(in_channels*2, out_channels)    

    def forward(self, p, g):
        out = F.relu(self.b1(g))
        out = self.pool(g)
        out = self.linear(out)
        out = out.unsqueeze(-1).unsqueeze(-1)
        return p+out


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
    
    
class ResidualBlock2D_PPO(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock2D_PPO, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same')

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        out = F.relu(out)
        return out

    
class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3):
        super(ResidualBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=1, bias=False, padding='same')
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel, stride=1, bias=False, padding='same')
        self.b1 = nn.BatchNorm2d(out_channels)
        self.b2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.b1(self.conv1(x)))
        out = self.b2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out
    

class ResidualBlock2D_Pooling(nn.Module):
    def __init__(self, in_channels, out_channels, pool_channels = 32, kernel=3):
        super(ResidualBlock2D_Pooling, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=1, bias=False, padding='same')
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel, stride=1, bias=False, padding='same')
        self.pool = GlobalPooling_BiasStructure(in_channels=pool_channels, out_channels=out_channels-pool_channels)
        self.b1 = nn.BatchNorm2d(out_channels)
        self.b2 = nn.BatchNorm2d(out_channels)
        self.pool_channels = pool_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        pool = self.pool(out[:, self.pool_channels:, :, :], out[:, :self.pool_channels, :, :])
        pooled = torch.concat([out[:, :self.pool_channels, :, :], pool], dim=1)
        out2 = F.relu(self.b1(pooled))
        out2 = self.b2(self.conv2(out2))
        out2 += residual
        out2 = F.relu(out2)
        return out2
    
    
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
    def __init__(self, filters_in=96, filters_out=32, in_channels=64, hidden_channels=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(filters_in, filters_out, kernel_size=1, stride=1, bias=True, padding='same'), # b x 32 x 4 x 4
            KataGoGlobalPooling(), # 2 x 32
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Tanh()
        )
        

    def forward(self, x):
        return self.net(x) 
    

class Critic_MCTS_Conv(nn.Module):
    def __init__(self, filters_in=96, filters_out=32, in_channels=64, hidden_channels=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(filters_in, filters_out, kernel_size=1, stride=1, bias=True, padding='same'), # b x 32 x 4 x 4
            KataGoGlobalPooling(), # 2 x 32
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 3)
        )
        

    def forward(self, x):
        return self.net(x) 
    

class Policy_Head(nn.Module):
    def __init__(self, in_channels=96, out_channels=8, hidden_channels=64):
        super(Policy_Head, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, bias=False, padding='same')
        self.conv2 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, bias=False, padding='same')
        self.pool = GlobalPooling_BiasStructure(in_channels=hidden_channels, out_channels=hidden_channels)
        self.b2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding='same')
        self.dirdist_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_channels*4*4, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 18),
        )
        
    def forward(self, x):
        p = self.conv1(x)
        g = self.conv2(x)
        out = F.relu(self.b2(self.pool(p, g)))
        pos = self.conv3(out)
        dir_dist = self.dirdist_net(pos)
        return pos[:, :4, :, :].view(pos.size(0), -1), pos[:, 4:, :, :].view(pos.size(0), -1), dir_dist[:,:16], dir_dist[:,16:]
    
    
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
            ResidualBlock2D_PPO(256, 256),
            ResidualBlock2D_PPO(256, 256),
            ResidualBlock2D_PPO(256, 256),
            ResidualBlock2D_PPO(256, 256),
            ResidualBlock2D_PPO(256, 256),
            ResidualBlock2D_PPO(256, 256),
            ResidualBlock2D_PPO(256, 256),
            ResidualBlock2D_PPO(256, 256),
            ResidualBlock2D_PPO(256, 256),
            ResidualBlock2D_PPO(256, 256),
            ResidualBlock2D_PPO(256, 256),
            ResidualBlock2D_PPO(256, 256),
            ResidualBlock2D_PPO(256, 256),
            ResidualBlock2D_PPO(256, 256),
            ResidualBlock2D_PPO(256, 256),
            ResidualBlock2D_PPO(256, 256),
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
        self.critic = Critic(self.fc_input_size//2, hidden_channels=16) 
        

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
            nn.Conv2d(self.input_channels, 128, kernel_size=5, stride=1, bias=False, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Residual blocks
            ResidualBlock2D(128, 128, kernel=3),
            ResidualBlock2D(128, 128, kernel=3),
            ResidualBlock2D(128, 128, kernel=3),
            ResidualBlock2D(128, 128, kernel=3),
            ResidualBlock2D_Pooling(128, 128, kernel=3),
            ResidualBlock2D(128, 128, kernel=3),
            ResidualBlock2D(128, 128, kernel=3),
            ResidualBlock2D(128, 128, kernel=3),
            ResidualBlock2D(128, 128, kernel=3),
            ResidualBlock2D_Pooling(128, 128, kernel=3),
        )
        
        # policy head
        self.policy = Policy_Head(in_channels=128, out_channels=8, hidden_channels=64)
        
        # critic head
        self.critic = Critic_MCTS_Conv(filters_in=128, filters_out=32, in_channels=64, hidden_channels=48)
    
    def get_policy(self, x):
        '''
        Get policy
        '''
        x = self.backbone(x)
        passive_position_logits, aggressive_position_logits, passive_dir_logits, passive_dist_logits = self.policy(x)
        
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
        x = self.backbone(x)
        # critic
        q_value = self.critic(x)
        return q_value

    
    def forward(self, x):
        '''
        Get policy and q value
        '''
        feat = self.backbone(x)
        
        # policy
        passive_position_logits, aggressive_position_logits, passive_dir_logits, passive_dist_logits = self.policy(feat)
        
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