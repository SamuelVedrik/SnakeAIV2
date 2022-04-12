from collections import deque
import random
import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels), 
        )
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
    
    def forward(self, x):
        
        identity = self.residual(x)
        x2 = self.net(x)
        return identity + x2
    
class ResNetSnake(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.net1 = ResNetBlock(1, 32)
        self.net2 = ResNetBlock(32, 64)
        self.net3 = ResNetBlock(64, 128)
        self.net4 = ResNetBlock(128, 256)
        self.maxpool = nn.MaxPool2d(2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(260, 64), # 256 + 4: additional 4 is direction data
            nn.ReLU(),
            nn.Linear(64, 4) # 4: Number of actions
        )
    
    def forward(self, boards, directions):
        
        x = self.net1(boards)
        x = self.maxpool(x)
        x = self.net2(x)
        x = self.maxpool(x)
        x = self.net3(x)
        x = self.maxpool(x)
        x = self.net4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.cat([x, directions], dim=1)
        
        return self.fc(x)
        
        
class SnakeAI():
    def __init__(self, device):
        self.device = device
        self.policy_net = ResNetSnake().to(device)
        self.target_net = ResNetSnake().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.step = 0
        self.gamma = 0.95
        
    def get_Q(self, state, next_state):
        
        state_board, state_dir = state
        nstate_board, nstate_dir = next_state
        Q_curr = self.policy_net(state_board.to(self.device), state_dir.to(self.device))
        Q_next = self.target_net(nstate_board.to(self.device), nstate_dir.to(self.device))
        return Q_curr, Q_next

    def update_target_net(self):
        if self.step % 5 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def update_step(self):
        self.step += 1
        
    def save_policy_net(self, save_path):
        torch.save(self.policy_net.state_dict(), save_path)
    
    def load_policy_net(self, load_path):
        self.policy_net.load_state_dict(torch.load(load_path, map_location = self.device))
        
    def get_action(self, state, epsilon):
        """
        Epsilon greedy action choice
        
        epsilon: float between 0 and 1, dictates how adventurous we should be.
        """
        state_board, state_dir = state
        state_board = state_board.unsqueeze(0)
        state_dir = state_dir.unsqueeze(0)
        self.policy_net.eval()
        with torch.no_grad():   
            Q_curr = self.policy_net(state_board.to(self.device), state_dir.to(self.device))
            
        self.policy_net.train()
        random_value = torch.rand(1)
        return torch.randint(0, state_dir.shape[1], (1,))  if random_value < epsilon else Q_curr.argmax()
        
        
class ExperienceBuffer():
    
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
    
    def collect(self, experience):
        self.buffer.append(experience)
    
    def sample_from_experience(self, sample_size):
        if len(self.buffer) < sample_size:
            sample_size = len(self.buffer)
        
        sample = random.sample(self.buffer, sample_size)
        state_boards = torch.stack([exp[0][0] for exp in sample])
        # A one-hot vector denoting direction.
        state_dir = torch.stack([exp[0][1] for exp in sample])
        action = torch.FloatTensor([exp[1] for exp in sample])
        reward = torch.FloatTensor([exp[2] for exp in sample])
        nstate_boards = torch.stack([exp[3][0] for exp in sample])
        nstate_dir = torch.stack([exp[3][1] for exp in sample])

        return (state_boards, state_dir), action, reward, (nstate_boards, nstate_dir)



    