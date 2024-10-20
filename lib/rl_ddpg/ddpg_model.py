import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_range, option='both'):
        super(Actor, self).__init__()
        self.action_range = action_range
        self.heatmap_channels = state_dim[0]
        self.option = option   
        
        if option == 'both':
            self.conv1_heatmap = nn.Conv2d(self.heatmap_channels, 32, kernel_size=3, padding=1)
            self.conv2_heatmap = nn.Conv2d(32, 32, kernel_size=3, padding=1)
            self.maxpool_heatmap = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv1_depthmap = nn.Conv2d(self.heatmap_channels, 32, kernel_size=3, padding=1)
            self.conv2_depthmap = nn.Conv2d(32, 32, kernel_size=3, padding=1)
            self.maxpool_depthmap = nn.MaxPool2d(kernel_size=2, stride=2)
            self.input_dim = 64 * 24 * 18
            
        elif option == 'heatmap':
            self.conv1_heatmap = nn.Conv2d(self.heatmap_channels, 32, kernel_size=3, padding=1)
            self.conv2_heatmap = nn.Conv2d(32, 32, kernel_size=3, padding=1)
            self.maxpool_heatmap = nn.MaxPool2d(kernel_size=2, stride=2)
            self.input_dim = 32 * 24 * 18
            
        elif option == 'depthmap':
            self.conv1_depthmap = nn.Conv2d(self.heatmap_channels, 32, kernel_size=3, padding=1)
            self.conv2_depthmap = nn.Conv2d(32, 32, kernel_size=3, padding=1)
            self.maxpool_depthmap = nn.MaxPool2d(kernel_size=2, stride=2)
            self.input_dim = 32 * 24 * 18
        
        self.fc1 = nn.Linear(self.input_dim, 1024)
        self.fc2 = nn.Linear(1024, action_dim)
            
    def forward(self, heatmap, depthmap):
        if self.option == 'both':
            heatmap = F.relu(self.conv1_heatmap(heatmap))
            heatmap = self.maxpool_heatmap(heatmap)
            heatmap = F.relu(self.conv2_heatmap(heatmap))
            heatmap = self.maxpool_heatmap(heatmap)
            depthmap = F.relu(self.conv1_depthmap(depthmap))
            depthmap = self.maxpool_depthmap(depthmap)
            depthmap = F.relu(self.conv2_depthmap(depthmap))
            depthmap = self.maxpool_depthmap(depthmap)
        
            combined = torch.cat((heatmap, depthmap), 1)
            
        elif self.option == 'heatmap':
            combined = F.relu(self.conv1_heatmap(heatmap))
            combined = self.maxpool_heatmap(combined)
            combined = F.relu(self.conv2_heatmap(combined))
            combined = self.maxpool_heatmap(combined)
        
        elif self.option == 'depthmap':
            combined = F.relu(self.conv1_depthmap(depthmap))
            combined = self.maxpool_depthmap(combined)
            combined = F.relu(self.conv2_depthmap(combined))
            combined = self.maxpool_depthmap(combined)
            
        combined = combined.view(-1, self.input_dim)
        combined = F.relu(self.fc1(combined))
        return torch.tanh(self.fc2(combined)) * self.action_range  


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, option='both'):
        super(Critic, self).__init__()
        self.heatmap_channels = state_dim[0]
        self.option = option    
        
        if option == 'both':
            self.conv1_heatmap = nn.Conv2d(self.heatmap_channels, 32, kernel_size=3, padding=1)
            self.conv2_heatmap = nn.Conv2d(32, 32, kernel_size=3, padding=1)
            self.maxpool_heatmap = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv1_depthmap = nn.Conv2d(self.heatmap_channels, 32, kernel_size=3, padding=1)
            self.conv2_depthmap = nn.Conv2d(32, 32, kernel_size=3, padding=1)
            self.maxpool_depthmap = nn.MaxPool2d(kernel_size=2, stride=2)
            self.input_dim = 64 * 24 * 18
            
        elif option == 'heatmap':
            self.conv1_heatmap = nn.Conv2d(self.heatmap_channels, 32, kernel_size=3, padding=1)
            self.conv2_heatmap = nn.Conv2d(32, 32, kernel_size=3, padding=1)
            self.maxpool_heatmap = nn.MaxPool2d(kernel_size=2, stride=2)
            self.input_dim = 32 * 24 * 18
            
        elif option == 'depthmap':
            self.conv1_depthmap = nn.Conv2d(self.heatmap_channels, 32, kernel_size=3, padding=1)
            self.conv2_depthmap = nn.Conv2d(32, 32, kernel_size=3, padding=1)
            self.maxpool_depthmap = nn.MaxPool2d(kernel_size=2, stride=2)
            self.input_dim = 32 * 24 * 18
        
        self.fc1 = nn.Linear(self.input_dim, 1024)
        self.fc2 = nn.Linear(1024 + action_dim, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, heatmap, depthmap, action):
        action = nn.Flatten()(action)
        if self.option == 'both':
            heatmap = F.relu(self.conv1_heatmap(heatmap))
            heatmap = self.maxpool_heatmap(heatmap)
            heatmap = F.relu(self.conv2_heatmap(heatmap))
            heatmap = self.maxpool_heatmap(heatmap)
            depthmap = F.relu(self.conv1_depthmap(depthmap))
            depthmap = self.maxpool_depthmap(depthmap)
            depthmap = F.relu(self.conv2_depthmap(depthmap))
            depthmap = self.maxpool_depthmap(depthmap)
            
            combined = torch.cat((heatmap, depthmap), 1)
        
        elif self.option == 'heatmap':
            combined = F.relu(self.conv1_heatmap(heatmap))
            combined = self.maxpool_heatmap(combined)
            combined = F.relu(self.conv2_heatmap(combined))
            combined = self.maxpool_heatmap(combined)
            
        elif self.option == 'depthmap':
            combined = F.relu(self.conv1_depthmap(depthmap))
            combined = self.maxpool_depthmap(combined)
            combined = F.relu(self.conv2_depthmap(combined))
            combined = self.maxpool_depthmap(combined)
        
        combined = combined.view(-1, self.input_dim)
        combined = F.relu(self.fc1(combined))
        combined = torch.cat((combined, action), 1)
        combined = F.relu(self.fc2(combined))
        return self.fc3(combined)        
    