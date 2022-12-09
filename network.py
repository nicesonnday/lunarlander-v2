import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, action_size)

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x


class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(DuelingQNetwork, self).__init__()
        self.seed=torch.manual_seed(seed)
        self.fc1=nn.Linear(state_size, 64)
        self.fc2=nn.Linear(64, 64)
        self.fc_value=nn.Linear(64, 128)
        self.fc_adv=nn.Linear(64, 128)

        self.get_value=nn.Linear(128, 1)
        self.get_adv=nn.Linear(128, action_size)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        v = F.relu(self.fc_value(x))
        a = F.relu(self.fc_adv(x))
        
        v = self.get_value(v)
        a = self.get_adv(a)
        a_avg = torch.mean(a, dim=-1, keepdim=True)

        x = v + a - a_avg

        return x