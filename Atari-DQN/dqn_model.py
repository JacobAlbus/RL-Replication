import torch.nn as nn
import torch
import torch.nn.functional as F
import random

class Q(nn.Module):
  def __init__(self, num_actions, history_len):
    '''
    Parameters:
      num_actions (int): number of actions in environment
      history_len (int): number of most recent frames being considered
    '''
    super(Q, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=history_len, out_channels=16, kernel_size=8, stride=4)
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)

    self.fc1 = nn.Linear(in_features=2592, out_features=256)
    self.fc2 = nn.Linear(in_features=256, out_features=num_actions)

  def forward(self, x):
    '''
    Parameters:
      x (tensor): tensor of size (frame_width, frame_width, history_len) containing most recent
                  frames from environment
      
    Returns:
      Tensor of size num_actions with each element corresponding to 
      expected returns of history-action pair
    '''
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))

    x = torch.flatten(x, start_dim=1)
    x = F.relu(self.fc1(x))

    return self.fc2(x)
  

class ReplayMemory:
    
  def __init__(self, N):
    '''
    Parameters:
      N (int): max size of replay memory
    '''
    self.memories = []
    self.index = 0
    self.N = N

  def append(self, new_memory):
    '''
    Parameters:
      new_memory (tuple): memory to be added to replay memory
    '''
    if len(new_memory) < self.N:
      self.memories.append(new_memory)
    else:
      self.memories[self.index] = new_memory
      self.index += 1

    if self.index == (self.N - 1):
      self.index = 0
  
  def sample(self, sample_size):
    '''
    Parameters:
      sample_size (int): number of samples
      
    Returns:
      Memories sampled from replay memory
    '''
    if len(self.memories) < sample_size:
      return self.memories
    
    return random.sample(self.memories, k=sample_size)
