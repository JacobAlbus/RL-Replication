import gym
import torch
import random
from tqdm import tqdm
import torch.nn.functional as F

from dqn_model import Q, ReplayMemory

class TrainingEnv:

  def __init__(self, environment_name, history_len, N=1000000, M=100, T=1000, frame_width=84, frame_height=84):
    '''
      Parameters:
        environment_name (str): name of Atari game
        num_actions (int): number of actions in environment
        history_len (int): number of most recent frames being considered
        N (int): maximum size of replay memory
        M (int): number of training rounds
        T (int): maximum length of single training round
    '''
    # self.env = gym.make(environment_name, full_action_space=False, obs_type="grayscale", render_mode="human")
    self.env = gym.make(environment_name, full_action_space=False, obs_type="grayscale")

    self.action_value_function = Q(self.env.action_space.n, history_len)
    self.optimizer = torch.optim.RMSprop(self.action_value_function.parameters())
    self.replay_memory = ReplayMemory(N)

    self.M = M
    self.T = T
    self.frame_width = frame_width
    self.frame_height = frame_height
    self.history_len = history_len
    # self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    self.device = torch.device('cpu')

  def preprocess_history(self, history):
    '''
    Parameters:
      history (list): all action-state pairs (state being a frame of the environment) from training round
      
    Returns:
      Tensor of size (history_len, frame_height, frame_width) where each element in first dimension
      is a frame of the environment (i.e we stack the most recent frames)
    '''
    recent_history = history[-self.history_len:]
    processed_history = torch.empty((self.history_len, self.frame_height, self.frame_width))

    for index, item in enumerate(recent_history):
      _, state = item
      processed_history[index] = state

    return processed_history.unsqueeze(dim=0)

  def preprocess_observation(self, observation):
    '''
    Parameters:
      observation (numpy array): frame from environment 
      
    Returns:
      Processed frame from environment
    '''
    height, width = observation.shape
    observation = observation.reshape((1, 1, height, width))
    
    # downscale image
    new_height, new_width = 110, 84
    observation = F.upsample(observation, size=(new_height, new_width), mode='bilinear').reshape((new_height, new_width))

    # crop image by taking off equal portions from both sides
    final_height, final_width = 84, 84
    height_cutoff = abs((final_height - new_height) // 2)
    width_cutoff = abs((final_width - new_width) // 2)

    observation = observation[height_cutoff:, width_cutoff:]

    if height_cutoff != 0:
      observation = observation[:-height_cutoff]
    if width_cutoff != 0:
      observation = observation[:, :-width_cutoff]

    return observation

  def select_action(self, observation):
    '''
    Parameters:
      observation (numpy array): stack of frames from environment
      
    Returns:
      Either random action or greedily chosen action
    '''
    epsilon = 0.05

    if random.random() < epsilon:
      return random.randint(0, 3)
    else:
      state_action_values = self.action_value_function.forward(observation)
      return torch.argmax(state_action_values, dim=1)
  
  def calculate_loss(self, minibatch):
    '''
    Parameters:
      minibatch (list): set of history, reward, and action tuples
      
    Returns:
      Sum of MSE loss
    '''
    y = torch.zeros(len(minibatch))
    state_action_values = torch.zeros(len(minibatch))

    for index, sample in enumerate(minibatch):
      phi_prior, action, reward, phi_post, terminated = sample

      y[index] = reward if terminated else reward + torch.max(self.action_value_function.forward(phi_post))
      state_action_values[index] = self.action_value_function.forward(phi_prior)[0][action]

    loss = torch.nn.MSELoss()
    return loss(state_action_values, y)

  def update_model(self):
    '''
    Updates the model according to the loss using RSMprop algorithm
    '''
    self.optimizer.zero_grad()

    sample_size = 32
    minibatch = self.replay_memory.sample(sample_size)

    loss = self.calculate_loss(minibatch)
    loss.backward()

    self.optimizer.step()

  def train(self):
    '''
    Main function to train model
    '''
    for _ in range(self.M):
      observation, _ = self.env.reset(seed=42)
      observation = torch.from_numpy(observation).to(device=self.device)

      history = [(None, self.preprocess_observation(observation))]

      for _ in tqdm(range(self.T)):
        processed_history = self.preprocess_history(history)
        action = self.select_action(processed_history)

        observation, reward, terminated, _, _ = self.env.step(action)
        observation = torch.from_numpy(observation).to(device=self.device)
        history.append((action, self.preprocess_observation(observation)))

        new_memory = (processed_history, action, reward, self.preprocess_history(history), terminated)
        self.replay_memory.append(new_memory)

        self.update_model()

        if terminated:
          break
