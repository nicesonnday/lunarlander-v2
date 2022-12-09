import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary as summary_
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import base64, io
from tqdm import tqdm
import numpy as np
from collections import deque, namedtuple
from network import QNetwork, DuelingQNetwork
# For visualization
from gym.wrappers.monitoring import video_recorder
from IPython.display import HTML
from IPython import display 
import glob
import argparse

parser=argparse.ArgumentParser()

parser.add_argument('--mode', type=str, help='train or test')
parser.add_argument('--save_video', type=bool, default=False)
parser.add_argument('--save_plot', type=bool, default=False)
parser.add_argument('--num_test', type=int, default=1000)

args=parser.parse_args()


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, mode, seed=0, double=False):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.mode = mode
        self.double=double

        if self.mode=='Q':
          # Q-Network
          self.network_local = QNetwork(state_size, action_size, seed).to(device)
          self.network_target = QNetwork(state_size, action_size, seed).to(device)
          self.optimizer = optim.Adam(self.network_local.parameters(), lr=LR)

        elif self.mode=='DQ':
          # Dueling Q-Network
          self.network_local=DuelingQNetwork(state_size, action_size, seed).to(device)
          self.network_target=DuelingQNetwork(state_size, action_size, seed).to(device)
          self.optimizer = optim.Adam(self.network_local.parameters(), lr=LR)
        else:
          raise RuntimeError("The network mode does not be defined !!")

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        if len(state)==2:
            state, _ = state
        state = torch.from_numpy(state).float().to(device)
        self.network_local.eval()
        with torch.no_grad():
          action_values = self.network_local(state)
        self.network_local.train()
          # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma=0.99):
        states, actions, rewards, next_states, dones = experiences

        self.optimizer.zero_grad()
        if self.double:
          expected = self.network_local(states).gather(-1, actions)
          with torch.no_grad():
              next_actions=self.network_local(next_states).argmax(-1, keepdim=True)
              targets_next = self.network_target(next_states).gather(-1, next_actions)
              targets = rewards + gamma * targets_next * (1 - dones)
        else:
          with torch.no_grad():
              targets_next = self.network_target(next_states).detach().max(-1, keepdim=True)[0]
              targets = rewards + gamma * targets_next * (1 - dones)
          expected = self.network_local(states).gather(-1, actions)
        
        ### Loss calculation (we used Mean squared error)
        loss = F.mse_loss(expected, targets)
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.network_local, self.network_target, TAU)

        return expected - targets

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

def dqn(agent, ckpt_name, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                       # list containing scores from each episode
    mean_scores=[]
    scores_window = deque(maxlen=20)  # last 100 scores
    eps = eps_start                   # initialize epsilon
    
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 20 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            mean_scores.append(np.mean(scores_window))
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-20, np.mean(scores_window)))
            torch.save(agent.network_local.state_dict(), '{}_checkpoint.pth'.format(ckpt_name))
            break

    #return scores
    return mean_scores

def show_video(env_name):
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4 = 'video/{}.mp4'.format(env_name)
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display.display(HTML(data=''''''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")
        
def show_video_of_model(agent, env_name, filename):
    env = gym.make(env_name, render_mode='rgb_array')
    
    agent.network_local.load_state_dict(torch.load("{}_checkpoint.pth".format(filename)))
    state = env.reset()
    done = False
    while not done:
        frame = env.render()
        
        action  = agent.act(state)
        state, reward, done, _, _ = env.step(action)        
    env.close()

def test_model(agent, env_name, filename, num_test, save_video=False):
    env=gym.make(env_name, render_mode='rgb_array')
    vid = video_recorder.VideoRecorder(env, path="./{}.mp4".format(filename))
    agent.network_local.load_state_dict(torch.load("{}_checkpoint.pth".format(filename)))
    state = env.reset()
    scores=[]
    max_score = -1.

    print("\nStart test {} model...".format(filename))

    for i in tqdm(range(num_test)):
        done = False
        score=0
        #print("test [{}]/[{}]...".format((i+1), num_test), end="")
        while not done:
            frame = env.render()
            if save_video and max_score > 200:
                vid.capture_frame()
            action  = agent.act(state)
            state, reward, done, _, _ = env.step(action)  
            score += reward

        if score >= max_score:
            max_score = score

        scores.append(score)

    env.close()

    return scores


if __name__=="__main__":

    env = gym.make('LunarLander-v2')

    print('State shape: ', env.observation_space.shape)
    (state_size,) = env.observation_space.shape
    print('Number of actions: ', env.action_space.n)
    action_size = env.action_space.n


    BUFFER_SIZE = int(1e5)  # replay buffer size
    BATCH_SIZE = 64         # minibatch size
    GAMMA = 0.99            # discount factor
    TAU = 1e-3              # for soft update of target parameters
    LR = 5e-4               # learning rate 
    UPDATE_EVERY = 4        # how often to update the network
    NUM_TRAININGS = 10
    mode = args.mode
    save_video=args.save_video
    num_test = args.num_test
    save_plot=args.save_plot

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    agent_Q = Agent(state_size=state_size, action_size=action_size, mode='Q', seed=0, double=False)
    agent_DueQ = Agent(state_size=state_size, action_size=action_size, mode='DQ', seed=1, double=False)
    agent_DQ = Agent(state_size=state_size, action_size=action_size, mode='Q', seed=2, double=True)
    agent_DDueQ = Agent(state_size=state_size, action_size=action_size, mode='DQ', seed=3, double=True)
    

    # Summary #
    print("===== Q_Network model summary =====")
    print(summary_(agent_Q.network_local, input_size=(1,state_size), batch_size=BATCH_SIZE))
    print("===== DueQ_Network model summary =====")
    print(summary_(agent_DueQ.network_local, input_size=(1,state_size), batch_size=BATCH_SIZE))

    if mode=='train':
        print("\n\n===== Q_Network training =====")
        scores_Q = dqn(agent_Q, 'Q_network')

        print("===== Dueling Q_Network training =====")
        scores_DueQ = dqn(agent_DueQ, 'DueQ_network')

        print("===== Double Q_Network training =====")
        scores_DQ = dqn(agent_DQ, 'DQ_network')

        print("===== Double Dueling Q_Network training =====")
        scores_DDueQ = dqn(agent_DDueQ, 'DDueQ_network')
        if save_plot:
            fig = plt.figure(figsize=(16,8))
            ax = fig.add_subplot(111)
            plt.plot(scores_Q, label='QN')
            plt.plot(scores_DueQ, label='Dueling QN')
            plt.plot(scores_DQ, label='Double QN')
            plt.plot(scores_DDueQ, label='Double Dueling QN')

            plt.ylabel('Score')
            plt.xlabel('Episode #')
            plt.legend()
            plt.savefig('training plot.png')
    
    if mode=='test':

        agent_Q = Agent(state_size=state_size, action_size=action_size, mode='Q', seed=0, double=False)
        test_score_Q = test_model(agent_Q, 'LunarLander-v2', 'Q_network', save_video=False, num_test=num_test)

        agent_DueQ = Agent(state_size=state_size, action_size=action_size, mode='DQ', seed=1, double=False)
        test_score_DueQ = test_model(agent_DueQ, 'LunarLander-v2', 'DueQ_network', save_video=False, num_test=num_test)

        agent_DQ = Agent(state_size=state_size, action_size=action_size, mode='Q', seed=2, double=True)
        test_score_DQ = test_model(agent_DQ, 'LunarLander-v2', 'DQ_network', save_video=False, num_test=num_test)

        agent_DDueQ = Agent(state_size=state_size, action_size=action_size, mode='DQ', seed=3, double=True)
        test_score_DDueQ = test_model(agent_DDueQ, 'LunarLander-v2', 'DDueQ_network', save_video=False, num_test=num_test)
        
        print("Average test score of Q: ", np.array(test_score_Q).mean())
        print("\nAverage test score of DueQ: ", np.array(test_score_DueQ).mean())
        print("\nAverage test score of DQ: ", np.array(test_score_DQ).mean())
        print("\nAverage test score of DDueQ: ", np.array(test_score_DDueQ).mean())


        # save test score plot #
        if save_plot:
            fig = plt.figure(figsize=(16,8))
            ax = fig.add_subplot(111)
            plt.scatter(np.arange(1, num_test + 1), test_score_Q, label='QN')
            plt.scatter(np.arange(1, num_test + 1), test_score_DueQ, label='Dueling QN')
            plt.scatter(np.arange(1, num_test + 1), test_score_DQ, label='Double QN')
            plt.scatter(np.arange(1, num_test + 1), test_score_DDueQ, label='Double Dueling QN')

            plt.ylabel('Score')
            plt.xlabel('Episode #')
            plt.legend()
            plt.savefig('testing plot.png')