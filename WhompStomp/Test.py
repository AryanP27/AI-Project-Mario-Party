"""
    WhompStomp/Test.py

    Tests a Reinforced Learning model to play a simulation of the game
    'Whomp Stomp' from Mario Party 9
"""

""" Imports """
import random
import gymnasium as gym
from gymnasium.spaces import Discrete, Dict
import numpy as np
import torch
import torch.nn as nn
from collections import deque

""" Definitions """

class WhompStompEnv(gym.Env):
    def __init__(self, **kwargs):
        self.num_players = kwargs.get("num_players", 4)
        self.action_space = Discrete(2)  # 0 or 1
        
        self.boss_health_start = kwargs.get("boss_health", 96)
        self.max_reward = kwargs.get("max_reward", 12)
        self.reward = kwargs.get("reward", 9)
        self.punishment = kwargs.get("punishment", -1)
        
        self.positions_rewards = [self.punishment, self.max_reward, self.reward, self.reward]
        self.phase_2_offsets = [0, 1, 2, 3]
        
        self.observation_space = Dict({
            "my_score": gym.spaces.Box(low=-5, high=50, shape=(1,), dtype=np.int32),
            "my_position": Discrete(4)
        })
        
        self.reset()
    
    def reset(self):
        self.scores = [0] * self.num_players
        self.positions = list(range(self.num_players))
        self.boss_health = self.boss_health_start
        self.phase = 1
        self.done = False
        return self.flatten_obs(self._get_obs())
    
    def _get_obs(self):
        return {
            "my_score": np.array([self.scores[0]], dtype=np.int32),
            "my_position": self.positions[0]
        }
    
    def step(self, actions):
        total_spins = sum(actions)
        
        if self.phase == 2:
            offset = random.choice(self.phase_2_offsets)
            total_spins += offset
        
        for i in range(self.num_players):
            self.positions[i] = (self.positions[i] + total_spins) % self.num_players
        
        damage = self.max_reward + (2 * self.reward)
        is_final_round = self.boss_health <= damage
        
        rewards = []
        if is_final_round:
            remaining_health = self.boss_health
            for i in range(self.num_players):
                position = self.positions[i]
                if position == 0:
                    reward_value = self.punishment
                else:
                    reward_value = remaining_health / 3
                self.scores[i] += reward_value
                self.scores[i] = max(0, self.scores[i])
                rewards.append(reward_value)
        else:
            for i in range(self.num_players):
                position = self.positions[i]
                reward_value = self.positions_rewards[position]
                self.scores[i] += reward_value
                self.scores[i] = max(0, self.scores[i])
                rewards.append(reward_value)
        
        self.boss_health -= damage
        
        if self.phase == 1 and self.boss_health <= self.boss_health_start / 2:
            self.phase = 2
        
        self.done = self.boss_health <= 0
        
        return self.flatten_obs(self._get_obs()), rewards, self.done, {}
    
    def flatten_obs(self, obs):
        parts = []
        parts.append(obs["my_score"].astype(np.float32))
        position_arr = np.array([obs["my_position"]], dtype=np.float32)
        parts.append(position_arr)
        return np.concatenate(parts, axis=0)
    
    def state_dim(self):
        return 2

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

class AgentPlayer():
    def __init__(self, state_dim, action_dim, **kwargs):
        self.gamma = kwargs.get("gamma", 0.99)
        self.epsilon = kwargs.get("epsilon", 1.0)
        self.epsilon_decay = kwargs.get("epsilon_decay", 0.995)
        self.epsilon_min = kwargs.get("epsilon_min", 0.1)
        self.batch_size = kwargs.get("batch_size", 32)
        
        self.replay_buffer = deque(maxlen=kwargs.get("buffer_size", 10000))
        self.q_net = QNetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=kwargs.get("lr", 1e-3))
        self.loss_fn = nn.MSELoss()
    
    def epsilon_greedy_action(self, env, s_tensor):
        if random.random() < self.epsilon:
            return env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = self.q_net(s_tensor)
                return torch.argmax(q_values).item()
    
    def append_replay_buffer(self, transition):
        self.replay_buffer.append(transition)
    
    def train_step(self):
        if len(self.replay_buffer) >= self.batch_size:
            batch = random.sample(self.replay_buffer, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.tensor(np.array(states), dtype=torch.float32)
            actions = torch.tensor(actions).unsqueeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)
            
            q_values = self.q_net(states).gather(1, actions).squeeze()
            
            with torch.no_grad():
                next_q_values = self.q_net(next_states).max(1)[0]
                targets = rewards + self.gamma * next_q_values * (1 - dones)
            
            loss = self.loss_fn(q_values, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, path):
        torch.save(self.q_net.state_dict(), path)
    
    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))
        self.q_net.eval()

class BasePlayer():
    def __init__(self, policy):
        self.policy = policy
    
    def epsilon_greedy_action(self, env, *args):
        if self.policy == 1:
            return 0
        elif self.policy == 2:
            return 1
        else:
            return env.action_space.sample()
    
    def append_replay_buffer(self, *args):
        pass
    
    def train_step(self, *args):
        pass
    
    def decay_epsilon(self, *args):
        pass

""" CODE """

random.seed(0)

# Hyperparameters
episodes = 500

# Environment
env = WhompStompEnv()
state_dim = env.state_dim()
action_dim = env.action_space.n

# Define player list
players = [
    AgentPlayer(state_dim=state_dim, action_dim=action_dim),
    BasePlayer(0),
    BasePlayer(1),
    BasePlayer(2),
]

# Pick up Data
total_reward_list = []
episode_scores = []
episode_turns = []
episode_wins = []
win_count = 0

# Load Player 1 (Agent)
players[0].load("output/whomp_q_network.pth")

# Testing loop
for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    turns = 0
    
    while not done:
        s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        # Get actions from all players
        action_list = []
        for player in players:
            action_list.append(player.epsilon_greedy_action(env, s_tensor))
        
        # Step environment
        next_state, reward, done, _ = env.step(action_list)
        total_reward += reward[0]
        turns += 1
        
        state = next_state
    
    # End of Episode
    total_reward_list.append(total_reward)
    episode_scores.append(env.scores)
    episode_turns.append(turns)
    
    # Check if agent won (highest score)
    episode_wins.append(int(np.argmax(env.scores) == 0))
    win_count += episode_wins[episode]

# Export Results
print(f"Win Rate: {(win_count/episodes):.2%}")
with open("output/WhompStomp_Test_Data.csv", "w") as output:
    output.write("Episode,Turns,Reward,Win,p1score,p2score,p3score,p4score\n")
    for i in range(episodes):
        output.write(f"{str(i + 1)},{episode_turns[i]},{total_reward_list[i]},{episode_wins[i]},{episode_scores[i][0]},{episode_scores[i][1]},{episode_scores[i][2]},{episode_scores[i][3]}\n")