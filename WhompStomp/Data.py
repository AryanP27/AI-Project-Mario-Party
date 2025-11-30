"""
    WhompStomp/Data.py

    Evaluates a Reinforced Learning model that played simulation of
    the game 'Whomp Stomp' from Mario Party 9

    Note: Run Train.py and Test.py first to get the expected files
"""

import random
import gymnasium as gym
from gymnasium.spaces import Discrete, Dict
import numpy as np
import torch
import torch.nn as nn
from collections import deque

import matplotlib.pyplot as plt
import pandas as pd

""" Definitions """

class WhompStompEnv(gym.Env):
    def __init__(self, **kwargs):
        self.num_players = kwargs.get("num_players", 4)
        self.action_space = Discrete(2)
        
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

""" Code """

# Environment
env = WhompStompEnv()
state_dim = env.state_dim()
action_dim = env.action_space.n

# Load Q-Network
q_net = QNetwork(state_dim, action_dim)
q_net.load_state_dict(torch.load("output/whomp_q_network.pth"))
q_net.eval()

# Plot Q Network Weights
print("Generating Q Network Weight Distributions...")
for name, param in q_net.named_parameters():
    plt.figure(figsize=(8, 5))
    plt.hist(param.detach().numpy().flatten(), bins=50)
    plt.title(f"Distribution of {name}")
    plt.xlabel("Weight value")
    plt.ylabel("Frequency")
    plt.savefig(f"output/Data_{name.replace('.','_')}_hist.png")
    plt.close()

# Visualize Q-values for Sample States
print("Generating Q-value Visualization...")
sample_state = env.reset()
s_tensor = torch.tensor(sample_state, dtype=torch.float32).unsqueeze(0)
q_values = q_net(s_tensor).detach().numpy()[0]

plt.figure(figsize=(8, 5))
plt.bar(range(action_dim), q_values)
plt.xlabel("Actions (0=Left, 1=Right)")
plt.ylabel("Q-value")
plt.title("Q-values for Initial State")
plt.savefig("output/Data_q_values.png")
plt.close()

# Graph Test Data
print("Generating Test Data Charts...")
df = pd.read_csv("output/WhompStomp_Test_Data.csv")

# Chart 1: All metrics over episodes
plt.figure(figsize=(14, 7))
ax1 = plt.gca()
ax1.set_xlabel("Episode")
ax1.set_ylabel("Score / Reward", color='tab:blue')
ax1.plot(df["Episode"], df["Reward"], label="Agent Reward", color='tab:blue', linewidth=2)
ax1.plot(df["Episode"], df["p1score"], label="Agent Final Score", color='tab:cyan', linewidth=1, alpha=0.7)
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.set_ylabel("Win Rate (rolling)", color='tab:red')
rolling_wins = df["Win"].rolling(window=50).mean()
ax2.plot(df["Episode"], rolling_wins, label="Win Rate (50-episode window)", color='tab:red', linewidth=2)
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.legend(loc='upper right')

plt.title("Agent Performance Over Episodes")
plt.tight_layout()
plt.savefig("output/Data_Episodes.png", dpi=100)
plt.close()

# Chart 2: Distribution of final scores
plt.figure(figsize=(10, 6))
plt.hist(df["p1score"], bins=20, alpha=0.7, label="Agent Scores", edgecolor='black')
plt.xlabel("Final Score")
plt.ylabel("Frequency")
plt.title("Distribution of Agent Final Scores Across Episodes")
plt.legend()
plt.savefig("output/Data_score_distribution.png")
plt.close()

# Chart 3: Win rate over time
plt.figure(figsize=(12, 6))
rolling_win_rate = df["Win"].rolling(window=50).mean() * 100
plt.plot(df["Episode"], rolling_win_rate, linewidth=2)
plt.fill_between(df["Episode"], rolling_win_rate, alpha=0.3)
plt.xlabel("Episode")
plt.ylabel("Win Rate (%)")
plt.title("Agent Win Rate Over Time (50-episode Rolling Average)")
plt.ylim([0, 100])
plt.savefig("output/Data_win_rate.png")
plt.close()

print("Data analysis complete! Charts saved to output/")