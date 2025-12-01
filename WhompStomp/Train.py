"""
    WhompStomp/Train.py

    Trains a Reinforced Learning model to play a simulation of the game
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
        
        # Position rewards: [-1, +12, +9, +9]
        self.positions_rewards = [self.punishment, self.max_reward, self.reward, self.reward]
        self.phase_2_offsets = [0, 1, 2, 3]
        
        self.observation_space = Dict({
            "my_score": gym.spaces.Box(low=-5, high=50, shape=(1,), dtype=np.int32),
            "my_position": Discrete(4)  # 0, 1, 2, or 3
        })
        
        self.reset()
    
    def reset(self):
        self.scores = [0] * self.num_players
        self.positions = list(range(self.num_players))  # [0, 1, 2, 3]
        self.boss_health = self.boss_health_start
        self.phase = 1
        self.done = False
        return self.flatten_obs(self._get_obs())
    
    def _get_obs(self):
        # Return observation for player 0 (the agent)
        return {
            "my_score": np.array([self.scores[0]], dtype=np.int32),
            "my_position": self.positions[0]
        }
    
    def step(self, actions):
        # Calculate rotation based on sum of actions
        total_spins = sum(actions)
        
        # Phase 2: add random offset
        if self.phase == 2:
            offset = random.choice(self.phase_2_offsets)
            total_spins += offset
        
        # Rotate positions
        for i in range(self.num_players):
            self.positions[i] = (self.positions[i] + total_spins) % self.num_players
        
        # Calculate normal damage
        damage = self.max_reward + (2 * self.reward)
        
        # Check if this is the final round (boss health will go to 0 or below)
        is_final_round = self.boss_health <= damage
        
        # Award rewards based on positions
        rewards = []
        if is_final_round:
            # Final round: remaining health gets split among 3 non-squished players
            remaining_health = self.boss_health
            for i in range(self.num_players):
                position = self.positions[i]
                if position == 0:
                    # Position 0 gets squished: loses 1 point
                    reward_value = self.punishment
                else:
                    # Split remaining health among the 3 safe players
                    reward_value = remaining_health / 3
                self.scores[i] += reward_value
                self.scores[i] = max(0, self.scores[i])  # Can't go below 0
                rewards.append(reward_value)
        else:
            # Normal round: standard rewards
            for i in range(self.num_players):
                position = self.positions[i]
                reward_value = self.positions_rewards[position]
                self.scores[i] += reward_value
                self.scores[i] = max(0, self.scores[i])  # Can't go below 0
                rewards.append(reward_value)
        
        # Reduce boss health
        self.boss_health -= damage
        
        # Check phase transition
        if self.phase == 1 and self.boss_health <= self.boss_health_start / 2:
            self.phase = 2
        
        # Check if game is over
        self.done = self.boss_health <= 0
        
        return self.flatten_obs(self._get_obs()), rewards, self.done, {}
    
    def flatten_obs(self, obs):
        # Flatten the observation into a 1D array for the neural network
        parts = []
        
        # Add my_score (already a 1D array)
        parts.append(obs["my_score"].astype(np.float32))
        
        # Add my_position as one-hot encoding or just as a value
        # Option 1: Just use the position value directly
        position_arr = np.array([obs["my_position"]], dtype=np.float32)
        parts.append(position_arr)
        
        return np.concatenate(parts, axis=0)
    
    def state_dim(self):
        # Score (1 value) + Position (1 value) = 2
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
            return 0  # Always choose 0
        elif self.policy == 2:
            return 1  # Always choose 1
        else:
            return env.action_space.sample()  # Random
    
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

# Players: Agent + 3 base players with different strategies
players = [
    AgentPlayer(state_dim=state_dim, action_dim=action_dim),
    BasePlayer(0),  # Random
    BasePlayer(1),  # Always 0
    BasePlayer(2),  # Always 1
]

# Data collection
total_reward_list = []
episode_scores = []

# Training loop
for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        # Get actions from all players
        action_list = []
        for player in players:
            action_list.append(player.epsilon_greedy_action(env, s_tensor))
        
        # Step environment
        next_state, reward, done, _ = env.step(action_list)
        total_reward += reward[0]
        
        # Train all players
        for i, p in enumerate(players):
            p.append_replay_buffer((state, action_list[i], reward[i], next_state, done))
            p.train_step()
        
        state = next_state
    
    # Decay epsilon for all players
    for player in players:
        player.decay_epsilon()
    
    # Logging
    if (episode + 1) % 50 == 0:
        print(f"Episode {episode + 1}, Agent Reward: {total_reward:.2f}")
    
    total_reward_list.append(total_reward)
    episode_scores.append(env.scores)

# Export results
with open("output/WhompStompScores.csv", "w") as f:
    f.write("Episode,AgentReward,P1Score,P2Score,P3Score,P4Score\n")
    for i in range(episodes):
        f.write(f"{i + 1},{total_reward_list[i]},{episode_scores[i][0]},{episode_scores[i][1]},{episode_scores[i][2]},{episode_scores[i][3]}\n")

# Save model
players[0].save("output/whomp_q_network.pth")
print("Training complete! Model saved.")