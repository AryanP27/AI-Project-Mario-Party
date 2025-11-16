"""
    SameIsLame/Train.py

    Trains a Reinforced Learning model to play a simulation of the game
    'Same Is Lame' from Mario Party 6
"""

""" Imports """

# Randomization for Base Policies
import random

# Game Environment
import gymnasium as gym # https://gymnasium.farama.org/
from gymnasium.spaces import Discrete, Dict
import numpy as np

# Reinforced Learning
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from collections import deque

# Exporting Data
import pickle
#import json

""" Definitions """

# Game Environment
class SameIsLameEnv(gym.Env):
    def __init__(self, config : dict = {
        "terminate_on" : 10,
        "win_on" : 3,
        "num_players" : 4,
        "action_space" : 4,
        "render_mode": True,
    }):
        self.num_players = config["num_players"]
        self.action_space = Discrete(config["action_space"])  # Choices: 0, 1, 2, 3

        self.terminate_on = config["terminate_on"]
        self.win_on = config["win_on"]
        self.render_mode = config["render_mode"]

        self.observation_space = Dict({
            "scores": gym.spaces.Box(low=0, high=self.terminate_on, shape=(self.num_players,), dtype=np.int32),
            "turn": Discrete(self.terminate_on + 1)
        })
        self.reset()

    def reset(self):
        self.scores = [0] * self.num_players
        self.turn = 1
        self.done = False
        return self._get_obs()

    def step(self, actions):
        self.unique_flags = [self._is_unique(a, actions) for a in actions]
        for i in range(self.num_players):
            self.scores[i] += self.unique_flags[i]
        self.turn += 1
        self.done = self.turn > self.terminate_on or any(score >= self.win_on for score in self.scores)
        rewards = self.unique_flags
        return self._get_obs(), rewards, self.done, {}

    def _get_obs(self):
        return {"scores": np.array(self.scores), "turn": self.turn}

    def _is_unique(self, action, action_list):
        return int(action_list.count(action) == 1)
    
    def render(self, actions=None): # Not sure about if this will be used but it's kinda essential
        if self.render_mode:
            print(f"Turn {self.turn}")
            print(f"Player Actions: {actions}")
            print(f"Who's unique?: {self.unique_flags}")
            print(f"End Turn")
            for i in range(0, len(self.scores)):
                print(f"Player {i + 1} pts: {self.scores[i].score}", end=" | ")
            print("")
        return
    
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x):
        return self.fc(x)

def basePolicies():
    p2 = random.choice([0,1,2,3])
    p3 = random.choice([0,1])
    p4 = 3
    return p2, p3, p4

""" CODE """

# Hyperparameters
lr = 1e-3 # Learning Rate
gamma = 0.99 # Discount Factor
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1
batch_size = 32
buffer_size = 10000
episodes = 500

# Environment
env = SameIsLameEnv()
state_dim = len(env._get_obs()["scores"]) + 1  # scores + turn
action_dim = env.action_space.n

# Q-network + optimizer
q_net = QNetwork(state_dim, action_dim)
optimizer = optim.Adam(q_net.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# Replay buffer
replay_buffer = deque(maxlen=buffer_size)

# Pick up Data
totalRewardList = []
#episodeResultList = []

# Training loop
for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    # The game itself
    while not done:
        # Convert state to tensor
        s = np.concatenate([state["scores"], [state["turn"]]])
        s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)

        # Epsilon-greedy action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = q_net(s_tensor)
                action = torch.argmax(q_values).item()

        # Base Policies
        p2, p3, p4 = basePolicies()

        # Step environment
        next_state, reward, done, _ = env.step([action, p2, p3, p4])  # single-agent for now
        total_reward += reward[0]

        # Store transition
        ns = np.concatenate([next_state["scores"], [next_state["turn"]]])
        replay_buffer.append((s, action, reward[0], ns, done))

        state = next_state

        # Train if enough samples, this is the part where the machine is learning
        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            #states = torch.tensor(states, dtype=torch.float32)
            states = torch.tensor(np.array(states), dtype=torch.float32)
            actions = torch.tensor(actions).unsqueeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            #next_states = torch.tensor(next_states, dtype=torch.float32)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)

            # Current Q-values
            q_values = q_net(states).gather(1, actions).squeeze()

            # Target Q-values
            with torch.no_grad():
                next_q_values = q_net(next_states).max(1)[0]
                targets = rewards + gamma * next_q_values * (1 - dones)

            # Loss + backprop
            loss = loss_fn(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # End of Episode
    print(f"Episode {episode}, Total Reward: {total_reward}")
    totalRewardList.append(total_reward)
    #episodeResultList.append({
    #    "final_scores": state["scores"],
    #    "turns": state["turn"]
    #})


# Export Results
with open("output/AgentScores.csv", "w") as output:
    output.write("Episode,Reward\n")
    for i in range(episodes):
        output.write(f"{str(i + 1)},{totalRewardList[i]}\n")

#with open("output/EpisodeResults.json", "w") as output:
#    json.dump(episodeResultList, output, indent=2)

# Save replay buffer
with open("output/replay_buffer.pkl", "wb") as output:
    pickle.dump(list(replay_buffer), output)

# Export Agent
torch.save(q_net.state_dict(), "output/q_network.pth")
