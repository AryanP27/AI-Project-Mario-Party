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
import torch.nn as nn
import numpy as np
from collections import deque

# Exporting Data
import pickle
#import json

""" Definitions """

# Game Environment
class SameIsLameEnv(gym.Env):
    def __init__(self, **kwargs):

        self.num_players = kwargs.get("num_players", 4)
        self.action_space = Discrete(kwargs.get("action_space", 4))  # Choices: 0, 1, 2, 3

        self.terminate_on = kwargs.get("terminate_on", 10)
        self.win_on = kwargs.get("win_on", 3)
        self.render_mode = kwargs.get("render_mode", False)

        self.observation_space = Dict({
            "scores": gym.spaces.Box(low=0, high=self.terminate_on, shape=(self.num_players,), dtype=np.int32),
            "turn": Discrete(self.terminate_on + 1)
        })

        self.reset()

    def reset(self):
        self.scores = [0] * self.num_players
        self.turn = 1
        self.done = False
        return self.flatten_obs(self._get_obs())

    def _get_obs(self):
        return {"scores": np.array(self.scores), "turn": self.turn}

    def step(self, actions):
        self.unique_flags = [self._is_unique(a, actions) for a in actions]
        for i in range(self.num_players):
            self.scores[i] += self.unique_flags[i]
        self.turn += 1
        self.done = self.turn > self.terminate_on or any(score >= self.win_on for score in self.scores)
        rewards = self.unique_flags
        if self.render_mode:
            self.render()
        return self.flatten_obs(self._get_obs()), rewards, self.done, {}

    def _is_unique(self, action, action_list):
        return int(action_list.count(action) == 1)

    def render(self, actions=None): # Not sure about if this will be used but it's kinda essential
        print(f"Turn {self.turn}")
        print(f"Player Actions: {actions}")
        print(f"Who's unique?: {self.unique_flags}")
        print(f"End Turn")
        for i in range(0, len(self.scores)):
            print(f"Player {i + 1} pts: {self.scores[i].score}", end=" | ")
        print("")

    def flatten_obs(self, obs):
        parts = []
        for key, value in obs.items():
            arr = np.array(value, dtype=np.float32).reshape(-1)
            parts.append(arr)
        return np.concatenate(parts, axis=0)

    def state_dim(self):
        return self.flatten_obs(self._get_obs()).shape[0]

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

class AgentPlayer():
    def __init__(self, state_dim, action_dim, **kwargs):
        # Each player needs their own components

        self.gamma = kwargs.get("gamma", 0.99)
        self.epsilon = kwargs.get("epsilon", 1.0)
        self.epsilon_decay = kwargs.get("epsilon_decay", 0.995)
        self.epsilon_min = kwargs.get("epsilon_min", 0.1)
        self.batch_size = kwargs.get("batch_size", 32)

        # Replay buffer
        self.replay_buffer = deque(maxlen=kwargs.get("buffer_size", 10000))

        # Q-Network
        self.q_net = QNetwork(state_dim, action_dim)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=kwargs.get("lr", 1e-3))
        self.loss_fn = nn.MSELoss()

    def epsilonGreedyAction(self, env : SameIsLameEnv, s_tensor):
        if random.random() < self.epsilon:
            return env.action_space.sample()
        else:
            with torch.no_grad():
                self.q_values = self.q_net(s_tensor)
                return torch.argmax(self.q_values).item()

    def appendReplayBuffer(self, transition : tuple):
        self.replay_buffer.append(transition)

    def trainStep(self):
        if len(self.replay_buffer) >= self.batch_size:
            batch = random.sample(self.replay_buffer, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.tensor(np.array(states), dtype=torch.float32)
            actions = torch.tensor(actions).unsqueeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)

            # Current Q-values
            q_values = self.q_net(states).gather(1, actions).squeeze()

            # Target Q-values
            with torch.no_grad():
                next_q_values = self.q_net(next_states).max(1)[0]
                targets = rewards + self.gamma * next_q_values * (1 - dones)

            # Loss + backprop
            loss = self.loss_fn(q_values, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def decayEpsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path : str):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path : str):
        self.q_net.load_state_dict(torch.load(path))
        self.q_net.eval()

class BasePlayer():
    def __init__(self, policy : int):
        self.policy = policy

    def epsilonGreedyAction(self, env : SameIsLameEnv, *args):
        if self.policy == 1:
            return random.choice([0,1])
        if self.policy == 2:
            return 3
        return env.action_space.sample()

    def appendReplayBuffer(self, *args):
        pass
    def trainStep(self, *args):
        pass
    def decayEpsilon(self, *args):
        pass

""" CODE """

# Seed RNG
random.seed(0)

# Hyperparameters
episodes = 500

# Environment
env = SameIsLameEnv()
state_dim = env.state_dim()  # scores + turn
action_dim = env.action_space.n

# Pick up Data
totalRewardList = []

# Define player list
players = [
    AgentPlayer(state_dim=state_dim, action_dim=action_dim),
    BasePlayer(0),
    BasePlayer(1),
    BasePlayer(2),
]

# Training loop
for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    # The game itself
    while not done:
        # Convert state to tensor
        s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # Epsilon-greedy action
        actionList = []
        for player in players:
            actionList.append(player.epsilonGreedyAction(env, s_tensor))

        # Step environment
        next_state, reward, done, _ = env.step(actionList)  # single-agent for now
        total_reward += reward[0] # This is for agent/p1 since we output that

        # Add event to replay buffer and do training (the machine is learning, reinforcing even)
        for i, p in enumerate(players):
            p.appendReplayBuffer((state, actionList[i], reward[i], next_state, done))
            p.trainStep()

        state = next_state

    # End of Episode

    # Decay epsilon
    for player in players:
        player.decayEpsilon()

    # Output
    print(f"Episode {episode}, Total Reward: {total_reward}")
    totalRewardList.append(total_reward)

# Export Results
with open("output/AgentScores.csv", "w") as output:
    output.write("Episode,Reward\n")
    for i in range(episodes):
        output.write(f"{str(i + 1)},{totalRewardList[i]}\n")

#with open("output/EpisodeResults.json", "w") as output:
#    json.dump(episodeResultList, output, indent=2)

# Save replay buffer
with open("output/replay_buffer.pkl", "wb") as output:
    pickle.dump(list(players[0].replay_buffer), output)

# Export Agent
players[0].save("output/q_network.pth")
