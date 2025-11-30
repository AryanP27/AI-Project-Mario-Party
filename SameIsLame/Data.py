"""
    SameIsLame/Data.py

    Evaluates a Reinforced Learning model that played simulation of
    the game 'Same Is Lame' from Mario Party 6

    Note: Run Train.py and Test.py first to get the expected files
"""

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

# Evaluation
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from scipy import stats
from math import sqrt

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

def zScoreMean(df: pd.DataFrame, column: str, expectedMean: float):
    sample_mean = df[column].mean()
    sample_std = df[column].std()
    n = len(df)

    z = (sample_mean - expectedMean) / (sample_std / sqrt(n))
    return z

def zScoreProp(df: pd.DataFrame, column: str, expectedProp: float):
    sample_prop = df[column].mean()
    n = len(df)

    z = (sample_prop - expectedProp) / sqrt((expectedProp * (1 - expectedProp)) / n)
    return z

""" Code """

# Environment
env = SameIsLameEnv()
state_dim = env.state_dim()  # scores + turn
action_dim = env.action_space.n

# Load Q-Network
q_net = QNetwork(state_dim, action_dim)
q_net.load_state_dict(torch.load("output/Train_q_network.pth"))
q_net.eval()

# Plot Q Network Weights
print("Q Network Weights")
for name, param in q_net.named_parameters():
    plt.hist(param.detach().numpy().flatten(), bins=50)
    plt.title(f"Distribution of {name}")
    plt.xlabel("Weight value")
    plt.ylabel("Frequency")
    plt.savefig(f"output/Data_{name.replace('.','_')}_hist.png")
    plt.close()

# Vizualize for Sample States
print("Q Values")
sample_state = env.reset()
s_tensor = torch.tensor(sample_state, dtype=torch.float32).unsqueeze(0)
q_values = q_net(s_tensor).detach().numpy()[0]

plt.bar(range(action_dim), q_values)
plt.xlabel("Actions")
plt.ylabel("Q-value")
plt.title("Q-values for sample state")
plt.savefig("output/Data_q_values.png")
plt.close()

# Analyse Replay Buffers
print("Train Action Distribution")
with open("output/Train_replay_buffer.pkl", "rb") as input:
    buffer_train = pickle.load(input)

actions = [t[1] for t in buffer_train]
plt.hist(actions, bins=range(action_dim+1))
plt.title("Action distribution in replay buffer_train")
plt.savefig("output/Data_train_action_dist.png")
plt.close()

print("Test Action Distribution")
with open("output/Test_replay_buffer.pkl", "rb") as input:
    buffer_test = pickle.load(input)

actions = [t[1] for t in buffer_test]
plt.hist(actions, bins=range(action_dim+1))
plt.title("Action distribution in replay buffer_test")
plt.savefig("output/Data_test_action_dist.png")
plt.close()

# Graph Testing Data
print("Test Data Chart")
df_test = pd.read_csv("output/Test_Data.csv")

plt.figure(figsize=(12,6))
for col in ["Reward","p1win","p1score","p2score","p3score","p4score"]:
    plt.plot(df_test["Episode"], df_test[col], label=col)

plt.xlabel("Episode")
plt.ylabel("Value")
plt.title("Training Results Over Episodes")
plt.legend()
plt.savefig("output/Data_Episodes.png")
plt.close()

plt.figure(figsize=(12,6))
for col in ["Turns"]:
    plt.plot(df_test["Episode"], df_test[col], label=col)

    plt.xlabel("Episode")
    plt.ylabel(col)
    plt.title("Training Results Over Episodes")
    plt.savefig(f"output/Data_Episode_{col}.png")
    plt.close()

# Graph Training Rewards
print("Train Data Chart")
df_train = pd.read_csv("output/Train_AgentScores.csv")

plt.figure(figsize=(12,6))
for col in ["Reward"]:
    plt.plot(df_train["Episode"], df_train[col], label=col)

    plt.xlabel("Episode")
    plt.ylabel(col)
    plt.title("Training Results Over Episodes")
    plt.savefig(f"output/Data_training_{col}.png")
    plt.close()

# Calcuate more stats for one file
print("Other Data")
with open("output/Data_other_stats.txt","w") as output:

    output.write("==== SCORE STATISTICS ====\n")

    output.write("=== MEAN SCORES ===\n")
    output.write(df_test[["Turns","Reward","p1win","p1score","p2score","p3score","p4score"]].mean().to_string())

    output.write("\n=== STANDARD DEVIATIONS OF SCORES ===\n")
    output.write(df_test[["Turns","Reward","p1win","p1score","p2score","p3score","p4score"]].std().to_string())

    output.write("\n=== Z SCORE P1 WINRATE VS 25% ===\n")
    output.write(f"Z = {zScoreProp(df_test, "p1win", 0.25)}")

    output.write("\n=== ANOVA OF SCORES ===\n")
    f_val, p_val = stats.f_oneway(df_test["p1score"], df_test["p2score"], df_test["p3score"], df_test["p4score"])
    output.write(f"F = {f_val} p = {p_val}\n")

    output.write("\n==== AGENT ACTION STATISTICS ====\n")

    actions = [t[1] for t in buffer_test]
    df_actions = pd.DataFrame(actions, columns=["Action"])

    output.write("=== CHI^2 ON ACTION DIST ===\n")
    action_counts = df_actions["Action"].value_counts().values
    chi2, p = stats.chisquare(action_counts)
    output.write(f"Chi-square = {chi2} p = {p}\n")
