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

""" Definitions """

# Game Environment
class BombardKBEnv(gym.Env):
    def __init__(self, **kwargs):

        self.bossHealth = kwargs.get("bossHealth", 26)
        self.rewards = kwargs.get("rewards", [1, 2, 3])
        self.punishment = kwargs.get("punishment", -1)
        self.finalHitBonus = kwargs.get("finalHitBonus", 3)
        self.numChoices = kwargs.get("numChoices", 4)

        self.num_players = kwargs.get("num_players", 4)
        self.terminate_on = kwargs.get("terminate_on", 20)
        self.render_mode = kwargs.get("render_mode", False)

        self.observation_space = Dict({
            "scores": gym.spaces.Box(low=0, high=(self.bossHealth + self.finalHitBonus), shape=(self.num_players,), dtype=np.int32), # The highest score possible assume one player is the only one who picks a unique action
            "turn": Discrete(self.terminate_on + 1),
            "bobombs": gym.spaces.Box(low=1, high=3, shape=(4,), dtype=np.int32), # This is slightly redundant due to how bobombs are spawned as far as I know
            "phase" : Discrete(2)
        })
        self.action_space = Discrete(self.numChoices)

        self.reset() # This declares the other variables in a nonredundant way. Python's mysterious

    def reset(self):
        self.scores = [0] * self.num_players
        self.HP = self.bossHealth
        self.secondPhase = False
        self.turn = 1
        self.done = False
        self.bobombs = self.getBobombs()
        return self.flatten_obs(self._get_obs())

    def _get_obs(self):
        return {
            "scores": np.array(self.scores, dtype=np.int32),
            "turn": self.turn,
            "bobombs": np.array(self.bobombs, dtype=np.int32),
            "phase": int(self.secondPhase)
        }

    def step(self, actions):
        # Check uniqueness per player
        self.unique_flags = [self._is_unique(a, actions) for a in actions]

        # Map bobomb slots to the player who uniquely chose them
        correctChoices = [-1 for _ in range(len(self.bobombs))]
        for playerIndex, action in enumerate(actions):
            if self.unique_flags[playerIndex]:
                correctChoices[action] = playerIndex

        # Rewards
        rewards = [0 for _ in range(self.num_players)]

        # Left to Right Bob-omb order
        for bobombIndex, playerIndex in enumerate(correctChoices):
            if self.HP > 0 and playerIndex != -1:
                pts = self.bobombs[bobombIndex]
                self.scores[playerIndex] += pts
                self.HP -= pts
                rewards[playerIndex] = pts

                # Final hit bonus
                if self.HP <= 0:
                    self.scores[playerIndex] += self.finalHitBonus
                    rewards[playerIndex] += self.finalHitBonus

        # Punishment in second phase if boss still alive
        if self.secondPhase and self.HP > 0:
            for playerIndex in range(self.num_players):
                if not self.unique_flags[playerIndex]:
                    self.scores[playerIndex] = max(0, self.scores[playerIndex] - self.punishment)
                    rewards[playerIndex] = (0 if self.scores[playerIndex] <= 0 else self.punishment)

        # End turn
        self.turn += 1
        self.done = self.turn > self.terminate_on or self.HP <= 0
        self.secondPhase = self.HP <= (self.bossHealth / 2)

        if self.render_mode:
            self.render()

        return self.flatten_obs(self._get_obs()), rewards, self.done, {}

    def _is_unique(self, action, action_list):
        return action_list.count(action) == 1

    # From what I've observed and remember from this game, there's at most 2 of one size in the selection
    def getBobombs(self):
        sizesCopy = self.rewards.copy()
        bobombs = []
        for _ in range(self.numChoices):
            bobomb = random.choice(sizesCopy)
            if bobomb in bobombs:
                sizesCopy.remove(bobomb)
            bobombs.append(bobomb)
        #bobombs.sort() # to reduce dimensionality of state space for RL AI... nevermind
        return bobombs

    def render(self):
        print(f"Turn {self.turn} Phase {self.secondPhase + 1}")
        print(f"Correctness {self.unique_flags} BossHP {self.HP}")
        print(f"Scores: {self.scores}")
        pass

    def showBobOmbs(self):
        print(f"There are {self.numChoices} bob-ombs:")
        for i, bb in enumerate(self.bobombs):
            print(f"[{i}] pow: {bb}", end=" | ")
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

""" Code """

# Environment
env = BombardKBEnv()
state_dim = env.state_dim()  # scores + turn
action_dim = env.action_space.n

# Load Q-Network
q_net = QNetwork(state_dim, action_dim)
q_net.load_state_dict(torch.load("output/q_network.pth"))
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

# Table: bobomb lineup vs Q-values
print("Q values per Bob-omb Lineup")
df_q = pd.DataFrame({
    "Bobomb": env.bobombs,
    "Q-value": q_values
})
print(df_q)

# Analyse Replay Buffer
print("Action Distribution")
with open("output/replay_buffer.pkl", "rb") as input:
    buffer = pickle.load(input)

actions = [t[1] for t in buffer]
plt.hist(actions, bins=range(action_dim+1))
plt.title("Action distribution in replay buffer")
plt.savefig("output/Data_action_dist.png")
plt.close()

# Graph Test Data
print("Test Data Chart")
df = pd.read_csv("output/Test_Data.csv")

plt.figure(figsize=(12,6))
for col in ["Turns","Reward","Win","p1score","p2score","p3score","p4score"]:
    plt.plot(df["Episode"], df[col], label=col)

plt.xlabel("Episode")
plt.ylabel("Value")
plt.title("Training Results Over Episodes")
plt.legend()
plt.savefig("output/Data_Episodes.png")
plt.close()

#with open("output/Data_other_stats.txt","w") as output:
#    pass

# Mean and std dev for each column
print(df[["Turns","Reward","Win","p1score","p2score","p3score","p4score"]].mean())
print(df[["Turns","Reward","Win","p1score","p2score","p3score","p4score"]].std())

# Action Distribution Table
print("Action Distribution Table")
actions = [t[1] for t in buffer]
df_actions = pd.DataFrame(actions, columns=["Action"])
print(df_actions.value_counts())

# ANOVA comparing scores across players
f_val, p_val = stats.f_oneway(df["p1score"], df["p2score"], df["p3score"], df["p4score"])
print("ANOVA F =", f_val, "p =", p_val)

# Chi-square test on action distribution
action_counts = df_actions["Action"].value_counts().values
chi2, p = stats.chisquare(action_counts)
print("Chi-square =", chi2, "p =", p)