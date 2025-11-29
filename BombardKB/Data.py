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
import itertools

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

def series_to_csv(series : pd.Series, columns : list[str], path : str):
    pass

def all_bobomb_lineups(rewards=[1,2,3], numChoices=4):
    # Each slot can be 1, 2, or 3
    # But you may want to enforce your "at most 2 of one size" rule
    lineups = []
    for combo in itertools.product(rewards, repeat=numChoices):
        # enforce constraint: max 2 of any size
        if all(combo.count(r) <= 2 for r in rewards):
            lineups.append(combo)
    return lineups

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
lineups = all_bobomb_lineups()
rows = []

for lineup in lineups:
    # sample state: scores + turn + bobombs + phase
    # zeros for scores/turn/phase
    sample_state = np.concatenate([np.zeros(env.num_players), [0], lineup, [0]])
    s_tensor = torch.tensor(sample_state, dtype=torch.float32).unsqueeze(0)
    q_values = q_net(s_tensor).detach().numpy()[0]

    rows.append({
        "Bobombs": lineup,
        **{f"Q_action{a}": q_values[a] for a in range(action_dim)}
    })

df_q = pd.DataFrame(rows)
#print(df_q.head())
df_q.to_csv("output/Data_QValueTable.csv")

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
plt.title("Action distribution in replay buffer_train")
plt.savefig("output/Data_test_action_dist.png")
plt.close()

# Graph Test Data
print("Test Data Chart")
df_test = pd.read_csv("output/Test_Data.csv")

plt.figure(figsize=(12,6))
for col in ["Turns","Reward","p1win","p1score","p2score","p3score","p4score"]:
    plt.plot(df_test["Episode"], df_test[col], label=col)

plt.xlabel("Episode")
plt.ylabel("Value")
plt.title("Testing Results Over Episodes")
plt.legend()
plt.savefig("output/Data_Episodes.png")
plt.close()

# Graph Training Rewards
print("Train Data Chart")
df_train = pd.read_csv("output/Train_AgentScores.csv")

plt.figure(figsize=(12,6))
for col in ["Reward"]:
    plt.plot(df_test["Episode"], df_test[col], label=col)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Results Over Episodes")
#plt.legend()
plt.savefig("output/Data_training_rewards.png")
plt.close()

# Calcuate more stats for one file
print("Other Data")
with open("output/Data_other_stats.txt","w") as output:

    output.write("==== SCORE STATISTICS ====\n")

    output.write("=== MEAN SCORES ===\n")
    output.write(df_test[["Turns","Reward","p1win","p1score","p2score","p3score","p4score"]].mean().to_string())

    output.write("\n=== STANDARD DEVIATIONS OF SCORES ===\n")
    output.write(df_test[["Turns","Reward","p1win","p1score","p2score","p3score","p4score"]].std().to_string())

    output.write("\n=== ANOVA OF SCORES ===\n")
    f_val, p_val = stats.f_oneway(df_test["p1score"], df_test["p2score"], df_test["p3score"], df_test["p4score"])
    output.write(f"F = {f_val} p = {p_val}\n")

    output.write("\n\n==== AGENT ACTION STATISTICS ====\n")

    #output.write("=== ACTION DISTRIBUTIONS ===\n")
    actions = [t[1] for t in buffer_test]
    df_actions = pd.DataFrame(actions, columns=["Action"])
    #output.write(df_actions.to_string())

    output.write("=== CHI^2 ON ACTION DIST ===\n")
    action_counts = df_actions["Action"].value_counts().values
    chi2, p = stats.chisquare(action_counts)
    output.write(f"Chi-square = {chi2} p = {p}")

    output.write("\n=== ANOVA OF QVALUES AND STATESPACE ===\n")
    f_val, p_val = stats.f_oneway(df_q["Q_action0"], df_q["Q_action1"], df_q["Q_action2"], df_q["Q_action3"])
    output.write(f"F = {f_val} p = {p_val}\n")
