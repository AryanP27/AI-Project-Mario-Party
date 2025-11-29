"""
    BombardLB/Train.py

    Trains a Reinforced Learning model to play a simulation of the game
    'Bombard King Bob-omb' from Mario Party 9

    Each turn, players will choose which Bob-omb they want to go for, but only get to throw the Bob-omb
    if their decision was unique. Players have to balance picking the largest Bob-omb while not picking
    the same Bob-omb as someone else.

    Note: Run Train.py first to get q_network
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

    def epsilonGreedyAction(self, env : BombardKBEnv, s_tensor):
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

    def epsilonGreedyAction(self, env : BombardKBEnv, *args):
        if self.policy == 1:
            return random.choice([0,1])
        if self.policy == 2:
            return 3
        if self.policy == 3:
            return int(np.argmin(env.bobombs))
        if self.policy == 4:
            return int(np.argmax(env.bobombs))
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
env = BombardKBEnv()
state_dim = env.state_dim()
action_dim = env.action_space.n

# Define player list
players = [
    AgentPlayer(state_dim=state_dim, action_dim=action_dim),
    BasePlayer(4), # Picking the largest bobomb to score most points
    BasePlayer(0), # Picking randomly to be unpredictable
    BasePlayer(3), # Picking smallest to try to guarantee points
]

# Pick up Data
totalRewardList = []
episodeScores = []
episodeTurns = []
episodeWins = []
winCount = 0

# Load Player 1
players[0].load("output/Train_q_network.pth")

# Testing loop
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
        total_reward += reward[0]

        # Add event to replay buffer and do training (the machine is learning, reinforcing even)
        for i, p in enumerate(players):
            p.appendReplayBuffer((state, actionList[i], reward[i], next_state, done))

        state = next_state

    # End of Episode

    # Output
    #print(f"Episode {episode}, Total Reward: {total_reward}")
    totalRewardList.append(total_reward)
    episodeScores.append(env.scores)
    episodeTurns.append(env.turn)

    episodeWins.append(int(np.argmax(env.scores) == 0))
    winCount += episodeWins[episode]

# Export Results
print(f"Win Rate: {(winCount/episodes):.2%}")
with open("output/Test_Data.csv", "w") as output:
    output.write("Episode,Turns,Reward,Win,p1score,p2score,p3score,p4score\n")
    for i in range(episodes):
        output.write(f"{str(i + 1)},{episodeTurns[i]},{totalRewardList[i]},{episodeWins[i]},{episodeScores[i][0]},{episodeScores[i][1]},{episodeScores[i][2]},{episodeScores[i][3]}\n")

# Save replay buffer
with open("output/Test_replay_buffer.pkl", "wb") as output:
    pickle.dump(list(players[0].replay_buffer), output)
