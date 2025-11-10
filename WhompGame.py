"""
    WhompGame.py

    A simulation of the tituar minigame from Mario Party 9
    Assumptions and Adjustments:
    - All active player will mash ground-pounds to obtain the maximum reward
    - 4 players will be active
    - People are computer random and not human random
    - Each player's optimal strategy does not involve psychological warfare (important assumption)
    - This will not include the 'Last Hit' bonus. Instead remaining health will be split evenly between
    non-squashed players on the final turn'
    - The offsets added in the second phase of the boss are calculated before decision making since we'll
    assume the players could calculate the trick themselves
"""

""" Imports """
import random
# need to import RL model
# Import Ray RL Lib
# Import tkinter

""" Definitions """

class WhompStomp():
    def __init__(self):
        self.state = self.init_state()
        self.done = False

    def reset(self):
        self.state = self.init_state()
        self.done = False
        return self.state

    def step(self, actions):
        # Apply actions from multiple agents
        rewards, next_state = self.update_state(actions)
        self.state = next_state
        return next_state, rewards, self.done

    def render(self):
        # Optional: visualize using Tkinter or matplotlib
        pass
    pass

class Player():
    def __init__(self, type : int = 0, position : int = None, rngType : int = 0):
        self.type = type # 0 for random, 1 for human, 2 for AI
        self.position = position
        self.score = 0
        self.rngType = rngType # 0 for random, 1 for optimal choice, 2 for random optimal choice
        #return self
    
    def decisionManagement(self):
        if self.type == 0:
            if self.rngType == 1:
                return self.optimalChoice()
            if self.rngType == 2:
                return self.optimalChoiceChance()
            return self.randomChoice()
        if self.type == 1:
            return self.manualChoice()
        if self.type == 2:
            return self.manualChoice() # not sure how the AI will pick

    def randomChoice(self):
        return random.choice([0,1])

    # These were determined by ZoomZike's Identifying Luck in Mario Party 9 YouTube video
    def optimalChoice(self):
        if self.position == 0:
            return random.choice([0,1])
        if self.position == 1:
            return 1
        if self.position == 2:
            return random.choice([0,1])
        if self.position == 3:
            return 0

    # These were determined by ZoomZike's Identifying Luck in Mario Party 9 YouTube video
    def optimalChoiceChance(self):
        if self.position == 0:
            return random.choice([0,1])
        if self.position == 1: # pick the optimal choice 75% of the time
            if random.randint(1,4) < 4:
                return 1
            return 0
        if self.position == 2:
            return random.choice([0,1])
        if self.position == 3: # pick the optimal choice 75% of the time
            if random.randint(1,4) < 4:
               return 0
            return 1
        pass

    # Non-zero answers are treated like 1
    def manualChoice(self):
        try:
            response = input("Pick 0 or 1: ")
            return int(bool(int(response)))
        except:
            return 1
        

""" Code """

# Seed RNG
RNG = 76
random.seed(RNG)
# Assume midpoint phase will always be bossHealth / 2
bossHealth = 96
offsets = [0, 1, 2, 3]
# rewards assume every player mashes properly
punishment = -1
reward = 9
maxReward = 12
# Positions (rotates counter-clockwise)
positions = [
    punishment, # top
    maxReward, # right
    reward, # bottom
    reward # left
]

# Game Start

WhompHP = bossHealth
spins = 0
# Initialize players
players = [
    Player(type=1,position=0,rngType=0),
    Player(type=0,position=1,rngType=0),
    Player(type=0,position=2,rngType=0),
    Player(type=0,position=3,rngType=0),
]

# First Phase
print("First Phase")
while WhompHP > bossHealth / 2:
    spins = 0
    # Decision Calculation
    for player in players:
        spins += player.decisionManagement()
        print(spins, end=" ")
    print("")
    # Rotation
    for player in players:
        player.position = (player.position + spins) % 4
    # Rewards
    for player in players:
        player.score += positions[player.position]
        if player.score < 0:
            player.score = 0
    WhompHP -= maxReward + (2 * reward)
    print("END TURN")
    for i in range(0, len(players)):
        print(f"Player {i + 1} pts: {players[i].score} pos: {players[i].position}", end=" | ")
    print("")
# Midpoint
print("Halfway Point")
for i in range(0, len(players)):
    print(f"Player {i + 1} pts: {players[i].score}")
# Second Phase
print("Second Phase")
while WhompHP > 0:
    spins = random.choice(offsets) # The Whomp steps on the counter and increases it a random amount
    print(f"offset: {spins}")
    # We are spinning it automatically here before decision making since it can assumed that each player would mentally do it anyway
    for player in players:
        player.position = (player.position + spins) % 4
    spins = 0
    # Decision Calculation
    for player in players:
        spins += player.decisionManagement()
        print(spins, end=" ")
    print("")
    # Rotation
    for player in players:
        player.position = (player.position + spins) % 4
    # Rewards
    if WhompHP <= maxReward + (2 * reward): # Remainder adjustment
        for player in players:
            if player.position == 0:
                player.score -= 1
            else:
                player.score += (WhompHP / 3)
        WhompHP = 0
    else:
        for player in players:
            player.score += positions[player.position]
            if player.score < 0:
                player.score = 0
        WhompHP -= maxReward + (2 * reward)
    print("END TURN")
    for i in range(0, len(players)):
        print(f"Player {i + 1} pts: {players[i].score} pos: {players[i].position}", end=" | ")
    print("")
print("Final Results")
for i in range(0, len(players)):
    print(f"Player {i + 1} pts: {players[i].score}")
