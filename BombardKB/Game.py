"""
    BombardKB/Game.py

    This is a simulation of the game 'Bombard King Bob-omb' from Mario Party 9

    Each turn, players will choose which Bob-omb they want to go for, but only get to throw the Bob-omb
    if their decision was unique. Players have to balance picking the largest Bob-omb while not picking
    the same Bob-omb as someone else.
"""

""" Imports """
import random

""" Definitions """

class Player():
    def __init__(self, type : int = 0, choiceType = 0):
        self.type = type # 0 for random, 1 for human, 2 for AI
        self.score = 0
        self.choiceType = choiceType

    def decisionManagement(self):
        if self.type == 0:
            if self.choiceType == 0:
                return self.randomChoice()
            if self.choiceType == 0:
                return self.pick1or2()
            return self.pick3or4()
        if self.type == 1:
            return self.manualChoice()
        if self.type == 2:
            return self.manualChoice() # AI choice to be implemented

    def randomChoice(self):
        return random.choice([0,1,2,3])

    def pick1or2(self):
        return random.choice([0,1])

    def pick3or4(self):
        return random.choice([2,3])

    def manualChoice(self):
        response = None
        while response not in [0,1,2,3]:
            response = input("Please select a number between 0 and 3: ")
        return response

    pass

# From what I've observed and remember from this game, there's at most 2 of one size in the selection
def getBobombs(sizes : list[int]):
    sizesCopy = sizes.copy()
    bobombs = []
    for i in range(0,4):
        bobomb = random.choice(sizesCopy)
        if bobomb in bobombs:
            sizesCopy.remove(bobomb)
        bobombs.append(bobomb)
    bobombs.sort() # to reduce dimensionality of state space for RL AI
    return bobombs

# Check if the Bobomb desired is unique
def isUniqueAction(action, actionList : list):
    copy = actionList.copy()
    copy.remove(action)
    if action in copy:
        return 0
    return 1

""" Code """

# Seed RNG
random.seed(0)
# Second phase starts at half health
bossHealth = 26

rewards = [1, 2, 3]
punishment = -1
finalHitBonus = 3
choices = 4 # At most two of one Bob-omb size is present

# Game Start

KingBBHP = bossHealth

# Initialize players
players = [
    Player(type=0,choiceType=0),
    Player(type=0,choiceType=0),
    Player(type=0,choiceType=0),
    Player(type=0,choiceType=0)
]

# First Phase
turns = 1
print("First Phase")
while KingBBHP > bossHealth / 2:
    if turns > 20:
        break
    # Generate Bob-ombs
    bobList = getBobombs(rewards)
    # Print them
    print(f"\nThere are {len(bobList)} Bobombs:")
    for i in range(0,len(bobList)):
        print(f"[{i}] pow: {bobList[i]}", end=" | ")
    print("")
    # Decision
    actionList = []
    for player in players:
        actionList.append(player.decisionManagement())
    # Uniqueness Check
    print(f"Player Actions: {actionList}")
    unique = []
    for i in range(0, len(actionList)):
        value = isUniqueAction(actionList[i], actionList)
        unique.append(value)
    # Damage
    for i in range(0, len(players)):
        if unique[i]:
            players[i].score += bobList[actionList[i]]
            KingBBHP -= bobList[actionList[i]]
    print(f"END TURN {turns}")
    for i in range(0, len(players)):
        print(f"Player {i + 1} pts: {players[i].score}", end=" | ")
    print(f"Boss Health: {KingBBHP}")
    turns += 1
# Status
print("\nHalfway Point")
for i in range(0, len(players)):
    print(f"Player {i + 1} pts: {players[i].score}")
# Second Phase
while KingBBHP > 0:
    if turns > 20:
        break
    # Generate Bob-ombs
    bobList = getBobombs(rewards)
    # Print them
    print(f"\nThere are {len(bobList)} Bobombs:")
    for i in range(0,len(bobList)):
        print(f"[{i}] pow: {bobList[i]}", end=" | ")
    print("")
    # Decision
    actionList = []
    for player in players:
        actionList.append(player.decisionManagement())
    # Uniqueness Check
    print(f"Player Actions: {actionList}")
    unique = []
    for action in actionList:
        unique.append(isUniqueAction(action, actionList))
    # Damage
    # The last bob-omb hit cancels the other attacks along with the other punishments. However, since the bob-ombs are
    # already sorted to simplify the action space for the RL model, the order here has to be randomized to not give
    # priority to the now-leftmost bob-ombs and somewhat
    indexes = [i for i in range(0, len(players))]
    random.shuffle(indexes)
    for i in range(0, len(indexes)):
        if unique[indexes[i]]:
            players[indexes[i]].score += bobList[actionList[indexes[i]]]
            KingBBHP -= bobList[actionList[indexes[i]]]
            if KingBBHP <= 0:
                players[indexes[i]].score += finalHitBonus
                break
    for i in range(0, len(indexes)):
        if not unique[indexes[i]]:
            players[indexes[i]].score += punishment
            if players[indexes[i]].score < 0:
                players[indexes[i]].score = 0
    print(f"END TURN {turns}")
    for i in range(0, len(players)):
        print(f"Player {i + 1} pts: {players[i].score}", end=" | ")
    print(f"Boss Health: {KingBBHP}")
    turns += 1
print("\nFinal Results")
for i in range(0, len(players)):
    print(f"Player {i + 1} pts: {players[i].score}")
