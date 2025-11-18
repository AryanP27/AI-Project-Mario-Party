"""
    SameIsLame/Game.py

    This is a simulation of the game 'Same Is Lame' from Mario Party 6
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

def isUniqueAction(action, actionList : list):
    copy = actionList.copy()
    copy.remove(action)
    if action in copy:
        return 0
    return 1

""" Code """

# Seed RNG
random.seed(0)

# Initialize players
players = [
    Player(type=0,choiceType=0),
    Player(type=0,choiceType=0),
    Player(type=0,choiceType=0),
    Player(type=0,choiceType=0)
]

# Variables
terminalTurns = 10
winTurns = 3

turns = 1
end = False
while turns < terminalTurns + 1:
    if end:
        break
    print(f"Turn {turns}")
    # Decision
    actionList = [player.decisionManagement() for player in players]
    print(f"Player Actions: {actionList}")
    # Unique Check
    uniqueChecks = [isUniqueAction(action, actionList) for action in actionList]
    print(f"Who's unique?: {uniqueChecks}")
    # Rewards
    for i in range(0, len(players)):
        players[i].score += uniqueChecks[i]
        if players[i].score >= winTurns:
            end = True

    print(f"End Turn")
    for i in range(0, len(players)):
        print(f"Player {i + 1} pts: {players[i].score}", end=" | ")
    print("")
    turns += 1
print("\nFinal Results")
for i in range(0, len(players)):
    print(f"Player {i + 1} pts: {players[i].score}")
