import os
import random

import numpy as np
import pyastar2d

from scipy.spatial.distance import cityblock

import settings as s

import pandas as pd


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.last_direction = [0,0]
    self.last_closest_coin = [np.inf, np.inf]
    self.path = []
    
    self.coordinates = [[(i,j) for j in range(s.COLS)] for i in range(s.ROWS)]
    
    # if self.train or not os.path.isfile("my-saved-model.pt"):
    #     self.logger.info("Setting up model from scratch.")
    #     weights = np.random.rand(len(ACTIONS))
    #     self.model = weights / weights.sum()
    # else:
    #     self.logger.info("Loading model from saved state.")
    #     with open("my-saved-model.pt", "rb") as file:
    #         self.model = pickle.load(file)

def cityblock_dist(x,y):
    return abs(x[0]-y[0]) + abs(x[1]-y[1])

def sign(x):
    if x == 0:
        return 0
    return -1 if x < 0 else 1

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
  
    self.logger.debug("Querying model for action.")
    # return np.random.choice(ACTIONS, p=self.model)

    self_pos = game_state['self'][3]
    coins_pos = game_state['coins']
    
    if len(coins_pos) == 0:
        # Just wait if no coin is in the game
        return ACTIONS[4]

    distance_to_closest = np.argmin(np.array([cityblock_dist(self_pos, coin)
                                              for coin in coins_pos]))    
    closest_coin = coins_pos[distance_to_closest]

    if closest_coin != self.last_closest_coin:
        self.last_closest_coin = closest_coin

        weights = np.array([[cityblock_dist(self_pos, coord)
                             for coord in row]
                            for row in self.coordinates], dtype=np.float32)
        weights = weights + 1
        weights[game_state['field'] != 0] = np.inf

        # Compute shortest path to closest coin using A*
        self.path = pyastar2d.astar_path(weights, self_pos, closest_coin, allow_diagonal=False)
        if self.path is None:
            return np.random.choice(ACTIONS, p=[1/4,1/4,1/4,1/4,0,0])
        self.path = self.path[1:]        

    if self.path is None or self.path.size == 0:
        return np.random.choice(ACTIONS, p=[1/4,1/4,1/4,1/4,0,0])

    next_coord, self.path = self.path[0], self.path[1:]
    direction = np.array([next_coord[0] - self_pos[0],  # vertical direction
                          next_coord[1] - self_pos[1]]) # horizontal direction
    
    leftright = lambda dir : ACTIONS[3] if dir < 0 else ACTIONS[1]
    updown    = lambda dir : ACTIONS[0] if dir < 0 else ACTIONS[2]

    if direction[1] != 0:
        return updown(direction[1])
    else:
        return leftright(direction[0])
