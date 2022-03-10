import numpy as np
import os
import pickle

from settings import SCENARIOS, ROWS, COLS

from .qfunction import QEstimator
from .helpers import ACTIONS, index_of_action, find_next_step_to_closest_coin, cityblock_dist

coin_count = SCENARIOS['coin-heaven']['COIN_COUNT']

EPSILON = 0.15 # Exploration/Exploitation parameter

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
    if os.path.isfile("models/model.pt"):
        with open("models/model.pt", "rb") as file:
            self.QEstimator = pickle.load(file)
    else:    
        self.QEstimator = QEstimator(learning_rate = 0.1,
                                     discount_factor = 0.95)

    self.initial_epsilon = 0.1

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    if (not self.train) or np.random.uniform() < 1-self.initial_epsilon:
        state = state_to_features(game_state)
        av = np.array([self.QEstimator.estimate(state, action) for action in ACTIONS])        
        best_action = ACTIONS[np.argmax(av)]
        
        return best_action
    else:
        action = np.random.choice(len(ACTIONS)-1)
        return ACTIONS[action]



def state_to_features(game_state: dict) -> np.array:
    if game_state is None:
        return np.array([])

    own_position    = np.zeros((ROWS, COLS), dtype=np.int8)
    crates_position = np.zeros((ROWS, COLS), dtype=np.int8)
    walls_position  = np.zeros((ROWS, COLS), dtype=np.int8)
    enemy_positions = np.zeros((ROWS, COLS), dtype=np.int8)
    coin_positions  = np.zeros(4, dtype=np.int8)
    bomb_positions  = np.zeros((ROWS, COLS), dtype=np.int8)

    field = np.array(game_state['field'])
    bombs = [(x,y) for ((x,y),_) in game_state['bombs']]
    others = [(x,y) for (_,_,_,(x,y)) in game_state['others']]
    (_,_,_, (x,y)) = game_state['self']
    coins = game_state['coins']

    # Assemble features
    crates_position[field == 1] = 1
    walls_position[field == -1] = 1

    n_closest_coins = min(len(coins), 10)
    coins = np.array(coins)
    coins = coins[np.argpartition(np.array([cityblock_dist((x,y), coin)
                                            for coin in coins]),
                                  n_closest_coins-1)]
    coord_to_closest_coin = find_next_step_to_closest_coin(field,
                                                           others,
                                                           (x,y),
                                                           coins[:n_closest_coins])

    if not ((coord_to_closest_coin[0] == x) and (coord_to_closest_coin[1] == y)):
        dist = coord_to_closest_coin - [x,y]

        if dist[0] == 0:
            if dist[1] == 1:
                coin_positions[0] = 1
            else:
                coin_positions[1] = 1
        else:
            if dist[0] == 1:
                coin_positions[2] = 1
            else:
                coin_positions[3] = 1

        assert(np.count_nonzero(coin_positions) == 1)
        
        #coin_positions[coord_to_closest_coin[0], coord_to_closest_coin[1]] = 1
    
    own_position[x,y] = 1
    explosion_positions = (game_state['explosion_map'] > 0).astype(np.int8)
    
    for i in range(ROWS):
        for j in range(COLS):
            if (i,j) in bombs:
                bomb_positions[i,j] = 1

            if (i,j) in others:
                enemy_positions[i,j] = 1
                
    features = np.concatenate([
        #crates_position.ravel(),
        #walls_position.ravel(),
        own_position.ravel(),
        enemy_positions.ravel(),
        coin_positions.ravel(),
        #bomb_positions.ravel(),
        #explosion_positions.ravel()
    ]).astype(np.int8)

    return features
