import numpy as np
import os
import pickle

from settings import SCENARIOS, ROWS, COLS

from .qfunction import QEstimator
from .helpers import ACTIONS, \
    index_of_action,\
    find_next_step_to_assets,\
    direction_from_coordinates,\
    cityblock_dist

coin_count = SCENARIOS['coin-heaven']['COIN_COUNT']

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

    self.initial_epsilon = 0.3

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

    # Assemble features

    ''' CRATES '''
    field = np.array(game_state['field'])
    
    crates_position = np.zeros((ROWS, COLS))
    crates_position[field == 1] = 1

    ''' OWN POSITION '''
    (_,_,_, self_pos) = game_state['self']
    own_position = np.zeros((ROWS, COLS))
    own_position[self_pos] = 1
    
    ''' DIRECTION TO CLOSEST COIN '''
    # Find 10 closest coins, where `close` is w.r.t. the cityblock distance
    game_coins = np.array(game_state['coins'])
    n_closest_coins = min(len(game_coins), 10)
    coins = game_coins[np.argpartition(np.array([cityblock_dist(self_pos, coin)
                                                 for coin in game_coins]),
                                       n_closest_coins-1)]
    closest_coins = coins[:n_closest_coins]

    enemies = [(x,y) for (_,_,_,(x,y)) in game_state['others']]
    coord_to_closest_coin = find_next_step_to_assets(field,
                                                     enemies,
                                                     self_pos,
                                                     closest_coins)

    coin_direction = direction_from_coordinates(self_pos,
                                                coord_to_closest_coin)

    ''' ENEMY DIRECTIONS '''    
    coord_to_closest_enemy = find_next_step_to_assets(field,
                                                      [],
                                                      self_pos,
                                                      enemies)
    closest_enemy_direction = direction_from_coordinates(self_pos,
                                                         coord_to_closest_enemy)
    
    features = np.concatenate([
        crates_position.ravel(),
        own_position.ravel(),
        coin_direction.ravel(),
        closest_enemy_direction.ravel(),
    ])

    return features
