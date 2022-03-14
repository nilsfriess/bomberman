import numpy as np
import os
import pickle

from settings import SCENARIOS, ROWS, COLS

from time import sleep

from .qfunction import QEstimator
from .helpers import ACTIONS, \
    find_next_step_to_assets,\
    direction_from_coordinates,\
    cityblock_dist
from .action_filter import action_is_stupid

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
    self.initial_learning_rate = 0.15
    self.learning_rate = self.initial_learning_rate
    
    self.initial_epsilon = 0.4
    self.epsilon = self.initial_epsilon
    
    self.filter_actions = True
    self.filter_prob = 0.9
    
    if os.path.isfile("models/model.pt"):
        with open("models/model.pt", "rb") as file:
            self.QEstimator = pickle.load(file)
            print("LOADED MODEL")
    else:
        self.QEstimator = QEstimator(learning_rate = self.initial_learning_rate,
                                     discount_factor = 0.95)
        
def random_action(allow_bombs = True):
    if allow_bombs:
        #print(ACTIONS)
        #return np.random.choice(ACTIONS)
        ind = np.random.randint(len(ACTIONS))
        return ACTIONS[ind]
    else:
        return np.random.choice(ACTIONS[:-1])
                
def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    # Compute stupid actions
    stupid_actions = []

    if self.filter_actions and (np.random.uniform() < self.filter_prob):
        for action in ACTIONS:
            if action_is_stupid(game_state, action):
                stupid_actions.append(action)

        if ('BOMB' not in stupid_actions) and (len(stupid_actions) == 5):
            # Too late, every direction is stupid
            stupid_actions = []

        if (len(stupid_actions) == 6):
            stupid_actions = []
            
    if np.random.uniform() < 1-self.epsilon:
        state = state_to_features(game_state)
        av = np.array([self.QEstimator.estimate(state, action) for action in ACTIONS])        

        action = ACTIONS[np.argmax(av)]

        while action in stupid_actions:
            action = random_action()
            
    else:
        action = random_action()
        while action in stupid_actions:
            action = random_action()

    # print(f"Chose: {action}")
    return action


def state_to_features(game_state: dict) -> np.array:
    if game_state is None:
        return np.array([])

    # Assemble features

    ''' CLOSEST CRATE '''
    field = np.array(game_state['field'])
    (_,_,_, self_pos) = game_state['self']
    
    crates = []
    for x in range(field.shape[0]):
        for y in range(field.shape[1]):
            if field[x,y] == 1:
                crates.append((x,y))

    enemies = [(x,y) for (_,_,_,(x,y)) in game_state['others']]
    coord_to_closest_crate = find_next_step_to_assets(field,
                                                      [],
                                                      self_pos,
                                                      crates)
    crate_direction = direction_from_coordinates(self_pos,
                                                 coord_to_closest_crate)

    ''' OWN POSITION '''
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

    ''' BOMB DIRECTION '''
    bombs = [pos for (pos, _) in game_state['bombs']]
    coord_to_closest_bomb = find_next_step_to_assets(field,
                                                     [],
                                                     self_pos,
                                                     bombs)
    closest_bomb_direction = direction_from_coordinates(self_pos,
                                                        coord_to_closest_bomb)


    ''' RISK FACTOR OF THE FOUR DIRECTIONS '''
    risk = np.zeros((4,1)) # UP, DOWN, LEFT, RIGHT

    for i,action in enumerate(['UP', 'DOWN', 'LEFT', 'RIGHT']):
            if action_is_stupid(game_state, action):
                risk[i] = 1
    
    ''' 7x7 window around agent of blocked (wall/crate), free tiles and explosions '''    
    crates_window = np.zeros((7,7))
    x,y = self_pos
    for i in [-3,-2,-1,0,1,2,3]:
        for j in [-3,-2,-1,0,1,2,3]:
            coord_on_field = (x+i, y+j)
            
            if (x+i < 0)\
               or (x+i >= field.shape[0])\
               or (y+j < 0)\
               or (y+j >= field.shape[1]):
                crates_window[i+2,j+2] = 1
            else:
                if field[coord_on_field] == 1:
                    crates_window[i+2,j+2] = 1

    explosions = game_state['explosion_map']
    explosion_window = np.zeros((7,7))

    for i in [-3,-2,-1,0,1,2,3]:
        for j in [-3,-2,-1,0,1,2,3]:
            coord_on_field = (x+i, y+j)
            
            if (x+i < 0)\
               or (x+i >= field.shape[0])\
               or (y+j < 0)\
               or (y+j >= field.shape[1]):
                explosion_window[i+2,j+2] = 1
            else:
                if explosions[coord_on_field] > 1:
                    explosion_window[i+2,j+2] = 1

    bombs_window = np.zeros((7,7))
    
    if len(bombs) > 0:
        for i in [-3,-2,-1,0,1,2,3]:
            for j in [-3,-2,-1,0,1,2,3]:
                coord_on_field = (x+i, y+j)
                
                if (x+i < 0)\
                   or (x+i >= field.shape[0])\
                   or (y+j < 0)\
                   or (y+j >= field.shape[1]):
                    bombs_window[i+2,j+2] = 1
                else:
                    if coord_on_field in bombs:
                        bombs_window[i+2,j+2] = 1

    ''' DISTANCE TO CLOSEST BOMB '''
    bombs_dist = [cityblock_dist(self_pos, bomb) for bomb in bombs]
    if len(bombs_dist) == 0:
        min_bomb_dist = -1
    else:
        min_bomb_dist = min(bombs_dist)
    
    features = np.concatenate([
        crate_direction.ravel(),
        own_position.ravel(),
        coin_direction.ravel(),
        closest_enemy_direction.ravel(),
        # explosions_around.ravel(),
        # risks.ravel(),
        crates_window.ravel(),
        explosion_window.ravel(),#
        bombs_window.ravel(),
        closest_bomb_direction.ravel(),
        #coord_to_closest_bomb.ravel()
        risk.ravel(),
        [min_bomb_dist]
    ])

    #print(len(features))

    return features
