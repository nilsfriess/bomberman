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
    if os.path.isfile("models/model.pt"):
        with open("models/model.pt", "rb") as file:
            self.QEstimator = pickle.load(file)
            print("LOADED MODEL")
    else:    
        self.QEstimator = QEstimator(learning_rate = 0.1,
                                     discount_factor = 0.98)

    self.initial_epsilon = 0.3



def random_action(allow_bombs = True):
    if allow_bombs:
        return np.random.choice(ACTIONS)
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
    for action in ACTIONS:
        if action_is_stupid(game_state, action):
            stupid_actions.append(action)

    #print(f"Stupid actions: {stupid_actions}")

    if ('BOMB' not in stupid_actions) and (len(stupid_actions) == 5):
        # Too late, every direction is stupid
        stupid_actions = []
    
    if np.random.uniform() < 1-self.initial_epsilon:
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
                                                      enemies,
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

    # ''' RISK-FACTORS OF SURROUNDING TILES '''
    # risks = np.zeros((9,1))

    # bombs = [(x,y) for ((x,y),_) in game_state['bombs']]
    # explosions = game_state['explosion_map']

    # x,y = self_pos

    # total_risk = 0
    
    # for k in [-2,-1,1,2]:
    #     if (x+k < 0) or (x+k) > ROWS-1:
    #         continue
    #     coord_on_board = (x+k,y)
    #     total_risk += explosions[coord_on_board]

    #     if (x+k,y) in bombs:
    #         total_risk += 1

    # for k in [-2,-1,1,2]:
    #     if (y+k < 0) or (y+k) > ROWS-1:
    #         continue
    #     coord_on_board = (x,y+k)
    #     total_risk += explosions[coord_on_board]

    #     if (x,y+k) in bombs:
    #         total_risk += 1

    # if self_pos in bombs:
    #     total_risk += 1
    # total_risk += 2*explosions[self_pos]
        
    # risks[0] = explosions[x-1, y]
    # risks[1] = explosions[x+1, y]
    # risks[2] = explosions[x, y-1]
    # risks[3] = explosions[x, y+1]
    # risks[4] = explosions[x,y]

    # if (x-1,y) in bombs:
    #     risks[0] += 1
    # if (x+1,y) in bombs:
    #     risks[1] += 1
    # if (x,y-1) in bombs:
    #     risks[2] += 1
    # if (x,y+1) in bombs:
    #     risks[3] += 1
    # if (x,y) in bombs:
    #     risks[4] += 1

    # total_risk = np.sum(risks)

    # ''' EXPLOSIONS AROUND '''
    # explosions_around = np.zeros((4,1))
    
    # explosions = game_state['explosion_map']
    # x,y = self_pos

    # if explosions[x-1,y] > 0:
    #     explosions_around[0] = 1
    # if explosions[x+1,y] > 0:
    #     explosions_around[1] = 1
    # if explosions[x,y-1] > 0:
    #     explosions_around[2] = 1
    # if explosions[x,y+1] > 0:
    #     explosions_around[3] = 1

    # ''' DIRECTION TO CLOSEST BOMB '''    
    # bombs = [(x,y) for ((x,y),_) in game_state['bombs']]
    # if len(bombs) == 0:
    #     bomb_direction = np.ones((4,1))
    #     coord_to_closest_bomb = np.array([-1,-1])
        
    # else:
    #     coord_to_closest_bomb = find_next_step_to_assets(field,
    #                                                      [],
    #                                                      self_pos,
    #                                                      bombs)
    #     bomb_direction = direction_from_coordinates(self_pos,
    #                                                 coord_to_closest_bomb)

    #     coord_to_closest_bomb = np.array(coord_to_closest_bomb)
    
    ''' 5x5 window around agent of blocked (wall/crate), free tiles and explosions '''    
    crates_window = np.zeros((5,5))
    x,y = self_pos
    for i in [-2,-1,0,1,2]:
        for j in [-2,-1,0,1,2]:
            coord_on_field = (x+i, y+j)
            
            if (x+i < 0)\
               or (x+i >= field.shape[0])\
               or (y+j < 0)\
               or (y+j >= field.shape[1]):
                continue
            else:
                if field[coord_on_field] == 1:
                    crates_window[i+2,j+2] = 1

    explosions = game_state['explosion_map']
    explosion_window = np.zeros((5,5))

    for i in [-2,-1,0,1,2]:
        for j in [-2,-1,0,1,2]:
            coord_on_field = (x+i, y+j)
            
            if (x+i < 0)\
               or (x+i >= field.shape[0])\
               or (y+j < 0)\
               or (y+j >= field.shape[1]):
                continue
            else:
                if explosions[coord_on_field] > 0:
                    explosion_window[i+2,j+2] = 1

    bombs = [pos for (pos, _) in game_state['bombs']]
    bombs_window = np.zeros((5,5))
    
    if len(bombs) > 0:
        for i in [-2,-1,0,1,2]:
            for j in [-2,-1,0,1,2]:
                coord_on_field = (x+i, y+j)
                
                if (x+i < 0)\
                   or (x+i >= field.shape[0])\
                   or (y+j < 0)\
                   or (y+j >= field.shape[1]):
                    continue
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
        #bomb_direction.ravel(),
        #coord_to_closest_bomb.ravel()
        [min_bomb_dist]
    ])

    #print(len(features))

    return features
