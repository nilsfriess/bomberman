import numpy as np
import pyastar2d
from settings import SCENARIOS, ROWS, COLS

from .qfunction import QEstimator

from .helpers import ACTIONS, index_of_action

coin_count = SCENARIOS['coin-heaven']['COIN_COUNT']

EPSILON = 0.4 # Exploration/Exploitation parameter

from collections import defaultdict

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
    self.QEstimator = QEstimator(learning_rate = 0.1,
                                 discount_factor = 0.95)

    self.initial_epsilon = 0.1

def cityblock_dist(x,y):
    return abs(x[0]-y[0]) + abs(x[1]-y[1])

coordinates = [[(i,j) for j in range(COLS)] for i in range(ROWS)]
def find_path(field, start, goal):
    # compute manhattan distance from `start` to all the squares in the field
    weights = np.array([[cityblock_dist(start, coord)
                         for coord in row]
                        for row in coordinates], dtype=np.float32)
    weights = weights + 1 # weights must >= 1
    weights[field != 0] = np.inf # walls have infinite weight
    
    # Compute shortest path from start to goal using A*
    path = pyastar2d.astar_path(weights, start, goal, allow_diagonal=False)
    if path is None:
        return []
    return path[1:] # discard first element in path, since it's the start position

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    if np.random.uniform() < 1-self.initial_epsilon:
        state = state_to_features(game_state)
        best_action = 'WAIT'
        best_action_val = float('-inf')

        av = np.array([self.QEstimator.estimate(state, action) for action in ACTIONS])       
        
        for action in ACTIONS:
            action_val = self.QEstimator.estimate(state, action)
            if action_val > best_action_val:
                best_action = action
                best_action_val = action_val

        return best_action
    else:
        action = np.random.choice(len(ACTIONS)-1)
        return ACTIONS[action]

   

def state_to_features(game_state: dict) -> np.array:
    if game_state is None:
        return np.array([])

    own_position    = np.zeros((ROWS, COLS), dtype=np.int8)
    enemy_positions = np.zeros((ROWS, COLS), dtype=np.int8)
    coin_positions  = np.zeros((ROWS, COLS), dtype=np.int8)
    bomb_positions  = np.zeros((ROWS, COLS), dtype=np.int8)

    bombs = [(x,y) for ((x,y),_) in game_state['bombs']]
    others = [(x,y) for (_,_,_,(x,y)) in game_state['others']]
    (_,_,_, (x,y)) = game_state['self']
    coins = game_state['coins']

    # Assemble features
    field = game_state['field']
    own_position[x,y] = 1
    explosion_positions = (game_state['explosion_map'] > 0).astype(np.int8)
    
    for i in range(ROWS):
        for j in range(COLS):
            if (i,j) in bombs:
                bomb_positions[i,j] = 1

            if (i,j) in others:
                enemy_positions[i,j] = 1

            if (i,j) in coins:
                coin_positions[i,j] = 1

    features = np.concatenate([
        field.ravel(),
        own_position.ravel(),
        enemy_positions.ravel(),
        coin_positions.ravel(),
        bomb_positions.ravel(),
        explosion_positions.ravel()]).astype(np.int8)

    return features
            
    
    # field = game_state['field'].ravel()
    
    # # bombs = game_state['bombs']
    # # bombs = np.ravel([[x,y,countdown] for ((x,y),countdown) in bombs])

    # explosion_map = game_state['explosion_map'].ravel()

    # coins_pos = np.zeros((coin_count, 2))
    # coins = np.array(game_state['coins'])
    
    # if coins.size > 0:
    #     coins_pos[:coins.shape[0]] = coins

    # coins_pos = coins_pos.ravel()

    # _,_,_,self_pos = game_state['self']
    # self_pos = np.asarray(self_pos)

    # others = game_state['others']
    # others_pos = np.ravel([np.asarray(pos) for (_,_,_,pos) in others])

    # if len(coins) == 0:
    #     dir_to_closest_coin = index_of_action('WAIT')
    # else:
    #     index_of_closest = np.argmin(np.array([cityblock_dist(self_pos, coin)
    #                                            for coin in coins]))    
    #     closest_coin = coins[index_of_closest]
    #     path = find_path(game_state['field'], self_pos, closest_coin)
    #     if path.size == 0:
    #         dir_to_closest_coin = index_of_action('WAIT')
    #     else:
    #         next_coord = path[0]
    #         direction = np.array([next_coord[0] - self_pos[0],  # vertical direction
    #                               next_coord[1] - self_pos[1]]) # horizontal direction
    
    #         leftright = lambda dir : 3 if dir < 0 else 1
    #         updown    = lambda dir : 0 if dir < 0 else 2
    
    #         if direction[1] != 0:
    #             dir_to_closest_coin = updown(direction[1])
    #         else:
    #             dir_to_closest_coin = leftright(direction[0])

    # features = np.concatenate([field,
    #                            explosion_map,
    #                            coins_pos,
    #                            self_pos,
    #                            others_pos,
    #                            [dir_to_closest_coin]]).astype(int)
    
    return features
