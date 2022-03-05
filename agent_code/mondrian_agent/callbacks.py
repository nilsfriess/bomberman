import numpy as np
import pyastar2d
from settings import SCENARIOS, ROWS, COLS

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

coin_count = SCENARIOS['coin-heaven']['COIN_COUNT']

EPSILON = 0.05

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

    """ 
    The total number of states is the sum of the following values:
    - 3 possible values for the each tile on the game field
    - 2 possible values for the explosion map
    - the coordinates of each of the revealed coins. (0,0) means the coin was not revealed yet
    - coordinates of each agent (assuming 4 agents are playing)
    """
    self.Q = defaultdict(default_action)

def default_action() -> np.array:
    action = np.array([0,0,0,0,0,0])
    action_ind = np.random.choice(4)
    action[action_ind] = 1
    return action

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

    if np.random.uniform() < 1-EPSILON:
        state = state_to_features(game_state)    
        action = np.argmax(self.Q[np.array2string(state)])
    else:
        print("Random move")
        action = np.random.choice(len(ACTIONS)-1)

    return ACTIONS[action]

def state_to_features(game_state: dict) -> np.array:
    if game_state is None:
        return np.array([])
    
    field = game_state['field'].ravel()
    
    # bombs = game_state['bombs']
    # bombs = np.ravel([[x,y,countdown] for ((x,y),countdown) in bombs])

    explosion_map = game_state['explosion_map'].ravel()

    coins_pos = np.zeros((coin_count, 2))
    coins = np.array(game_state['coins'])
    
    if coins.size > 0:
        coins_pos[:coins.shape[0]] = coins

    coins_pos = coins_pos.ravel()

    _,_,_,self_pos = game_state['self']
    self_pos = np.asarray(self_pos)

    others = game_state['others']
    others_pos = np.ravel([np.asarray(pos) for (_,_,_,pos) in others])

    if len(coins) == 0:
        dir_to_closest_coin = index_of_actions('WAIT')
    else:
        index_of_closest = np.argmin(np.array([cityblock_dist(self_pos, coin)
                                               for coin in coins]))    
        closest_coin = coins[index_of_closest]
        path = find_path(game_state['field'], self_pos, closest_coin)
        if path.size == 0:
            dir_to_closest_coin = index_of_actions('WAIT')
        else:
            next_coord = path[0]
            direction = np.array([next_coord[0] - self_pos[0],  # vertical direction
                                  next_coord[1] - self_pos[1]]) # horizontal direction
    
            leftright = lambda dir : 3 if dir < 0 else 1
            updown    = lambda dir : 0 if dir < 0 else 2
    
            if direction[1] != 0:
                dir_to_closest_coin = updown(direction[1])
            else:
                dir_to_closest_coin = leftright(direction[0])

    features = np.concatenate([field,
                               explosion_map,
                               coins_pos,
                               self_pos,
                               others_pos,
                               [dir_to_closest_coin]]).astype(int)
    
    return features

def index_of_actions(action: str) -> int:
    return ACTIONS.index(action)
