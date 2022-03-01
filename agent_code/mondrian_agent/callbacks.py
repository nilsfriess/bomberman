import numpy as np

from settings import SCENARIOS, WIDTH, HEIGHT

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

coin_count = SCENARIOS['classic']['COIN_COUNT']

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
    n_states = (WIDTH * HEIGHT * 3) \
             + (WIDTH * HEIGHT * 2) \
             + (coin_count * 2) \
             + (4 * 2)
    n_actions = len(ACTIONS)

    # default action in Q-table is waiting
    default_action =  np.array([0,0,0,0,1,0])

    # self.Q = np.zeros((n_states, n_actions))
    self.Q = defaultdict(lambda : default_action)
    

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    features = state_to_features(game_state)
    # print(features.shape)
    return 'WAIT'

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
    m = np.amax(explosion_map)

    coins_pos = coins_pos.ravel()

    _,_,_,self_pos = game_state['self']
    self_pos = np.asarray(self_pos)

    others = game_state['others']
    others_pos = np.ravel([np.asarray(pos) for (_,_,_,pos) in others])

    features = np.concatenate([field, explosion_map, coins_pos, self_pos, others_pos]).astype(int)
    
    return features
