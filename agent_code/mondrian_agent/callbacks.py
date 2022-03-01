import numpy as np

import settings as s

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
    pass

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    features = state_to_features(game_state)
    return 'WAIT'

def state_to_features(game_state: dict) -> np.array:
    field = game_state['field'].ravel()

    bombs = game_state['bombs']
    bombs = np.ravel([[x,y,countdown] for ((x,y),countdown) in bombs])

    explosion_map = game_state['explosion_map'].ravel()

    coins = np.array(game_state['coins']).ravel()

    _,_,_,self_pos = game_state['self']
    self_pos = np.asarray(self_pos)

    others = game_state['others']
    others_pos = np.ravel([np.asarray(pos) for (_,_,_,pos) in others])

    features = np.concatenate([field, bombs, explosion_map, coins, self_pos, others_pos])
    
    print(features)
    return np.array([])
