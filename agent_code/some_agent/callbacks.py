import numpy as np
import os
import pickle

from settings import SCENARIOS, ROWS, COLS

from .qfunction import QEstimator
from .helpers import ACTIONS, index_of_action, find_next_step_to_closest_coin, cityblock_dist, direction_to_best_coin, blocked_neighbourhood
from .more_helpers import get_neighbourhood, get_step_neighbourhood

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
    self.QEstimator = QEstimator(learning_rate = 0.1,
                                 discount_factor = 0.8)
    if os.path.isfile("model.pt"):
        with open("model.pt", "rb") as file:
            try:
                stored_QEstimator = pickle.load(file)
                with open("some_state.pt", "rb") as file2:
                    some_state = pickle.load(file2)
                temp = stored_QEstimator.estimate(state_to_features(some_state), "UP")
                print("Using stored regression parameters")
                self.QEstimator = stored_QEstimator
            except ValueError:
                print("Stored regression parameters have another shape, beginning to train new parameters, will overwrite old model after 50 steps.")


    self.initial_epsilon = EPSILON

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
    bomb_positions  = np.zeros((ROWS, COLS), dtype=np.int8)

    explosion_positions = (game_state['explosion_map'] > 0).astype(np.int8)
    field = np.array(game_state['field'])
    bombs = [(x,y) for ((x,y),_) in game_state['bombs']]
    others = [(x,y) for (_,_,_,(x,y)) in game_state['others']]
    (_,_,_, (x,y)) = game_state['self']
    coins = np.array(game_state['coins'])

    this_agent_coo = np.array((x,y))
    other_agent_coo = np.array(others)

    # Assemble features
    crates_position[field == 1] = 1
    walls_position[field == -1] = 1



    # BOMBS
    # one hot encoding whether or not the 5 fields that can be reached in one step are in bomb danger
    bomb_danger = np.zeros(5)

    explosions = np.zeros(4)


    coin_positions = direction_to_best_coin(field, x, y, coins, 3)
    blocked = blocked_neighbourhood(field, others, x, y, 2)


    features = np.concatenate([
        #crates_position.ravel(),
        #walls_position.ravel(),
        #own_position.ravel(),
        #enemy_positions.ravel(),
        coin_positions,
        #bomb_positions.ravel(),
        #explosion_positions.ravel()
        #bomb_danger,
        blocked
    ]).astype(np.int8)

    return features
