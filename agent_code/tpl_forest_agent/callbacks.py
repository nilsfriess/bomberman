import numpy as np
import os
import pickle

from settings import SCENARIOS, ROWS, COLS

from time import sleep

from qfunction import QEstimator
from base_helpers import ACTIONS, \
    find_next_step_to_assets,\
    direction_from_coordinates,\
    cityblock_dist
from action_filter import action_is_stupid

from .state_transform import state_to_features, train_act
#from .state_transform_nils import state_to_features, train_act

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

    if False and os.path.isfile("models/model.pt"):
        with open("models/model.pt", "rb") as file:
            self.QEstimator = pickle.load(file)
            print("LOADED MODEL")
    else:
        self.QEstimator = QEstimator(learning_rate = self.initial_learning_rate,
                                     discount_factor = 0.95)

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    if not self.train:
        state = state_to_features(game_state)
        av = np.array([self.QEstimator.estimate(state, action) for action in ACTIONS])
        best_action = ACTIONS[np.argmax(av)]

        return best_action

    else:
        return train_act(self, game_state)
