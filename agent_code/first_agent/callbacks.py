import os
import pickle
import random

import numpy as np

import lin_q_policy as lq


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

INITIAL_PARAMS = np.array([[0]])
NUM_FEATURES = 0


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

    self.num_features = NUM_FEATURES

    if not os.path.isfile("params.pt"):
        self.logger.info("Setting up params from scratch.")
        self.params = INITIAL_PARAMS
    else:
        self.logger.info("Loading params from saved state.")
        with open("params.pt", "rb") as file:
            self.params = pickle.load(file)

    self.policy = lq.LinearQPolicy(self.params)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    if self.train:
        self.logger.debug("Choosing action according to training class.")
        return self.trainer.act(state_to_features(game_state))

    self.logger.debug("Querying model for action.")
    return self.policy.optimal_action(state_to_features(game_state))


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
