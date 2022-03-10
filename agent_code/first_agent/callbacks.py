import os
import pickle
import random

import numpy as np
from numpy import linalg as NLA

from .lin_q_policy import LinearQPolicy, TrainLinearQPolicy


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
    self.num_features = state_to_features()[0]

    INITIAL_PARAMS = np.zeros((len(ACTIONS), self.num_features))

    if not os.path.isfile("params.pt"):
        self.logger.info("Setting up params from scratch.")
        self.params = INITIAL_PARAMS
    else:
        self.logger.info("Loading params from saved state.")
        with open("params.pt", "rb") as file:
            self.params = pickle.load(file)

    self.policy = LinearQPolicy(self.params)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    if self.train:
        self.logger.debug("Choosing action according to training class.")
        return self.trainer.act(state_to_features(game_state))

    self.logger.debug("Querying model for action.")
    return self.policy.optimal_action(state_to_features(game_state))

# returns a 1d array of features. The last feature is always one, accounting for non-centered data.
# if the argument is None, returns the number of features
def state_to_features(game_state: dict = None) -> np.array:
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
    num_features = 8
    features = np.empty(num_features)

    if game_state is None:
        return np.array([num_features])

    this_agent_coo = np.array(game_state["self"][3]) #shape [coo_index]
    num_opponents = len(game_state["others"])

    other_agent_coo = np.empty((num_opponents, 2))
    for index, tuple in enumerate(game_state["others"]):
        other_agent_coo[index] = np.array(tuple[3])

    coins = np.array(game_state["coins"]) #shape [num_coin_index, coo_index]

    # COIN POSITIONS
    if coins.shape[0] > 0:
        # use autom. broadcasting
        vectors_to_coins = coins - this_agent_coo
        # contract the coo_index
        vector_to_closest_coin = vectors_to_coins[np.argmin(np.einsum("nc,nc->n", vectors_to_coins, vectors_to_coins))]
        distance_to_coin = NLA.norm(vector_to_closest_coin)
    else:
        distance_to_coin = 10
        vector_to_closest_coin = np.random.rand(2)
    #coins should not be on top of the agent, but this occured during training...
    if distance_to_coin == 0:
        distance_to_coin = 10 # make it large because decision is not too important then.
    cos_angle_coin = vector_to_closest_coin[0] / distance_to_coin
    sin_angle_coin = vector_to_closest_coin[1] / distance_to_coin

    # INVALID MOVEMENTS
    # one hot encoding which direction is blocked, one means blocked:
    field = game_state["field"]
    blocked = np.zeros(4)
    for index, (dx, dy) in [(0,(0,1)), (1,(0,-1)), (2,(1,0)), (3,(-1,0))]:
        test_coordinate = this_agent_coo + np.array([dx, dy])
        if field[test_coordinate[0],test_coordinate[1]] != 0:
            blocked[index] = 1
            continue
        else:
            for dummy_index, agent_coo in enumerate(other_agent_coo):
                if np.array_equal(test_coordinate, agent_coo):
                    blocked[index] = 1
                    break # out of the agent loop



    # COIN POSITIONS
    # how near is the closest coin? is it worth going there?
    features[0] = 1 / distance_to_coin**2
    # in which direction does one have to go?
    features[1] = cos_angle_coin
    features[2] = sin_angle_coin

    # INVALID MOVEMENTS
    features[3:7] = blocked # next feature has index 7

    # add one constant feature to account for offset
    features[num_features - 1] = 1
    return features
