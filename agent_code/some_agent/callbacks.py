import numpy as np
import os
import pickle

from settings import SCENARIOS, ROWS, COLS

from .qfunction import QEstimator
from .helpers import ACTIONS, index_of_action, find_next_step_to_closest_coin, cityblock_dist, direction_to_best_coin, blocked_neighbourhood, bomb_risk_neighbourhood, explosion_neighbourhood, neighbouring_bomb_locations_t, bomb_danger_in_t, valid_actions, death_implying_actions
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
                stored_regressor = pickle.load(file)
                with open("some_state.pt", "rb") as file2:
                    some_state = pickle.load(file2)
                self.QEstimator.regressor = stored_regressor

                # test two member functions:
                temp = self.QEstimator.estimate(state_to_features(some_state), "UP")
                self.QEstimator.update([(state_to_features(some_state), "UP", state_to_features(some_state), 0),
                                        (state_to_features(some_state), "UP", state_to_features(some_state), 0)])

                # reset after testing update:
                self.QEstimator.regressor = stored_regressor
                print("Using stored regression parameters")

            except ValueError:
                print("Stored regression parameters have another shape, beginning to train new parameters, will overwrite old model after 50 steps.")
                self.QEstimator = QEstimator(learning_rate = 0.1,
                                             discount_factor = 0.8)


    self.initial_epsilon = EPSILON

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



def state_to_features(game_state: dict) -> np.array:
    if game_state is None:
        return np.array([])


    (_,_,_, (x,y)) = game_state['self']
    coins = np.array(game_state['coins'])
    field = np.array(game_state["field"])


    coin_positions = direction_to_best_coin(field, x, y, coins, 3)

    #blocked = blocked_neighbourhood(game_state, x, y, 1)

    bombs = neighbouring_bomb_locations_t(game_state, x, y, 1)
    danger = bomb_danger_in_t(game_state, x, y, 2)



    features = np.concatenate([
    bombs,
    danger
    ]).astype(np.int8)

    return features


def train_act(self, game_state: dict) -> str:
    # tweak probability to dodge a bomb:
    NEW_ACTIONS = np.array(["UP", "DOWN", "LEFT", "RIGHT"])
    if train_act.counter != 0:

        train_act.counter += 1
        if train_act.counter == 5:
            train_act.counter = 0

        bomb_risk = bomb_risk_neighbourhood(game_state, game_state["self"][3][0], game_state["self"][3][1], 1)[:-1]

        blocked = blocked_neighbourhood(game_state, game_state["self"][3][0], game_state["self"][3][1], 1)

        return NEW_ACTIONS[np.argmin(bomb_risk + 10*blocked)]

    if np.random.uniform() < 0.0:
        train_act.counter = 1
        return "BOMB"

    if np.random.uniform() < 1-self.initial_epsilon:

        death_actions = death_implying_actions(game_state)
        VALID_ACTIONS = np.setdiff1d(valid_actions(game_state), death_actions)

        if len(VALID_ACTIONS) > 0:
            state = state_to_features(game_state)

            av = np.array([self.QEstimator.estimate(state, action) for action in VALID_ACTIONS])
            best_action = VALID_ACTIONS[np.argmax(av)]

        elif len(VALID_ACTIONS) == 1:
            best_action = VALID_ACTIONS[0]

        else:
            best_action = "WAIT"

        return best_action

    else:
        action = np.random.choice(len(ACTIONS))
        return ACTIONS[action]

train_act.counter = 0
