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
from .state_transform import state_to_features

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
    
    self.filter_actions = True
    self.filter_prob = 0.9
    
    if os.path.isfile("models/model.pt"):
        with open("models/model.pt", "rb") as file:
            self.QEstimator = pickle.load(file)
            print("LOADED MODEL")
    else:
        self.QEstimator = QEstimator(learning_rate = self.initial_learning_rate,
                                     discount_factor = 0.95)
        
def random_action(allow_bombs = True):
    if allow_bombs:
        #print(ACTIONS)
        #return np.random.choice(ACTIONS)
        ind = np.random.randint(len(ACTIONS))
        return ACTIONS[ind]
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

    if self.filter_actions and (np.random.uniform() < self.filter_prob):
        for action in ACTIONS:
            if action_is_stupid(game_state, action):
                stupid_actions.append(action)

        if ('BOMB' not in stupid_actions) and (len(stupid_actions) == 5):
            # Too late, every direction is stupid
            stupid_actions = []

        if (len(stupid_actions) == 6):
            stupid_actions = []
            
    if np.random.uniform() < 1-self.epsilon:
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



