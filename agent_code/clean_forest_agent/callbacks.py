import numpy as np

from .estimator import AMFEstimator
from .state_action_helpers import generate_stupid_actions, random_action
from .base_helpers import ACTIONS

def setup(self):
    self.epsilon = 0.2
    self.discount_factor = 0.99

    self.estimator = AMFEstimator(0.1, self.discount_factor)

def act(self, game_state):
    stupid_actions = generate_stupid_actions(game_state)
    
    if np.random.uniform() < 1-self.epsilon:
        qvalues = self.estimator.estimate(game_state)
        action = ACTIONS[np.argmax(qvalues)]

        

        while action in stupid_actions:
            action = random_action()
    else:
        action = random_action()
        while action in stupid_actions:
            action = random_action()

    return action
