import os
from pickle import load 

import numpy as np

#from .estimator import GBTEstimator
from .qtable import QTableEstimator
from .state_action_helpers import generate_stupid_actions, random_action, generate_suicidal_actions
from .base_helpers import ACTIONS

def setup(self):
    self.epsilon = 0.4
    self.learning_rate = 0.1
    self.discount_factor = 0.99

    self.initial_epsilon = self.epsilon
    self.initial_learning_rate = self.learning_rate

    self.action_filter_prob = 0.5
    self.initial_action_filter_prop = self.action_filter_prob
    
    #self.estimator = GBTEstimator(0.1, self.discount_factor)
    self.estimator = QTableEstimator(self.learning_rate, self.discount_factor)

    if os.path.isfile("models/model.pt"):
        with open("models/model.pt", "rb") as file:
            self.estimator = load(file)

            self.initial_epsilon = 0.2
            self.epsilon = self.initial_epsilon

            self.initial_learning_rate = 0.1
            self.learning_rate = 0.1

            self.action_filter_prob = 0.0
            self.initial_action_filter_prop = self.action_filter_prob
            
            print("LOADED MODEL")

def act(self, game_state):
    if np.random.uniform() < 1 - self.action_filter_prob: # Only filter sometimes
        stupid_actions = set()
    else:
        stupid_actions = generate_stupid_actions(game_state)

    suicidal_actions = generate_suicidal_actions(game_state)

    stupid_actions = suicidal_actions | stupid_actions

    if len(stupid_actions) == 6:
        stupid_actions = set()
    
    if np.random.uniform() < 1-self.epsilon:
        action = self.estimator.estimate(game_state)
        
        while action in stupid_actions:
            action = random_action()
    else:
        action = random_action()
        while action in stupid_actions:
            action = random_action()

    return action
