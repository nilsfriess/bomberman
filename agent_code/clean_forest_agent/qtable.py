import numpy as np

from collections import defaultdict

from .base_helpers import ACTIONS

from .state_action_helpers import random_action, one_hot_action
from .state_transform import state_to_features

def default_action():
    return [0] * 6

class QTableEstimator:
    def __init__(self, learning_rate, discount_factor):
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        self.table = defaultdict(default_action)

        self.first_update = True
        self.not_fitted   = True

        self.print_importance_every = 10
        self.print_cnt = 0

    def update_learning_rate(self, new_rate):
        self.regressor.learning_rate = new_rate

    def estimate(self, game_state):
        state = np.concatenate(state_to_features(game_state)).tobytes()

        best_action = ACTIONS[np.argmax(self.table[state])]
        return best_action
    
    def update(self, transitions):
        if self.not_fitted:
            first_game_state = np.concatenate(transitions[0][0])
            self.feature_size = first_game_state.size
            
        self.qlearning(transitions)

    # 3 step TD
    def qlearning(self, transitions):
        num_trans = len(transitions)

        if num_trans < 3:
            return None, None

        for i in range(len(transitions)):
            (old_state, action, new_state, reward) = transitions[i]
            
            q_max = max(self.table[np.concatenate(new_state).tobytes()])

            old_state = np.concatenate(old_state).tobytes()
            q_old = self.table[old_state][ACTIONS.index(action)]
            q_new = q_old + self.learning_rate*(reward + self.discount_factor*q_max - q_old)

            self.table[old_state][ACTIONS.index(action)] = q_new
