import numpy as np

from collections import defaultdict

from .base_helpers import ACTIONS

from .state_action_helpers import random_action, one_hot_action, rotate_action
from .state_transform import state_to_features

def default_action():
    return [0] * 6

class QTableEstimator:
    def __init__(self, learning_rate, discount_factor):
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        self.table = defaultdict(default_action)

    def estimate(self, game_state):
        features = state_to_features(game_state)
        
        rotations = 0
        
        while features[0][0] != 1:
            features[0] = np.roll(features[0], 1)
            features[1] = np.roll(features[1], 1)
            features[2] = np.roll(features[2], 1)

            rotations += 1
            
        state = np.concatenate(features).tobytes()

        best_action = ACTIONS[np.argmax(self.table[state])]
        rot_action = rotate_action(best_action, -rotations)
        
        return rot_action
    
    def update(self, transitions):
        for i in range(len(transitions)):
            (old_state, action, new_state, reward) = transitions[i]
            new_state = new_state.copy()
            
            ''' 
            To reduce the size of the state space, we rotate the features
            such that the target direction is always 'UP', i.e., the component
            that is 1 in the first component vector of the feature is in
            the first place.
            '''
            
            rotations = 0
            while old_state[0][0] != 1:
                old_state[0] = np.roll(old_state[0], 1)
                old_state[1] = np.roll(old_state[1], 1)
                old_state[2] = np.roll(old_state[2], 1)

                rotations += 1
            
                
            while new_state[0][0] != 1:
                new_state[0] = np.roll(new_state[0], 1)
                new_state[1] = np.roll(new_state[1], 1)
                new_state[2] = np.roll(new_state[2], 1)

            # Rotate action
            action = rotate_action(action, rotations)
            
            q_max = max(self.table[np.concatenate(new_state).tobytes()])

            old_state = np.concatenate(old_state).tobytes()
            q_old = self.table[old_state][ACTIONS.index(action)]
            q_new = q_old + self.learning_rate*(reward + self.discount_factor*q_max - q_old)

            self.table[old_state][ACTIONS.index(action)] = q_new
