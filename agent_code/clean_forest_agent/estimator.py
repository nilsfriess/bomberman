import numpy as np

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor

from .base_helpers import ACTIONS

from .state_action_helpers import random_action, one_hot_action, rotate_game_to_upper_left, rotate_action, rotate_game_state
from .state_transform import state_to_features

class AMFEstimator:
    def __init__(self, learning_rate, discount_factor):
        self.discount_factor = discount_factor

        self.regressor = GradientBoostingRegressor(warm_start=True,
                                                   max_depth=3,
                                                   learning_rate = learning_rate,
                                                   n_estimators=1)

        self.first_update = True
        self.not_fitted   = True

    def update_learning_rate(self, new_rate):
        self.regressor.learning_rate = new_rate

    ''' If action is None, just return the best action and corresponding value '''
    def estimate(self, game_state: dict, rotate = True):
        if self.not_fitted:
            return np.zeros((len(ACTIONS),))

        if rotate:
            game_state, rotations = rotate_game_to_upper_left(game_state)
            
        state = state_to_features(game_state)

        if rotate:
            actions = [rotate_action(action, rotations) for action in ACTIONS]
        else:
            actions = ACTIONS
            
        qvalues = np.zeros((len(ACTIONS),))
        for k, action in enumerate(actions):
            X = np.append(state, one_hot_action(action)).reshape(1,-1)
            qvalues[k] = self.regressor.predict(X)[0]
                
        return qvalues

    def update(self, transitions):
        if self.not_fitted:
            first_game_state = transitions[0][0]
            first_transformed_transition = state_to_features(first_game_state)
        
            self.feature_size = first_transformed_transition.size
            
        X,y = self.qlearning(transitions)

        self.regressor.fit(X, y)
        self.regressor.n_estimators += 1

        self.not_fitted = False

    def qlearning(self, transitions):
        num_trans = len(transitions)
        
        X = np.empty((num_trans, self.feature_size + len(ACTIONS)))
        y = np.empty((num_trans,))

        for i, (old_game_state, action, new_game_state, reward) in enumerate(transitions):
            old_game_state, rotations = rotate_game_to_upper_left(old_game_state)
            new_game_state = rotate_game_state(new_game_state, rotations)

            action = rotate_action(action, -rotations)
            old_state = state_to_features(old_game_state)
            new_state = state_to_features(new_game_state)

            if self.not_fitted:
                pred = 0
            else:
                pred = self.regressor.predict(np.append(new_state, one_hot_action(action)).reshape(1,-1))
            
            X[i,:] = np.append(old_state, one_hot_action(action))
            y[i] = reward + self.discount_factor * pred

        return X,y
