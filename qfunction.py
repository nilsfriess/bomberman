import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from collections import defaultdict

from base_helpers import ACTIONS, one_hot_action

class QEstimator:
    def __init__(self, learning_rate, discount_factor):
        self.discount_factor = discount_factor

        self.regressor = GradientBoostingRegressor(warm_start=True,
                                                   max_depth=32,
                                                   learning_rate = learning_rate,
                                                   n_estimators=1)

        self.first_update = True
        self.not_fitted = True

        # Number of steps for n-step temporal difference
        self.steps = 2

    def update_learning_rate(self, new_rate):
        self.regressor.learning_rate = new_rate

    def estimate(self, state: np.array, action: str):
        if self.not_fitted:
            return 0.0

        X = np.append(state, one_hot_action(action)).reshape(1,-1)
        return self.regressor.predict(X)[0]

    def update(self, transitions):
        # in the first update, calculate the number of features
        if self.first_update:
            self.first_update = False
            self.feature_size = transitions[0][2].size

        X,y = self.temporal_difference(transitions, self.steps)

        if (X is None) or (y is None):
            # Too few transitions recorded
            return

        self.regressor.fit(X, y.ravel())
        self.regressor.n_estimators += 1

        #self.regressor.max_depth = 1
        self.not_fitted = False


    def temporal_difference(self, transitions, steps, flag_reduce_steps=True):
        num_estim = len(transitions) - steps # for the n_step scheme
        num_trans = len(transitions)

        if flag_reduce_steps:
            X = np.empty((num_trans, self.feature_size + len(ACTIONS)))
            y = np.empty((num_trans, 1))
        else:
            if num_estim <= 0:
                return None, None
            X = np.empty((num_estim, self.feature_size + len(ACTIONS)))
            y = np.empty((num_estim, 1))


        for i in range(num_estim):
            disc_reward = 0
            for t in range(steps):
                (_,_,_, reward) = transitions[i+t]
                disc_reward += pow(self.discount_factor, t) * reward

            (old_state_n, action_n, _, _) = transitions[i+steps]
            Q_vals = [self.estimate(old_state_n, action_n) for action in ACTIONS]
            Q_max = max(Q_vals)

            (old_state, action, _, _) = transitions[i]

            X[i,:] = np.append(old_state, one_hot_action(action))
            y[i] = disc_reward + pow(self.discount_factor, steps) * Q_max

        if flag_reduce_steps:
            '''
            Use one step TD for all but the last old_state.
            Do this by calling this function with flag_reduce_steps set to False.
            '''
            X[num_estim:num_trans-1,:], y[num_estim:num_trans-1]\
                = self.temporal_difference(transitions[num_estim:num_trans], 1, False)

            # Last transition:
            (old_state, action, _, reward) = transitions[num_trans-1]
            X[num_trans-1,:] = np.append(old_state, one_hot_action(action)) # last old_state
            y[num_trans-1] = reward

        return X, y
