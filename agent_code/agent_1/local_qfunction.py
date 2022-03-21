import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from collections import defaultdict

from base_helpers import ACTIONS, one_hot_action

class LocalQEstimator:
    def __init__(self, learning_rate = 0.1, discount_factor = 0.8, num_hard_sep_states = 3, max_depth=8, steps = 2):
        self.discount_factor = discount_factor

        self.num_hard_sep_states = num_hard_sep_states

        # initialize a list of regressors for each action and separated states
        self.regressor = [ []*len(ACTIONS) ]*num_hard_sep_states
        for state_index in range(num_hard_sep_states):
            for action_index in range(len(ACTIONS)):
                self.regressor[state_index][action_index] = GradientBoostingRegressor(warm_start=True,
                                          max_depth=max_depth,
                                          learning_rate = learning_rate,
                                          n_estimators=1)

        self.first_update = True
        self.not_fitted = True

        self.action_to_index = {
            'UP' : 0,
            'RIGHT' : 1,
            'DOWN' : 2,
            'LEFT' : 3,
            'WAIT' : 4,
            'BOMB': 5
            }

        # Number of steps for n-step temporal difference
        self.steps = steps

    def update_learning_rate(self, new_rate):
        self.regressor.learning_rate = new_rate

    def estimate(self, state: np.array, action: str):
        if self.not_fitted:
            return 0.0

        state_index = state[0]
        action_index = self.action_to_index[action]

        return self.regressor[state_index][action_index].predict(state[0:])[0]


    def update(self, transitions):
        # in the first update, calculate the number of features
        if self.first_update:
            self.first_update = False
            self.feature_size = transitions[0][2].size

        X,y = self.temporal_difference(transitions, self.steps)

        if (X is None) or (y is None):
            # Too few transitions recorded
            return

        for state_index in range(num_hard_sep_states):
            for action_index in range(len(ACTIONS)):

                # transform list of 1d arrays to 2d array
                X_state_action = np.array(X[state_index][action_index])
                # transform list of scalars to 1d array
                y_state_action = np.array(y[state_index][action_index])

                self.regressor[state_index][action_index].fit(X_state_action, y_state_action)

                self.regressor[state_index][action_index].n_estimators += 1

        self.not_fitted = False


    def temporal_difference(self, transitions, steps, flag_reduce_steps=True):
        num_estim = len(transitions) - steps # for the n_step scheme
        num_trans = len(transitions)

        # create a structure where we can append data, not knowing the size beforehand, as X[act_ind][state_index].append(features)
        X = [ [[]]*len(ACTIONS) ]*num_hard_sep_states
        y = [ [[]]*len(ACTIONS) ]*num_hard_sep_states

        if not flag_reduce_steps:
            if num_estim <= 0:
                return None, None

        for i in range(num_trans):
            disc_reward = 0
            n_step_TD_Q = 0

            # determine y:
            steps_used = steps
            if i >= num_estim and i < num_trans - 1:
                if not flag_reduce_steps:
                    break
                    # do not use the last transistions then
                steps_used = 1
            for t in range(steps_used):
                (_,_, reward) = transitions[i+t]
                disc_reward += pow(self.discount_factor, t) * reward

            (old_state_n, action_n, _) = transitions[i+steps]
            Q_vals = [self.estimate(old_state_n, action_n) for action in ACTIONS]
            Q_max = max(Q_vals)

            n_step_TD_Q = disc_reward + pow(self.discount_factor, steps) * Q_max

            if i == num_trans - 1:
                (_, _, reward) = transitions[i]
                n_step_TD_Q = reward
            # determine X:
            (old_state, action, _) = transitions[i]
            state_index = old_state[0]
            action_index = self.action_to_index[action]

            X[state_index][action_index].append(old_state)
            y[state_index][action_index].append(n_step_TD_Q)

        return X, y
