import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from collections import defaultdict

from base_helpers import ACTIONS, one_hot_action

"""
Class allowing to use several regression forest to estimate the Q-function. Expects a feature vector where the zeroth entry is an index running from 0 to self.num_hard_sep_states - 1, indicating a characteristic of a state that is supposed to trigger a hard separation of these state-subspaces by using a different regression forest for each of the subspaces. (instead of differentiating them by a feature that has to be categorized by one regressor)
"""

class SubspaceQEstimator:
    def __init__(self, learning_rate = 0.1, discount_factor = 0.8, num_hard_sep_states = 3, max_depth=8, steps = 2):
        self.discount_factor = discount_factor

        self.num_hard_sep_states = num_hard_sep_states

        # initialize a list of regressors for each action and separated states
        self.regressor = [None for y in range(num_hard_sep_states)]
        for state_index in range(num_hard_sep_states):
            self.regressor[state_index] = GradientBoostingRegressor(warm_start=True,
                                        max_depth=max_depth,
                                        learning_rate=learning_rate,
                                        n_estimators=1)

        self.first_update = True
        self.not_fitted = [True for y in range(num_hard_sep_states)]


        # Number of steps for n-step temporal difference
        self.steps = steps

    def update_learning_rate(self, new_rate):
        for state_index in range(self.num_hard_sep_states):
            self.regressor[state_index].learning_rate = new_rate

    def estimate(self, state: np.array, action: str):

        state_index = state[0]
        if self.not_fitted[state_index]:
            return 0.0

        X = np.append(state[0:], one_hot_action(action)).reshape(1,-1)
        return self.regressor[state_index].predict(X)[0]


    def update(self, transitions):
        # in the first update, calculate the number of features
        if self.first_update:
            self.first_update = False
            self.feature_size = transitions[0][0].shape[0]

        X,y = self.temporal_difference(transitions, self.steps)

        if (X is None) or (y is None):
            # Too few transitions recorded
            return

        for state_index in range(self.num_hard_sep_states):

            # transform list of 1d arrays to 2d array
            X_state = np.array(X[state_index])

            # transform list of scalars to 1d array
            y_state = np.array(y[state_index])

            if (X_state is None) or (y_state is None):
                continue
            elif (X_state.shape[0] == 0) or (y_state.shape[0] == 0):
                # Too few transitions recorded
                continue


            self.regressor[state_index].fit(X_state, y_state)

            self.regressor[state_index].n_estimators += 1

            self.not_fitted[state_index] = False


    def temporal_difference(self, transitions, steps, flag_reduce_steps=True):
        num_estim = len(transitions) - steps # for the n_step scheme
        num_trans = len(transitions)

        # create a structure where we can append data, not knowing the size beforehand, as
        # X[act_ind][state_index].append(features),
        # X[act_ind][state_index].append(corr_Q_value)
        X = [[] for z in range(self.num_hard_sep_states)]
        y = [[] for z in range(self.num_hard_sep_states)]

        if not flag_reduce_steps:
            if num_estim <= 0:
                return None, None

        for i in range(num_trans):
            disc_reward = 0
            n_step_TD_Q = 0

            # determine y for all but the last transition:
            if i < num_trans-1:
                steps_used = steps
                if i >= num_estim:
                    steps_used = 1 # only use the next step at the last n transitions
                    if not flag_reduce_steps:
                        break
                        # do not use the last transistions then
                for t in range(steps_used):
                    (_,_, reward) = transitions[i+t]
                    disc_reward += pow(self.discount_factor, t) * reward

                (old_state_n, action_n, _) = transitions[i+steps_used]
                Q_vals = [self.estimate(old_state_n, action_n) for action in ACTIONS]
                Q_max = max(Q_vals)

                n_step_TD_Q = disc_reward + pow(self.discount_factor, steps) * Q_max

            else:
                # if i == num_trans-1
                (_, _, reward) = transitions[i]
                n_step_TD_Q = reward
            # determine X:
            (old_state, action, _) = transitions[i]
            state_index = old_state[0]

            X[state_index].append(np.append(old_state, one_hot_action(action)))
            y[state_index].append(n_step_TD_Q)

        return X, y
