import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from collections import defaultdict

from .helpers import ACTIONS, index_of_action, one_hot_encode_action

# IMPLEMENT CASE WHERE TRANSITIONS ARE SMALLER THAN NUMBER OF STEPS

class QEstimator:
    def __init__(self, learning_rate, discount_factor):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.regressor = GradientBoostingRegressor(warm_start=True,
                                                   learning_rate = self.learning_rate,
                                                   loss='squared_error',
                                                   n_estimators=1)

        self.count_trained_games = 0

        self.first_update = True
        self.not_fitted = True

        # Number of steps for n-step temporal difference
        self.steps = 5

    # return the regressed value of Q(state, action) using the current regression model
    # state is a feature array
    def estimate(self, state: np.array, action: str):
        if self.not_fitted:
            return 0.0
        # the reshape does the same thing as X = np.array([np.append(...)])
        X = np.append(state, one_hot_encode_action(action)).reshape(1,-1)
        return self.regressor.predict(X)[0]

    # updates the internal model used for Q-function estimation
    # transitions: List of tuples (old_features, action to new state, new_features ,rewards for that action)
    def update(self, transitions):
        # in the first update, calculate the number of features
        if self.first_update:
            self.first_update = False
            self.feature_size = transitions[0][2].size

        # n_step_TD can only calculate the y-regression values for n_recorded_tr - n transitions
        # if the function is called in this way, step size is reduced for later states, therefore the X, y have length len(transitions)
        X, y = self.n_step_TD(transitions, self.steps)

        '''
        The feature for the regression problem is a vector of the state
        and the index of the action that was taken. The response y is a n-step
        temporal difference estimation of the Q-function (see p.159f.
        in the lecture notes).
        '''

        self.regressor.fit(X, y.ravel())
        self.count_trained_games += 1
        self.regressor.n_estimators += max(round(self.count_trained_games/1000),2)
        self.not_fitted = False


    # for n_recorded_tr recorded transitions, calculate an estimation for the q-function
    # returns the input X and response y for the regression of the q-function as X, y
    # X is an array of form [state_index, feature_action_index] describing points in state-action ("old_state") space, where the action is one-hot-encoded and appended at the end of the array
    # Y is an array of form [state_index] describing the Q-estimation at these points
    # This can be done for n_recorded_tr - n_steps states because one needs the information of old_state i + n_steps to calculate y[i]
    # If flag_reduce_steps is True, TD with less than n_steps is performed to estimate the Q-function for the later states too. This is important to consider the rewards received at the end of the game
    # for the later transitions, use n_step_TD with only one step:
    # for the last transition, the Q-function of the new state is zero (update is called when the game is over), therefore y is only the reward for the last action
    # set the flag_advanced to True to gradually reduce the steps towards the end
    # transitions: List of tuples (old_features, action to new state, new_features, rewards for that action)

    def n_step_TD(self, transitions, n_steps, flag_reduce_steps = True, flag_advanced = False):
        num_estim = len(transitions) - n_steps # for the n_step scheme
        num_trans = len(transitions)

        if flag_reduce_steps:
            X = np.empty((num_trans, self.feature_size + len(ACTIONS)))
            y = np.empty((num_trans, 1))
        else:
            assert num_estim > 0, "Number of rec transitions is smaller or equal than n_steps, set flag_reduce_steps to true"
            X = np.empty((num_estim, self.feature_size + len(ACTIONS)))
            y = np.empty((num_estim, 1))

        for i in range(num_estim):
            # calculate the discounted reward
            # disc_reward = reward(this_tr) gamma^0 + reward(this_tr + 1) gamma^1 + ... + reward(this_tr + n_steps - 1) gamma ^(n_steps - 1)
            # then, y = disc_reward + gamma^(n_steps) max_{action} Q([x_old, action](this_tr + n_steps))
            disc_reward = 0
            for t in range(n_steps):
                (_,_,_, reward) = transitions[i+t]
                disc_reward += pow(self.discount_factor, t) * reward

            (old_state_n, action_n, _, _) = transitions[i+n_steps]
            Q_vals = [self.estimate(old_state_n, action_n) for action in ACTIONS]
            Q_max = max(Q_vals)

            (old_state, action, _, _) = transitions[i]
            X[i,:] = np.append(old_state, one_hot_encode_action(action))
            y[i] = disc_reward + pow(self.discount_factor, n_steps) * Q_max


        if flag_reduce_steps:
            if flag_advanced:
                assert False, "in QEstimator::late_n_step_TD: advanced option not implemented yet"

            else:
                # use one step TD for all but the last old_state
                # Do this by calling this function with flag_reduce_steps set to False
                X[num_estim:num_trans-1,:], y[num_estim:num_trans-1] = self.n_step_TD(transitions[num_estim:num_trans], 1, False)
                # last transition:
                (old_state, action, _, reward) = transitions[num_trans-1]
                X[num_trans-1,:] = np.append(old_state, one_hot_encode_action(action)) # last old_state
                y[num_trans-1] = reward

        return X, y
