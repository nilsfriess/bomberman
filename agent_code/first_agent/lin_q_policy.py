import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
actions = np.array(ACTIONS)

# class used to decide in which way to act
class LinearQPolicy:

    def __init__(self, q_regr_params_in):

        # 2d array with [action_index, feature_index] of size ACTIONS.size x (features.size + 1), accounting for a potential offset.
        self.q_regr_params = q_regr_params_in

    # calculate the q-function for one state, characterised by a 2d array "features" with [state_index, feature_index].
    # returns a 2d array of form [state_index, action_index]
    def regress_q(self, features):
        # append 1 at the end of each feature vector features[state_index], then contract the feature index.
        return np.einsum("sf,af->sa", np.append(features, np.tile(np.array([1]), (features.shape[0],1)), axis = 1), self.q_regr_params)

    # return a 1d array of form [state_index]
    # features: 2d array  with [state_index, feature_index]
    def optimal_action_indices(self, features):
        return np.argmin(self.regress_q(features), axis = 1)

    # return a 1d array of strings describing the best actions of form [state_index]
    # features: 2d array  with [state_index, feature_index]
    def optimal_actions(self, features):
        return actions[self.optimal_action_indices(features)]

    # return the string describing the best action for a single state
    # features_: 1d array  with [feature_index]
    def optimal_action(self, features_):
        return self.optimal_actions(np.array([features_]))[0]


# class inheriting from the policy with additional training infrastructure
class TrainLinearQPolicy(LinearQPolicy):

    def __init__(self, q_regr_params_in, learning_rate_in, update_period_in, exploration_parameter_in, discount_factor_in, num_features):

        # matrix with [action_index, feature_index] of size actions.size x (features.size + 1), accounting for a potential offset.
        self.q_regr_params = q_regr_params_in

        # scalar
        self.learning_rate = learning_rate_in

        # int
        self.update_period = update_period_in

        # int
        self.step_counter = 0

        # scalar between 0 and 1
        self.exploration_parameter = exploration_parameter_in

        # scalar between 0 and 1
        self.discount_factor = discount_factor_in

        # list containing numpy arrays [old_features, action, reward]
        # these have index structures (state_index corresponds to a state recorded at some step):
        #   old_features[state_index, feature_index]
        #   action[state_index]
        #   reward[state_index]
        self.recorded_data = [np.empty((update_period_in, num_features)), np.empty(update_period_in), np.empty(update_period_in)]


    def record(old_features, self_action, reward_for_action):
        self.recorded_data[0][self.step_counter] = old_features
        self.recorded_data[1][self.step_counter] = self_action
        self.recorded_data[2][self.step_counter] = reward_for_action
        self.step_counter = self.step_counter + 1


    # updates the internally stored parameters "self.q_regr_params" and returns them.
    def updated_params(self):

        # if function is called before update_period is reached, shrink data array accordingly and expand it later again (should not happen too often)
        if self.step_counter < self.update_period:
            # only take vali entries, last valid entry is at step_counter - 1
            self.recorded_data[0] = self.recorded_data[0][0:self.step_counter]
            self.recorded_data[1] = self.recorded_data[1][0:self.step_counter]
            self.recorded_data[2] = self.recorded_data[2][0:self.step_counter]

        else if self.step_counter > self.update_period:
            print("In TrainLinearQPolicy::update_params: step_counter is greater than update_period!")

        # use the recorded data to obtain regression data:
        # y: estimated optimal Q function values - structure [state_index]
        # x: features corresponding to y - structure [state_index, feature_index]
        # a: action  - structure [state_index]
        y, x, a = self.temporal_difference()

        # leave recorded_data in a valid state
        if self.step_counter < self.update_period:
            # enlarge the arrays again
            self.recorded_data = [np.empty((update_period_in, num_features)), np.empty(update_period_in), np.empty(update_period_in)]
        # otherwise the arrays still have the length update_period.
        self.step_counter = 0

        # use this regression data to update the parameters:

        # filter the recordings for actions and update the params
        for a_index, action in enumerate(ACTION):
            feat_action, response_action = x[a == action], y[a == action]
            N_batch = response_action.shape[0]
            # contract feature_index, get residuals[state_index]:
            residuals = y - np.einsum("sf,f->s",x,self.q_regr_params[a_index])
            # contract state_index, get correction[feature_index]
            correction = np.einsum("sf,s->f",x,residuals)

            self.q_regr_params[a_index] = self.q_regr_params[a_index] + self.learning_rate / N_batch * correction

        # finally, return the updated params
        return self.q_regr_params

    # acts according to epsilon greedy exploration
    def act(self):
        a=1

    # returns 1d array of length update_period - 1 because reward of 1 future step is needed
    def temporal_difference(self):
        # the last action cannot be used because no future is known: or can it? wann werden nochmal rewards verteilt?
        R = np.roll(self.recorded_data[1], 1)[0:self.recorded_data[1].shape[0] - 1]
