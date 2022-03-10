import numpy as np
from numpy import linalg as NLA
from scipy import linalg as la

# class used to decide in which way to act
class LinearQPolicy:

    def __init__(self, q_regr_params_in):

        # 2d array with [action_index, feature_index] of size acstions.size x features.size
        self.q_regr_params = q_regr_params_in

        # store the actions in the class instead of globally
        self.actions = np.array(['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB'])

    # calculate the q-function for one state, characterised by a 2d array "features" with [state_index, feature_index].
    # returns a 2d array of form [state_index, action_index]
    def regress_q(self, features):
        # append 1 at the end of each feature vector features[state_index], then contract the feature index.
        return np.einsum("sf,af->sa", features, self.q_regr_params)

    # return a 1d array of form [state_index]
    # features: 2d array  with [state_index, feature_index]
    def optimal_action_indices(self, features):
        return np.argmax(self.regress_q(features), axis = 1)

    # return a 1d array of strings describing the best actions of form [state_index]
    # features: 2d array  with [state_index, feature_index]
    def optimal_actions(self, features):
        return self.actions[self.optimal_action_indices(features)]

    # return the string describing the best action for a single state
    # features_: 1d array  with [feature_index]
    def optimal_action(self, features_) -> str:
        return self.optimal_actions(np.array([features_]))[0]


# class inheriting from the policy with additional training infrastructure
class TrainLinearQPolicy(LinearQPolicy):

    def __init__(self, q_regr_params_in, learning_rate_in, update_period_in, exploration_parameter_in, discount_factor_in, num_features):

        # matrix with [action_index, feature_index] of size actions.size x features.size
        self.q_regr_params = q_regr_params_in

        self.actions = np.array(['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB'])

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

        self.num_features = num_features

        # list containing numpy arrays [old_features, action, reward_for_that_action]
        # these have index structures (state_index corresponds to a state recorded at some step):
        #   old_features[state_index, feature_index]
        #   action[state_index]
        #   reward[state_index]
        self.recorded_data = [np.empty((self.update_period, self.num_features)), np.empty(self.update_period, dtype = "U10"), np.empty(self.update_period)]


    def record(self, old_features, self_action, reward_for_action):
        self.recorded_data[0][self.step_counter] = old_features
        self.recorded_data[1][self.step_counter] = self_action
        self.recorded_data[2][self.step_counter] = reward_for_action
        self.step_counter = self.step_counter + 1

    def update_now(self):
        if self.step_counter == self.update_period:
            return True
        else:
            return False

    # updates the internally stored parameters "self.q_regr_params" and returns them.
    def updated_params(self, last_state_features):
        # if nothing was recorded since last update
        if self.step_counter == 0:
            return self.q_regr_params


        # if function is called before update_period is reached, shrink data array accordingly and expand it later again (should not happen too often)
        if self.step_counter < self.update_period:
            # only take valid entries, last valid entry is at step_counter - 1
            self.recorded_data[0] = self.recorded_data[0][0:self.step_counter]
            self.recorded_data[1] = self.recorded_data[1][0:self.step_counter]
            self.recorded_data[2] = self.recorded_data[2][0:self.step_counter]

        elif self.step_counter > self.update_period:
            print("In TrainLinearQPolicy::update_params: step_counter is greater than update_period!")

        # use the recorded data to obtain regression data:
        # y: estimated optimal Q function values - structure [state_index]
        # x: features corresponding to y - structure [state_index, feature_index]
        # a: action  - structure [state_index]
        y, x, a = self.temporal_difference(last_state_features)

        # leave recorded_data in a valid state
        if self.step_counter < self.update_period:
            # enlarge the arrays again
            self.recorded_data = [np.empty((self.update_period, self.num_features)), np.empty(self.update_period, dtype = "U10"), np.empty(self.update_period)]
        # otherwise the arrays still have the length update_period.
        self.step_counter = 0

        # use this regression data to update the parameters:

        # filter the recordings for actions and update the params
        for a_index, action in enumerate(self.actions):
            feat_action = x[a == action]
            response_action = y[a == action]
            N_batch = response_action.shape[0]
            # if this action was not recorded, do not update
            if N_batch == 0:
                continue

            # REGRESSION:
            # gradient descent:
            # # contract feature_index, get residuals[state_index]:
            # residuals = y - np.einsum("sf,f->s",x,self.q_regr_params[a_index])
            # # contract state_index, get correction[feature_index]
            # correction = np.einsum("sf,s->f",x,residuals)
            #
            # self.q_regr_params[a_index] = self.q_regr_params[a_index] + self.learning_rate / N_batch * correction

            # ridge regression:
            tau = 0.01
            # contract feature index:
            Gramian = np.einsum("sf,zf->sz", feat_action, feat_action)
            corrected_gramian = Gramian + tau * np.identity(feat_action.shape[0])
            alpha = la.solve(corrected_gramian, response_action)
            self.q_regr_params[a_index] = (1 - self.learning_rate) * self.q_regr_params[a_index] + self.learning_rate * np.einsum("s,sf->f", alpha, feat_action)


        # finally, return the updated params
        return self.q_regr_params

    # acts according to epsilon greedy exploration
    # features_: 1d array of form [feature_index]
    def act(self, features_) -> str:
        r_num = np.random.rand(1)[0]
        if r_num >= self.exploration_parameter:
            return self.optimal_action(features_)
        # exploration:
        else:
            return np.random.choice(self.actions, p=[.25, .25, .25, .25, .0, .0])
            # actions are 'UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB'

    # returns y, x, a:
    #   y: estimated optimal Q function values - structure [state_index]
    #   x: features corresponding to y - structure [state_index, feature_index]
    #   a: action  - structure [state_index]
    def temporal_difference(self, last_state_features):
        rewards = self.recorded_data[2]
        actions = self.recorded_data[1]

        # write features of state i to position i-1
        features_for_Q = np.roll(self.recorded_data[0], -1)
        # set the last state to the actual last state
        features_for_Q[features_for_Q.shape[0] - 1] = last_state_features
        # TD:
        y = rewards + self.discount_factor * self.maximum_q_values(features_for_Q)

        return y, self.recorded_data[0], actions


    # return a 1d array of form [state_index]
    # features: 2d array  with [state_index, feature_index]
    def maximum_q_values(self, features):
        return np.max(self.regress_q(features), axis = 1)
