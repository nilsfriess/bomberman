import numpy as np

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor

from .base_helpers import ACTIONS

from .state_action_helpers import random_action, one_hot_action
from .state_transform import state_to_features

class GBTEstimator:
    def __init__(self, learning_rate, discount_factor):
        self.discount_factor = discount_factor

        self.regressor = GradientBoostingRegressor(warm_start=True,
                                                   max_depth=3,
                                                   learning_rate = learning_rate,
                                                   n_estimators=1)

        self.first_update = True
        self.not_fitted   = True

        self.print_importance_every = 10
        self.print_cnt = 0

    def update_learning_rate(self, new_rate):
        self.regressor.learning_rate = new_rate

    ''' If action is None, just return the best action and corresponding value '''
    def estimate(self, game_state):
        if self.not_fitted:
            return random_action(allow_bombs = False)
            
        state = state_to_features(game_state)
        qvalues = [self.regressor.predict(np.append(state,
                                                    one_hot_action(action)
                                                    ).reshape(1,-1))[0] for action in ACTIONS]
        best_action = ACTIONS[np.argmax(qvalues)]

        return best_action

    def report_feature_importance(self):
        importances = self.regressor.feature_importances_
        
        idx = 0
        for feature in self.feature_names:
            f_importances = importances[idx:idx+self.feature_names[feature]]

            with np.printoptions(precision=3, suppress=True):
                print(f"Importances of '{feature}': {f_importances}")
            
            idx += self.feature_names[feature]
    
    def update(self, transitions):
        if self.not_fitted:
            first_game_state = transitions[0][0]
            first_transformed_transition, names = state_to_features(first_game_state, with_feature_list=True)
        
            self.feature_size = first_transformed_transition.size

            self.feature_names = names
            
        X,y = self.qlearning(transitions)

        self.regressor.fit(X, y)
        self.regressor.n_estimators += 1
                
        self.not_fitted = False

        self.print_cnt += 1
        if self.print_cnt == self.print_importance_every:
            self.report_feature_importance()
            self.print_cnt = 0

    def qlearning(self, transitions):
        num_trans = len(transitions)
        
        X = np.empty((num_trans-1, self.feature_size + len(ACTIONS)))
        y = np.empty((num_trans-1,))

        for i in range(len(transitions) - 1):
            (now_old_state, now_action, _, now_reward) = transitions[i]
            (_, _, next_new_state, next_reward) = transitions[i+1]
            
            rewards = now_reward + self.discount_factor*next_reward

            if self.not_fitted:
                qvalues = [0]
            else:
                state = state_to_features(next_new_state)
                qvalues = [self.regressor.predict(np.append(state, one_hot_action(a)).reshape(1,-1)) for a in ACTIONS]

            state = state_to_features(now_old_state)
            
            X[i,:] = np.append(state, one_hot_action(now_action))
            y[i] = rewards + self.discount_factor * self.discount_factor * max(qvalues)
            
        return X,y
