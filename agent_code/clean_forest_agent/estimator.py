import numpy as np

from sklearn.ensemble import GradientBoostingRegressor

from .base_helpers import ACTIONS

from .state_action_helpers import random_action, one_hot_action
from .state_transform import state_to_features

class GBTEstimator:
    def __init__(self, learning_rate, discount_factor):
        self.discount_factor = discount_factor

        self.regressor = GradientBoostingRegressor(warm_start=True,
                                                   max_depth=6,
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
    
    def update(self, transitions):
        if self.not_fitted:
            first_game_state = transitions[0][0]
            first_transformed_transition = first_game_state
        
            self.feature_size = first_transformed_transition.size
            
        X,y = self.qlearning(transitions)
        
        self.regressor.fit(X, y)
        self.regressor.set_params(n_estimators=self.regressor.n_estimators + 1)
                
        self.not_fitted = False

    # 3 step TD
    def qlearning(self, transitions):
        num_trans = len(transitions)
        
        X = np.empty((num_trans-2, self.feature_size + len(ACTIONS)))
        y = np.empty((num_trans-2,))

        for i in range(len(transitions) - 2):
            (now_old_state, now_action, _, now_reward) = transitions[i]
            (_, _, _, next_reward) = transitions[i+1]
            (_, _, next_next_new_state, next_next_reward) = transitions[i+2]
            
            rewards = now_reward + \
                self.discount_factor*next_reward + \
                self.discount_factor**2 * next_next_reward

            if self.not_fitted:
                qvalues = [0]
            else:
                state = next_next_new_state
                qvalues = [self.regressor.predict(np.append(state, one_hot_action(a)).reshape(1,-1)) for a in ACTIONS]

            state = now_old_state
            
            X[i,:] = np.append(state, one_hot_action(now_action))
            y[i] = rewards + self.discount_factor**3 * max(qvalues)
            
        return X,y
