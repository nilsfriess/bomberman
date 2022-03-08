import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from collections import defaultdict

from .helpers import ACTIONS, index_of_action

class QEstimator:
    def __init__(self, learning_rate, discount_factor):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.regressor = GradientBoostingRegressor(warm_start=True,
                                                   learning_rate = self.learning_rate,
                                                   loss='squared_error')

        self.first_update = True
        self.not_fitted = True
    
    def estimate(self, state: np.array, action: str):
        if self.not_fitted:
            return 0.0
        X = np.append(state, index_of_action(action)).reshape(1,-1)
        return self.regressor.predict(X)
    
    def update(self, transitions):
        if self.first_update:
            self.first_update = False
            self.feature_size = transitions[0][2].size
        X = np.empty((len(transitions)-1, self.feature_size + 1))
        y = np.empty((len(transitions)-1, 1))
        for i, (old_state, action, new_state, reward) in enumerate(transitions):
            if (old_state is None) or (new_state is None):
                continue

            # Compute the SARSA estimate
            Q_vals = [self.estimate(new_state, action) for action in ACTIONS]
            Q_max = max(Q_vals)
                    
            Q_estimate = self.estimate(old_state, action)

            '''
            The feature for the regression problem is a vector of the state
            and the index of the action that was taken. The response is a 
            temporal difference estimation of the expected return (see p.159f.
            in the lecture notes) minus the current guess of Q (i.e., Y is the
            current residual).
            '''            
            X[i-1,:] = np.append(old_state, index_of_action(action))
            y[i-1] = (reward + self.discount_factor * Q_max) - Q_estimate
            
        self.regressor.fit(X, y.ravel())
        self.not_fitted = False

            
