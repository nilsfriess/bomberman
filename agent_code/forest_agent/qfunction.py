import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from collections import defaultdict

from .helpers import ACTIONS, one_hot_action

class QEstimator:
    def __init__(self, learning_rate, discount_factor):
        self.discount_factor = discount_factor

        self.regressor = GradientBoostingRegressor(warm_start=True,
                                                   max_depth=3,
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

        a = one_hot_action(action)
        
        X = np.append(state, a).reshape(1,-1)
        return self.regressor.predict(X)[0]
    
    def update(self, transitions):
        if self.first_update:
            self.first_update = False
            self.feature_size = transitions[0][2].size
        X = np.empty((len(transitions)-1, self.feature_size + 6)) # 6 for one-hot-encoded action
        y = np.empty((len(transitions)-1, 1))

        for i in range(1, len(transitions)):
            (old_state, action, new_state, rewards) = transitions[i]
            
            accum = 0
            for t in range(self.steps):
                if i+t < len(transitions):
                    (_,_,_, reward) = transitions[i+t]
                    accum += pow(self.discount_factor, t) * reward

            if i+self.steps > len(transitions) - 1:
                (_,_,new_state,_) = transitions[i]
            else:
                (_,_,new_state,_) = transitions[i+self.steps]
            Q_vals = [self.estimate(new_state, action) for action in ACTIONS]
            Q_max = max(Q_vals)
            
            # Q_estimate = self.estimate(old_state, action)

            '''
            The feature for the regression problem is a vector of the state
            and the index of the action that was taken. The response is a n-step
            temporal difference estimation of the expected return (see p.159f.
            in the lecture notes) minus the current guess of Q (i.e., Y is the
            current residual).
            '''
            X[i-1,:] = np.append(old_state, one_hot_action(action))
            y[i-1] = accum + pow(self.discount_factor, self.steps) * Q_max

        self.regressor.fit(X, y.ravel())
        self.regressor.n_estimators += 1

        self.regressor.max_depth = 3
        self.not_fitted = False

            
