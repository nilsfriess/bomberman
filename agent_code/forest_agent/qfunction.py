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
                                                   loss='squared_error',
                                                   n_estimators=1)

        self.first_update = True
        self.not_fitted = True

        # Number of steps for n-step temporal difference
        self.steps = 1
    
    def estimate(self, state: np.array, action: str):
        if self.not_fitted:
            return 0.0
        X = np.append(state, index_of_action(action)).reshape(1,-1)
        return self.regressor.predict(X)[0]
    
    def update(self, transitions):
        if self.first_update:
            self.first_update = False
            self.feature_size = transitions[0][2].size
        X = np.empty((len(transitions)-self.steps, self.feature_size + 1))
        y = np.empty((len(transitions)-self.steps, 1))

        for i in range(len(transitions)):
            (old_state, action, new_state, rewards) = transitions[i]
            if (i == 0) or (i > len(transitions) - self.steps):
                continue
            
            accum = 0
            for t in range(self.steps):
                (_,_,_, reward) = transitions[i+t]
                accum += pow(self.discount_factor, t) * reward

            (_,_,new_state,_) = transitions[i+self.steps-1]
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
            X[i-1,:] = np.append(old_state, index_of_action(action))
            y[i-1] = accum + pow(self.discount_factor, self.steps) * Q_max
            
        self.regressor.fit(X, y.ravel())
        self.regressor.n_estimators += 1 
        self.not_fitted = False

            
