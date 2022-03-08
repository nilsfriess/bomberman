import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from collections import defaultdict

class ActionEstimator:
    def __init__(self, learning_rate):
        self.regressors = []
        self.learning_rate = learning_rate

    def estimate(self, state):
        res = 0.0
        for regressor in self.regressors:
            res += alpha * regressor.predict(state)

        return res

class QEstimator:
    def __init__(self, default_action, learning_rate):
        self.action_estimators = dict()

        self.learning_rate = learning_rate
    
    def estimate(self, state: np.array, action: str):
        if action not in self.action_estimators:
            return 0.0

        return self.estimators[action].estimate(state)        
    
    def update(self, transitions):
        print(len(transitions))
