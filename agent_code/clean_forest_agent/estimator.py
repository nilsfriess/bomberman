import numpy as np

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor

from .base_helpers import ACTIONS

from .state_action_helpers import random_action, one_hot_action, rotate_game_to_upper_left, rotate_action, rotate_game_state, mirror_game_state_lr
from .state_transform import state_to_features

class GBTEstimator:
    def __init__(self, learning_rate, discount_factor):
        self.discount_factor = discount_factor

        self.regressors = dict()
        for action in ACTIONS:
            self.regressors[action] = GradientBoostingRegressor(warm_start=True,
                                                   max_depth=3,
                                                   learning_rate = learning_rate,
                                                   n_estimators=1)

        self.first_update = True
        self.not_fitted   = True

        self.mirror = False

    def update_learning_rate(self, new_rate):
        pass
        # self.regressor.learning_rate = new_rate

    ''' If action is None, just return the best action and corresponding value '''
    def estimate(self, game_state: dict, rotate = True):
        if self.not_fitted:
            return np.zeros((len(ACTIONS),))

        if rotate:
            game_state, rotations = rotate_game_to_upper_left(game_state)
            
        state = state_to_features(game_state)

        qvalues = [self.regressors[action].predict(np.append(state,
                                                             one_hot_action(action)
                                                             ).reshape(1,-1))[0] for action in ACTIONS]

        if rotate:
            rot_actions = [rotate_action(action, rotations) for action in ACTIONS]

            rot_qvals = np.zeros((len(ACTIONS),))
            for i, action in enumerate(ACTIONS):
                index_in_rot = rot_actions.index(action)
                rot_qvals[index_in_rot] = qvalues[i]
            qvalues = rot_qvals
                
        return qvalues

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

        # Transform transitions by mirroring every state and train again
        if self.mirror:
            for i, (old_game_state, action, new_game_state, reward) in enumerate(transitions):
                mir_old_game_state = mirror_game_state_lr(old_game_state)
                mir_new_game_state = mirror_game_state_lr(new_game_state)

                if action == 'UP':
                    mir_action = 'DOWN'
                elif action == 'DOWN':
                    mir_action = 'UP'
                else:
                    mir_action = action

                transitions[i] = (mir_old_game_state, mir_action, mir_new_game_state, reward)

        X,y = self.qlearning(transitions)

        self.regressor.fit(X, y)
        self.regressor.n_estimators += 1
                
        self.not_fitted = False

        self.report_feature_importance()

    def qlearning(self, transitions):
        num_trans = len(transitions)
        
        X = np.empty((num_trans, self.feature_size + len(ACTIONS)))
        y = np.empty((num_trans,))

        for i, (old_game_state, action, new_game_state, reward) in enumerate(transitions):            
            old_game_state, rotations = rotate_game_to_upper_left(old_game_state)
            new_game_state, _ = rotate_game_to_upper_left(new_game_state)

            
            old_state = state_to_features(old_game_state)
            new_state = state_to_features(new_game_state)

            action = rotate_action(action, rotations)
            
            if self.not_fitted:
                qvalues = 0
            else:
                qvalues = [self.regressor.predict(np.append(new_state, one_hot_action(action)).reshape(1,-1))
                           for action in ACTIONS]
                
            X[i,:] = np.append(old_state, one_hot_action(action))
            y[i] = reward + self.discount_factor * np.amax(qvalues)

        return X,y
