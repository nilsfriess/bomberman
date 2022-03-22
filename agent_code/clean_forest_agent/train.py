from pickle import dump
import datetime

from .rewards import *
import events as e

import numpy as np
from scipy.spatial.distance import cdist

from .state_transform import state_to_features
from .base_helpers import compute_risk_map, find_next_step_to_assets, action_from_direction
from .bomb_helpers import bomb_usefulness, should_drop_bomb

def setup_training(self):
    self.transitions = []

    # Setup counters
    self.invalid = 0
    self.moved_away = 0
    self.avoided_bomb = 0
    self.bombs = 0
    self.killed_self = 0
    self.turned = 0

    self.old_game_state = None
    self.new_game_features = None

def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if old_game_state is None:
        return

    if self.new_game_features is None:
        # In the first step, there is no previous game.
        # Afterwards, the current old_game_state is the new_game_state from the previous round
        self.old_game_features = state_to_features(old_game_state)
        self.new_game_features = state_to_features(new_game_state)
    else:
        self.old_game_features = self.new_game_features
        self.new_game_features = state_to_features(new_game_state)

    self.old_game_state = old_game_state
        
    # Compute custom events and append them to `events`
    compute_custom_events(self,
                          old_game_state,
                          self.old_game_features,
                          self_action,
                          new_game_state,
                          events)

    reward = reward_from_events(events)
    self.transitions.append((self.old_game_features,
                             self_action,
                             self.new_game_features,
                             reward))

def end_of_round(self, last_game_state, last_action, events):
    compute_custom_events(self,
                          self.old_game_state,
                          self.old_game_features,
                          last_action,
                          last_game_state,
                          events)
    
    self.transitions.append((self.old_game_features,
                            last_action,
                            state_to_features(last_game_state),
                            reward_from_events(events)))

    self.learning_rate = self.initial_learning_rate / (1 + 0.01*last_game_state['round'])
    self.epsilon = self.initial_epsilon / (1 + 0.03*last_game_state['round'])

    self.estimator.regressor.learning_rate = self.learning_rate

    self.action_filter_prob = self.initial_action_filter_prop / (1 + 0.005*last_game_state['round'])
    
    total_reward = 0
    for _,_,_,reward in self.transitions:
        total_reward += reward

    print_progress(self, last_game_state, last_action, events)
    print(f"Total reward {total_reward}")
    
    # Update the gb tree and throw away the transitions
    if len(self.transitions) > 1:
        # Only train with sufficiently many transitions
        self.estimator.update(self.transitions)
    self.transitions = []

    if last_game_state['round'] % 50 == 0:
        dt = datetime.datetime.now()
        st = dt.strftime('%Y-%m-%d %H:%M:%S')
        with open(f"models/model_{st}.pt", "wb") as file:
            dump(self.estimator, file)

def compute_custom_events(self, old_game_state, old_features, self_action, new_game_state, events):
    if self_action == 'WAIT':
        events.append(e.INVALID_ACTION)

    if (e.INVALID_ACTION not in events) and (self_action != 'WAIT'):
        events.append(VALID_ACTION)


    # Check if we walked towards target        
    target_direction = old_features[:4]
    target_action = action_from_direction(target_direction)

    if (VALID_ACTION in events) and (target_action == self_action):
        events.append(WALKED_TOWARDS_TARGET)
    else:
        events.append(WALKED_AWAY_FROM_TARGET)
                
    # Bomb-related events
    old_self_pos = old_game_state['self'][3]
    new_self_pos = new_game_state['self'][3]
    explosion_map = new_game_state['explosion_map']
    bombs = old_game_state['bombs']

    # Check if we increased the distance to a bomb if it is near us
    for (bomb_pos,_) in bombs:
        # Are we in the same row or column as the bomb?
        if (bomb_pos[0] == old_self_pos[0]) or (bomb_pos[1] == old_self_pos[1]):
            # Are we near the bomb?
            if cdist([bomb_pos], [old_self_pos], 'cityblock') < 5:
                # Did we move towards the bomb?
                if cdist([bomb_pos], [new_self_pos], 'cityblock') <= cdist([bomb_pos], [old_self_pos], 'cityblock'):
                    events.append(MOVED_TOWARDS_BOMB)
                else:
                    events.append(MOVED_AWAY_FROM_BOMB)
                
    if (self_action == 'BOMB') and (e.INVALID_ACTION not in events):
        # Check if bomb was dropped in corner which is probably a stupid idea
        corners = [(1,1), (1,15), (15,1), (15,15)]
        if new_game_state['self'][3] in corners:
            events.append(BOMB_IN_CORNER)

        n_destroyable_crates, n_destroyable_enemies = bomb_usefulness(old_game_state)
        
        if n_destroyable_crates + n_destroyable_enemies == 0:
            events.append(USELESS_BOMB)
        else:
            # Either destroys crates or enemies
            if n_destroyable_enemies == 0:
                # Bomb destroys some crates, the more, the better
                if n_destroyable_crates < 3:
                    events.append(USEFUL_BOMB)
                else:
                    events.append(VERY_USEFUL_BOMB)
            else:
                # Bomb tries to kill enemy
                events.append(EXTREMELY_USEFUL_BOMB)

        n_escape_squares, _ = should_drop_bomb(old_game_state)

        if n_escape_squares == 0:
            bomb_safety = -1
        elif n_escape_squares < 8:
            bomb_safety = 0
        else:
            bomb_safety = 1

        if bomb_safety == -1:
            events.append(DROPPED_SUICIDE_BOMB)
        elif bomb_safety == 0:
            events.append(DROPPED_UNSAFE_BOMB)
        else:
            events.append(DROPPED_SAFE_BOMB)
            
    # Check if we went in a direction with lower risk
    risk_map = compute_risk_map(old_game_state)

    if (risk_map[old_self_pos] > 0) and (risk_map[new_self_pos] < risk_map[old_self_pos]):
        events.append(DECREASED_RISK)
    if (risk_map[new_self_pos] >= risk_map[old_self_pos]) and (risk_map[old_self_pos] > 0):
        events.append(INCREASED_RISK)

    # Check if we went in the direction with the lowest risk
    risk_factors = np.zeros((5,))

    x,y = old_self_pos    
    risk_factors[0] = risk_map[(x+1,y)]
    risk_factors[1] = risk_map[(x-1,y)]
    risk_factors[2] = risk_map[(x,y+1)]
    risk_factors[3] = risk_map[(x,y-1)]
    risk_factors[4] = risk_map[(x,y)]

    if (risk_map[old_self_pos] > 0) and (risk_map[new_self_pos] == np.amin(risk_factors)):
        # Took direction with lowest risk
        events.append(TOOK_LOWEST_RISK_DIRECTION)

    if (e.INVALID_ACTION in events) or (e.INVALID_ACTION in events):
        self.invalid += 1

    if MOVED_AWAY_FROM_BOMB in events:
        self.moved_away += 1
        
    if self_action == 'BOMB' and (e.INVALID_ACTION not in events):
        self.bombs += 1

    if ESCAPED_BOMB_BY_TURNING in events:
        self.turned += 1

def print_progress(self, last_game_state, last_action, events):
    if e.KILLED_SELF in events:
        self.killed_self += 1

    print(f"Survived {last_game_state['step']} steps")
    print(f"Coins collected: {last_game_state['self'][1]}")
    print(f"Invalid or waited: {self.invalid / last_game_state['step'] * 100:.0f}%")
    self.invalid = 0
    print(f"Planted {self.bombs} bombs, killed itself {self.killed_self} times")
    self.bombs = 0
    self.killed_self = 0
    print(f"Moved away from bomb {self.moved_away} times, avoided {self.avoided_bomb} times")
    self.avoided_bomb = 0
    self.moved_away = 0
    print(f"Turned after bomb {self.turned} times")
    print(f"Action filter probability: {self.action_filter_prob*100 : .2f}%")
    #self.action_filter_prob *= 0.99
    self.turned = 0
    print(f"Used: epsilon = {self.epsilon:.2f}, alpha = {self.learning_rate:.2f}")
    #self.epsilon = self.initial_epsilon/(1 + 0.002*last_game_state['round'])
    #self.learning_rate = self.initial_learning_rate/(1 + 0.02*last_game_state['round'])
    #self.learning_rate = 0.999*self.learning_rate
    #self.QEstimator.update_learning_rate(self.learning_rate)
    print()
