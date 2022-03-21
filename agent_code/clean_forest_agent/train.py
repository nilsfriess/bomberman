from pickle import dump
import datetime

from .rewards import *
import events as e

import numpy as np
from scipy.spatial.distance import cdist

from .base_helpers import compute_risk_map, find_next_step_to_assets
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

def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if old_game_state is None:
        return

    self.old_game_state = old_game_state

    # Compute custom events and append them to `events`
    compute_custom_events(self,
                          old_game_state,
                          self_action,
                          new_game_state,
                          events)

    reward = reward_from_events(events)
    self.transitions.append((old_game_state,
                             self_action,
                             new_game_state,
                             reward))

def end_of_round(self, last_game_state, last_action, events):
    compute_custom_events(self,
                          self.old_game_state,
                          last_action,
                          last_game_state,
                          events)
    
    self.transitions.append((self.old_game_state,
                            last_action,
                            last_game_state,
                            reward_from_events(events)))

    total_reward = 0
    for _,_,_,reward in self.transitions:
        total_reward += reward
        
    print_progress(self, last_game_state, last_action, events)
    print(f"Total reward {total_reward}")
    
    # Update the amf tree and throw away the transitions
    if len(self.transitions) > 1:
        # Only train with sufficiently many transitions
        self.estimator.update(self.transitions)
    self.transitions = []

    if last_game_state['round'] % 50 == 0:
        dt = datetime.datetime.now()
        st = dt.strftime('%Y-%m-%d %H:%M:%S')
        with open(f"models/model_{st}.pt", "wb") as file:
            dump(self.estimator, file)

def compute_custom_events(self, old_game_state, self_action, new_game_state, events):
    if self_action == 'WAIT':
        events.append(e.INVALID_ACTION)

    if (e.INVALID_ACTION not in events) and (self_action != 'WAIT'):
        events.append(VALID_ACTION)
    
    # Find 5 closest coins, where `close` is w.r.t. the cityblock distance
    (_,_,_,self_pos) = old_game_state['self']
    (_,_,_,new_self_pos) = new_game_state['self']
    field = np.array(old_game_state['field'])
    game_coins = np.array(old_game_state['coins'])
    enemies = [(x,y) for (_,_,_,(x,y)) in old_game_state['others']]
    
    if len(game_coins) > 0:
        dist_to_coins = cdist(game_coins, [self_pos], 'cityblock')

        n_closest_coins = min(len(game_coins), 5)
        coins = game_coins[np.argpartition(dist_to_coins.ravel(),
                                           n_closest_coins-1)]
        closest_coins = coins[:n_closest_coins]

        coord_to_closest_coin = find_next_step_to_assets(field,
                                                         [],
                                                         self_pos,
                                                         closest_coins)

        if np.array_equal(new_self_pos, coord_to_closest_coin):
            events.append(MOVED_TOWARDS_COIN)
        else:
            if e.INVALID_ACTION not in events:
                events.append(MOVED_AWAY_FROM_COIN)
                
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

        bomb_useful = bomb_usefulness(old_game_state)
        
        if bomb_useful == 0:
            events.append(USELESS_BOMB)
        elif (bomb_useful > 0) and (bomb_useful < 10):
            events.append(USEFUL_BOMB)
        else:
            # Bomb tries to kill enemy
            events.append(VERY_USEFUL_BOMB)

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

            
    # Update counters
    if AVOIDED_EXPLOSION in events:
        self.avoided_bomb += 1

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
    #print(f"Used: epsilon = {self.epsilon:.2f}, alpha = {self.learning_rate:.2f}")
    #self.epsilon = self.initial_epsilon/(1 + 0.002*last_game_state['round'])
    #self.learning_rate = self.initial_learning_rate/(1 + 0.02*last_game_state['round'])
    #self.learning_rate = 0.999*self.learning_rate
    #self.QEstimator.update_learning_rate(self.learning_rate)
    print()
