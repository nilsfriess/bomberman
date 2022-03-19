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
    self.transitions.append((self.old_game_state,
                            last_action,
                            last_game_state,
                            reward_from_events(events)))

    print_progress(self, last_game_state, last_action, events)
    
    # Update the amf tree and throw away the transitions
    self.estimator.update(self.transitions)
    self.transitions = []

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

        n_closest_coins = min(len(game_coins), 10)
        coins = game_coins[np.argpartition(dist_to_coins.ravel(),
                                           n_closest_coins-1)]
        closest_coins = coins[:n_closest_coins]

        #enemies = []
        coord_to_closest_coin = find_next_step_to_assets(field,
                                                         enemies,
                                                         self_pos,
                                                         closest_coins)

        if np.array_equal(new_self_pos, coord_to_closest_coin):
            events.append(MOVED_TOWARDS_COIN)
        else:
            events.append(MOVED_AWAY_FROM_COIN)
                
    # Bomb-related events
    old_self_pos = old_game_state['self'][3]
    new_self_pos = new_game_state['self'][3]
    explosion_map = new_game_state['explosion_map']
    bombs = old_game_state['bombs']
    
    # Check if we avoided an explosion
    if (explosion_map[old_self_pos] > 0) and\
       (explosion_map[new_self_pos] == 0) and\
       (old_self_pos != new_self_pos):
        events.append(AVOIDED_EXPLOSION)

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

    ''' 
    Check if we were previously on the same row/column as the bomb
    but are not anymore, i.e., we took a turn (which is one of the 
    most important bomb-avoiding skills to learn).
    '''
    for (bomb_pos,_) in bombs:
        # Have we been in the same row or column as the bomb?
        if (bomb_pos[0] == old_self_pos[0]) or (bomb_pos[1] == old_self_pos[1]):
            # Are we even near the bomb but not on the bomb?
            dist_to_bomb = cdist([bomb_pos], [old_self_pos], 'cityblock')
            if (dist_to_bomb == 0) or (dist_to_bomb > 3):
                continue
            
            # Are we still in the same row/column as the bomb?
            if not ((bomb_pos[0] == new_self_pos[0]) or (bomb_pos[1] == new_self_pos[1])):
                events.append(ESCAPED_BOMB_BY_TURNING)
                #print("Escaped by turning")
            else:
                # We did not turn
                events.append(DID_NOT_TURN_AFTER_BOMB)
                
    if (self_action == 'BOMB') and (e.INVALID_ACTION not in events):
        # Check if bomb was dropped in corner which is probably a stupid idea
        corners = [(1,1), (1,15), (15,1), (15,15)]
        if new_game_state['self'][3] in corners:
            events.append(BOMB_IN_CORNER)

        bomb_useful = bomb_usefulness(old_game_state)
        bomb_safety, _ = should_drop_bomb(old_game_state)
        
        if bomb_useful == 0:
            # bomb was useless
            events.append(USELESS_BOMB)
        else:
            if bomb_safety != 0:
                events.append(USEFUL_BOMB)

        if bomb_safety == 0:
            events.append(DROPPED_SUICIDE_BOMB)
        elif bomb_safety <= 10:
            events.append(DROPPED_UNSAFE_BOMB)
        else:
            events.append(DROPPED_SAFE_BOMB)
            

    elif (self_action == 'BOMB') and (e.INVALID_ACTION in events):
        events.append(ILLEGAL_BOMB)
            
    # Check if we went in a direction with lower risk
    risk_map = compute_risk_map(old_game_state)

    if (risk_map[old_self_pos] > 0) and (risk_map[new_self_pos] < risk_map[old_self_pos]):
        events.append(DECREASED_RISK)
    if (risk_map[new_self_pos] >= risk_map[old_self_pos]) and (risk_map[old_self_pos] > 0):
        events.append(INCREASED_RISK)

    if (risk_map[old_self_pos] > 0) and (risk_map[new_self_pos] == 0):
        events.append(ESCAPED_RISK)
            
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
    self.turned = 0
    #print(f"Used: epsilon = {self.epsilon:.2f}, alpha = {self.learning_rate:.2f}")
    #self.epsilon = self.initial_epsilon/(1 + 0.002*last_game_state['round'])
    #self.learning_rate = self.initial_learning_rate/(1 + 0.02*last_game_state['round'])
    #self.learning_rate = 0.999*self.learning_rate
    #self.QEstimator.update_learning_rate(self.learning_rate)
    print()
