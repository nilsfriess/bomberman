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
    self.bombs = 0
    self.killed_self = 0
    self.useless_bombs = 0

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
    
    total_reward = 0
    for _,_,_,reward in self.transitions:
        total_reward += reward

    print_progress(self, last_game_state, last_action, events, total_reward)
    
    # Update the qtable and throw away the transitions
    if len(self.transitions) >= 3:
        # Only train with sufficiently many transitions
        self.estimator.update(self.transitions)
        self.epsilon = max(0.09, self.initial_epsilon / (1 + 0.03*last_game_state['round']))

        if len(self.transitions) >= 150:
            self.action_filter_prob = self.initial_action_filter_prop / (1 + 0.001*last_game_state['round'])

        self.transitions = []
            
    if last_game_state['round'] % 50 == 0: # Save every 50th model
        dt = datetime.datetime.now()
        st = dt.strftime('%Y-%m-%d %H:%M:%S')
        with open(f"models/model_{st}.pt", "wb") as file:
            dump(self.estimator, file)

def compute_custom_events(self, old_game_state, old_features, self_action, new_game_state, events):
    if self_action == 'WAIT':
        events.append(e.INVALID_ACTION)

    if (e.INVALID_ACTION not in events) and (self_action != 'WAIT'):
        events.append(VALID_ACTION)

    old_self_pos = old_game_state['self'][3]
    new_self_pos = new_game_state['self'][3]
    explosion_map = new_game_state['explosion_map']
    bombs = old_game_state['bombs']

        
    # Check if we walked towards target (only if we are not currently trying to escape bomb)
    risk_map = compute_risk_map(old_game_state)
    x,y = old_self_pos
    own_risk = risk_map[(x,y)]
    
    if own_risk == 0:
        target_direction = old_features[0][:4]
        target_action = action_from_direction(target_direction)

        if (VALID_ACTION in events) and (target_action == self_action):
            events.append(WALKED_TOWARDS_TARGET)
        if target_action != self_action:
            events.append(WALKED_AWAY_FROM_TARGET)
                
    # Bomb-related events                
    if (self_action == 'BOMB') and (e.INVALID_ACTION not in events):
        n_destroyable_crates, n_destroyable_enemies = bomb_usefulness(old_game_state)
        
        if n_destroyable_crates + n_destroyable_enemies == 0:
            events.append(USELESS_BOMB)
        else:
            events.append(USEFUL_BOMB)
            
    # Check if we went in a direction with lower risk
    lower_risk_directions = np.zeros((4,))

    sign = lambda x : 0 if x < 0 else 1

    lower_risk_directions[0] = sign(own_risk - risk_map[(x,y-1)])
    lower_risk_directions[1] = sign(own_risk - risk_map[(x-1,y)])
    lower_risk_directions[2] = sign(own_risk - risk_map[(x,y+1)])
    lower_risk_directions[3] = sign(own_risk - risk_map[(x+1,y)])

    directions = ['UP', 'LEFT', 'DOWN', 'RIGHT']

    if own_risk > 0:
        if self_action == 'WAIT':
            events.append(WAITED_IN_RISK)
        
        if np.any(lower_risk_directions == 1): # If there is a direction with lower risk
            for k, d in enumerate(directions):
                # Check if we walked into a direction with lower risk
                if (lower_risk_directions[k] == 1) and (self_action == d) and (e.INVALID_ACTION not in events):
                    events.append(DECREASED_RISK)
                    break
            if DECREASED_RISK not in events:
                events.append(INCREASED_RISK)

        neighbors =  [(x+1,y), (x-1,y), (x,y-1), (x,y+1)]
        directions = ['RIGHT', 'LEFT', 'UP', 'DOWN']

        for k, neighbor in enumerate(neighbors):
            if (risk_map[neighbor] == 0) and (self_action == directions[k]):
                events.append(TOOK_ZERO_RISK_DIRECTION)

        # Check if we did not take zero risk direction, even if one was available
        if TOOK_ZERO_RISK_DIRECTION not in events:
            for neighbor in neighbors:
                if risk_map[neighbor] == 0:
                    events.append(DID_NOT_TAKE_ZERO_RISK_DIRECTION)

    else: # Our risk is zero
        # Check if we actively walked into a risk region
        if self_action != 'BOMB':
            if (risk_map[old_self_pos] == 0) and (risk_map[new_self_pos] > 0):
                events.append(INCREASED_RISK)

    ''' Update counters ''' 
    if (e.INVALID_ACTION in events):
        self.invalid += 1
        
    if self_action == 'BOMB' and (e.INVALID_ACTION not in events):
        self.bombs += 1

    if USELESS_BOMB in events:
        self.useless_bombs += 1
        
def print_progress(self, last_game_state, last_action, events, total_reward):
    if e.KILLED_SELF in events:
        self.killed_self += 1

    summary = ""
        
    summary += f"Survived {last_game_state['step']} steps (killed itself: {self.killed_self > 0})\n"
    summary += f"Total reward: {total_reward}\n"
    summary += f"Total points: {last_game_state['self'][1]}\n"
    summary += f"Invalid or waited: {self.invalid / last_game_state['step'] * 100:.0f}%\n"
    if self.bombs > 0:
        summary += f"Planted {self.bombs} bombs, {self.useless_bombs / self.bombs * 100:.0f}% useless\n"
    else:
        summary += "Planted 0 bombs\n"
    summary += f"Parameters: epsilon = {self.epsilon:.2f}, alpha = {self.learning_rate:.2f}, filter = {self.action_filter_prob*100 :.2f}%\n"

    min_trained = min(self.estimator.table.values(), key=lambda x: x[0])[0]
    summary += f"Entries in QTable: {len(self.estimator.table)}\n"
    summary += f"Every entry trained at least {min_trained} times\n"
        
    
    self.invalid = 0
    self.bombs = 0
    self.killed_self = 0
    self.useless_bombs = 0
    
    print(summary)
