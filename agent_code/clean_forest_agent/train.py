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
    
    # Update the gb tree and throw away the transitions
    if len(self.transitions) > 10:
        # Only train with sufficiently many transitions
        self.estimator.update(self.transitions)

        # self.learning_rate = self.initial_learning_rate / (1 + 0.01*last_game_state['round'])
        self.epsilon = self.initial_epsilon / (1 + 0.03*last_game_state['round'])

        self.estimator.regressor.learning_rate = self.learning_rate

        
        #self.action_filter_prob = self.initial_action_filter_prop / (1 + 0.0008*last_game_state['round'])
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

        # n_escape_squares, _ = should_drop_bomb(old_game_state)

        # if n_escape_squares == 0:
        #     bomb_safety = -1
        # elif n_escape_squares < 8:
        #     bomb_safety = 0
        # else:
        #     bomb_safety = 1

        # if bomb_safety == -1:
        #     events.append(DROPPED_SUICIDE_BOMB)
        # elif bomb_safety == 0:
        #     events.append(DROPPED_UNSAFE_BOMB)
        # else:
        #     events.append(DROPPED_SAFE_BOMB)
            
    # Check if we went in a direction with lower risk
    risk_map = compute_risk_map(old_game_state)
    x,y = old_self_pos
    own_risk = risk_map[(x,y)]
    
    risk_differences = np.zeros((4,))

    def sign(x):
        if x == 0:
            return 0
        else:
            return -1 if x < 0 else 1

    risk_differences[0] = sign(own_risk - risk_map[(x+1,y)]) #right
    risk_differences[1] = sign(own_risk - risk_map[(x-1,y)]) #left
    risk_differences[2] = sign(own_risk - risk_map[(x,y+1)]) #down
    risk_differences[3] = sign(own_risk - risk_map[(x,y-1)]) #up

    directions = ['RIGHT', 'LEFT', 'DOWN', 'UP']

    if own_risk > 0:
        if np.any(risk_differences != -1): # If there is a direction with lower risk
            for k, d in enumerate(directions):
                if (risk_differences[k] != -1) and (self_action == d):
                    events.append(DECREASED_RISK)
                    break
            if DECREASED_RISK not in events:
                events.append(INCREASED_RISK)

    else:
        # Check if we actively walked into a risk region
        if self_action != 'BOMB':
            if (risk_map[old_self_pos] == 0) and (risk_map[new_self_pos] > 0):
                events.append(INCREASED_RISK)
        
    if (e.INVALID_ACTION in events) or (e.INVALID_ACTION in events):
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
    summary += f"Parameters: epsilon = {self.epsilon:.2f}, alpha = {self.learning_rate:.2f}, filter = {self.action_filter_prob*100 :.2f}%"
    
    self.invalid = 0
    self.bombs = 0
    self.killed_self = 0
    self.useless_bombs = 0
    
    print(summary)
