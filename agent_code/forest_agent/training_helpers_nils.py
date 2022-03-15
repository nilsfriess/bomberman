from typing import List

import events as e

from .state_transform_nils import state_to_features

import numpy as np

MOVED_TOWARDS_COIN = 'MOVED_TOWARDS_COIN'
MOVED_AWAY_FROM_COIN = 'MOVED_AWAY_FROM_COIN'
VALID_ACTION = 'VALID_ACTION'
NO_COIN_COLLECTED = 'NO_COIN_COLLECTED'

STAYED_IN_RISK_REGION = 'STAYED_IN_RISK_REGION'
MOVED_AWAY_FROM_RISK = 'MOVED_AWAY_FROM_RISK'

MOVED_AWAY_FROM_BOMB = 'DID_MOVE_AWAY_FROM_BOMB'
MOVED_NOT_AWAY_FROM_BOMB = 'DID_NOT_MOVE_AWAY_FROM_BOMB'

USEFUL_DIRECTION = 'USEFUL_DIRECTION'
NOT_USEFUL_DIRECTION = 'NOT_USEFUL_DIRECTION'

AVOIDED_BOMB = 'AVOIDED_BOMB'

DID_NOT_SURVIVE = 'DID_NOT_SURVIVE'

BOMB_IN_CORNER = 'BOMB_IN_CORNER'

USEFUL_BOMB = 'USEFUL_BOMB'
USELESS_BOMB = 'USELESS_BOMB'

WAITED_ON_BOMB = 'WAITED_ON_BOMB'

def setup_custom_vars(self):
    self.invalid = 0
    self.moved_away = 0
    self.avoided_bomb = 0
    self.bombs = 0
    self.killed_self = 0

def reward_from_events(events: List[str]) -> int:
    # if (DID_MOVE_AWAY_FROM_BOMB in events) or\
    #    (DID_NOT_MOVE_AWAY_FROM_BOMB in events) or\
    #    (USEFUL_DIRECTION in events) or\
    #    (NOT_USEFUL_DIRECTION in events):
    #     print("Custom event")
    
    game_rewards = {
        e.KILLED_OPPONENT: 500,
        e.COIN_COLLECTED: 100,
        e.CRATE_DESTROYED: 50,
        e.INVALID_ACTION: -10,
        e.KILLED_SELF: -1000,
        VALID_ACTION: -1,
        MOVED_AWAY_FROM_BOMB: 50,
        MOVED_NOT_AWAY_FROM_BOMB: -80,
        MOVED_TOWARDS_COIN: 1,
        MOVED_AWAY_FROM_COIN: -2,
        BOMB_IN_CORNER: -50,
        USELESS_BOMB: -200,
        USEFUL_BOMB: 10,
        WAITED_ON_BOMB: -100
        # DID_NOT_SURVIVE: -1000,
        # e.SURVIVED_ROUND: 2000
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    # self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def compute_custom_events(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):

    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)
    
    new_events = []

    if self_action == 'WAIT':
        new_events.append(e.INVALID_ACTION)

    if e.INVALID_ACTION in events:
        self.invalid += 1

    if self_action == 'BOMB' and (e.INVALID_ACTION not in events):
        self.bombs += 1

    corners = [(1,1), (1,15), (15,1), (15,15)]
    if (self_action == 'BOMB'):
        if new_game_state['self'][3] in corners:
            new_events.append(BOMB_IN_CORNER)
        
    old_self_pos = old_game_state['self'][3]
    new_self_pos = new_game_state['self'][3]
    explosion_map = new_game_state['explosion_map']
    
    if (explosion_map[old_self_pos] > 0) and\
       (explosion_map[new_self_pos] == 0) and\
       (old_self_pos != new_self_pos):
        new_events.append(AVOIDED_BOMB)
        self.avoided_bomb += 1
        
    return new_events

def print_progress(self, last_game_state, last_action, events):
    if e.KILLED_SELF in events:
        self.killed_self += 1
        
    # After the gradient boost update, discard the transitions
    self.transitions = []
 
    print(f"Coins collected: {last_game_state['self'][1]}")
    print(f"Invalid or waited: {self.invalid / last_game_state['step'] * 100:.0f}%")
    self.invalid = 0
    print(f"Planted {self.bombs} bombs, killed itself {self.killed_self} times")
    self.bombs = 0
    self.killed_self = 0
    print(f"Moved away from bomb {self.moved_away} times, avoided {self.avoided_bomb} times")
    self.avoided_bomb = 0
    self.moved_away = 0
    print(f"Used: epsilon = {self.epsilon:.2f}, alpha = {self.learning_rate:.2f}")
    self.epsilon = self.initial_epsilon/(1 + 0.04*last_game_state['round'])
    self.learning_rate = self.initial_learning_rate/(1 + 0.02*last_game_state['round'])
    self.QEstimator.update_learning_rate(self.learning_rate)
    print()
