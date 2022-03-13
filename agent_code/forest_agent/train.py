from collections import deque
from typing import List

import pickle
import datetime
import random

from settings import COLS, ROWS
import events as e
from .callbacks import state_to_features

from .helpers import ACTIONS, cityblock_dist, find_path, action_from_direction

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

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # experience buffer
    self.transitions = []

    self.invalid = 0
    self.moved_away = 0
    self.avoided_bomb = 0
    self.bombs = 0
    self.killed_self = 0
    
def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)

    # if e.WAITED in events:
    #     events.append(e.INVALID_ACTION)
    
    if e.INVALID_ACTION not in events:
        events.append(VALID_ACTION)

    if e.INVALID_ACTION in events:
        self.invalid += 1

    if self_action == 'BOMB' and (e.INVALID_ACTION not in events):
        self.bombs += 1

    corners = [(1,1), (1,15), (15,1), (15,15)]
    if (self_action == 'BOMB'):
        if new_game_state['self'][3] in corners:
            events.append(BOMB_IN_CORNER)
        
    if len(self.old_features) > 0:
        old_self_pos = old_game_state['self'][3]
        new_self_pos = new_game_state['self'][3]
        explosion_map = new_game_state['explosion_map']
        
        if (explosion_map[old_self_pos] > 0) and\
           (explosion_map[new_self_pos] == 0) and\
           (old_self_pos != new_self_pos):
            events.append(AVOIDED_BOMB)
            self.avoided_bomb += 1

        old_bomb_dist = self.old_features[-1]
        new_bomb_dist = new_features[-1]

        if old_bomb_dist != -1:
            if new_bomb_dist > old_bomb_dist:
                events.append(MOVED_AWAY_FROM_BOMB)
                self.moved_away += 1
            else:
                events.append(MOVED_NOT_AWAY_FROM_BOMB)

        bombs = [pos for (pos, _) in old_game_state['bombs']]
        if (old_self_pos in bombs) and (old_self_pos == new_self_pos):
            print("Waited on bomb")
            events.append(WAITED_ON_BOMB)

        # RISK
        action_risks = self.old_features[-5:-1]
        action_risk_strings = []
        if action_risks[0] == 1:
            action_risk_strings.append('UP')
        if action_risks[1] == 1:
            action_risk_strings.append('DOWN')
        if action_risks[2] == 1:
            action_risk_strings.append('LEFT')
        if action_risks[3] == 1:
            action_risk_strings.append('RIGHT')

        if len(action_risk_strings) > 0:
            if self_action not in action_risk_strings:
                events.append(USEFUL_DIRECTION)
            else:
                events.append(NOT_USEFUL_DIRECTION)

        if self_action == 'BOMB':
            (x,y) = old_game_state['self'][3]
            field = np.array(old_game_state['field'])
            enemies = [(x,y) for (_,_,_,(x,y)) in old_game_state['others']]
            
            bomb_was_useless = True
            for i in [-3,-2,-1,1,2,3]:
                # Look for targets left and right
                coord_on_field = (x+i, y)

                if (x+i < 0)\
                   or (x+i >= field.shape[0]):
                    continue

                if field[coord_on_field] == 1:
                    bomb_was_useless = False
                    break

                if coord_on_field in enemies:
                    bomb_was_useless = False
                    break

                # Look for targets above and below
                coord_on_field = (x, y+i)

                if (y+i < 0)\
                   or (y+i >= field.shape[1]):
                    continue

                if field[coord_on_field] == 1:
                    bomb_was_useless = False
                    break

                if coord_on_field in enemies:
                    bomb_was_useless = False
                    break
                
            if bomb_was_useless:
                events.append(USELESS_BOMB)
            else:
                events.append(USEFUL_BOMB)
                
        # if self_action not in action_risk_strings:
        #     # Action was not risky
        #     print("Not risky action")
        # else:
        #     print("Risky action")
            
        # coin_direction = self.old_features[4 + 17*17:4 + 17*17 + 4]
        # if self_action == action_from_direction(coin_direction):
        #     events.append(MOVED_TOWARDS_COIN)
        # else:
        #     events.append(MOVED_AWAY_FROM_COIN)
        
        # old_risk = self.old_features[-2]
        # new_risk = new_features[-2]

        # if old_risk > 0:
        #     # print(f"Old risk: {old_risk}, new risk: {new_risk}. ", end='')
            
        #     if new_risk >= old_risk:
        #         events.append(STAYED_IN_RISK_REGION)
        #     else:
        #         events.append(MOVED_AWAY_FROM_RISK)
        
        # towards_bomb_coord = self.old_features[-2:]
        # if not np.array_equal(towards_bomb_coord, [-1,-1]):
        #     if np.array_equal(towards_bomb_coord, new_game_state['self'][3]):
        #         events.append(DID_NOT_MOVE_AWAY_FROM_BOMB)
        #     else:
        #         events.append(DID_MOVE_AWAY_FROM_BOMB)

        # bomb_direction = self.old_features[-6:-2]
        # if not np.all(bomb_direction == 1) and not np.all(bomb_direction == 0):
        #     action_towards_bomb = action_from_direction(bomb_direction)

        #     if (action_towards_bomb == 'UP') or (action_towards_bomb == 'DOWN'):
        #         if (self_action == 'LEFT') or (self_action == 'RIGHT'):
        #             events.append(USEFUL_DIRECTION)
        #         else:
        #             events.append(NOT_USEFUL_DIRECTION)
        #     else:
        #         if (self_action == 'UP') or (self_action == 'DOWN'):
        #             events.append(USEFUL_DIRECTION)
        #         else:
        #             events.append(NOT_USEFUL_DIRECTION)

        # if USEFUL_DIRECTION in events:
        #     print("Useful direction")
        # if NOT_USEFUL_DIRECTION in events:
        #     print("Not useful direction")
        # old_bomb_dist = self.old_features[-1]
        # new_bomb_dist = new_features[-1]

        # if (old_bomb_dist >= 0) and (old_bomb_dist < 5) and (new_bomb_dist != -1):
        #     if new_bomb_dist <= old_bomb_dist:
        #         events.append(DID_NOT_MOVE_AWAY_FROM_BOMB)
        #     else:
        #         events.append(DID_MOVE_AWAY_FROM_BOMB)
    
    self.transitions.append((self.old_features,
                             self_action,
                             new_features,
                             reward_from_events(self, events)))

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    if e.SURVIVED_ROUND not in events:
        events.append(DID_NOT_SURVIVE)
    
    self.transitions.append((self.old_features,
                            last_action,
                            state_to_features(last_game_state),
                            reward_from_events(self, events)))
    
    # Update Q by a Gradient Boost step
    self.QEstimator.update(self.transitions)

    s = 0
    for (_,_,_,reward) in self.transitions:
        s += reward

    if e.KILLED_SELF in events:
        self.killed_self += 1
        
    # After the gradient boost update, discard the transitions
    self.transitions = []

    print(f"Total reward: {s}")
    print(f"Coins collected: {last_game_state['self'][1]}")
    print(f"Invalid or waited: {self.invalid / 400.0 * 100:.0f}%")
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

    # Store the model
    if last_game_state['round'] % 100 == 0:
        dt = datetime.datetime.now()
        st = dt.strftime('%Y-%m-%d %H:%M:%S')
        with open(f"models/model_{st}.pt", "wb") as file:
            pickle.dump(self.QEstimator, file)


def reward_from_events(self, events: List[str]) -> int:
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
