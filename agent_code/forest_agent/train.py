from collections import deque
from typing import List

import pickle
import datetime
import random

from settings import COLS, ROWS
import events as e
from .callbacks import state_to_features

from .helpers import ACTIONS, index_of_action, cityblock_dist, find_path

from main import N_ROUNDS

import numpy as np

MOVED_TOWARDS_COIN = 'MOVED_TOWARDS_COIN'
MOVED_AWAY_FROM_COIN = 'MOVED_AWAY_FROM_COIN'
VALID_ACTION = 'VALID_ACTION'
NO_COIN_COLLECTED = 'NO_COIN_COLLECTED'

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # experience buffer
    self.transitions = []

    self.waited = 0
    self.invalid = 0
    self.moved_towards = 0
    
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

    if e.COIN_COLLECTED not in events:
        events.append(NO_COIN_COLLECTED)

    if e.INVALID_ACTION not in events:
        events.append(VALID_ACTION)

    if e.INVALID_ACTION in events:
        self.invalid += 1
    if e.WAITED in events:
        self.waited += 1

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
    self.transitions.append((self.old_features,
                            last_action,
                            state_to_features(last_game_state),
                            reward_from_events(self, events)))
    
    # Update Q by a Gradient Boost step
    self.QEstimator.update(self.transitions)

    s = 0
    for (_,_,_,reward) in self.transitions:
        s += reward

    # After the gradient boost update, discard the transitions
    self.transitions = []

    print(f"Total reward: {s}")
    print(f"Coins collected: {last_game_state['self'][1]}")
    print(f"Waited {self.waited} times")
    self.waited = 0
    print(f"Invalid moves: {self.invalid}")
    self.invalid = 0
    print(f"Moved towards coin {self.moved_towards} times")
    self.moved_towards = 0
    print(f"Survived {last_game_state['step']} steps")
    print()
    
    # Store the model
    if last_game_state['round'] >= N_ROUNDS-1:
        dt = datetime.datetime.now()
        st = dt.strftime('%Y-%m-%d %H:%M:%S')
        with open(f"models/model_{st}.pt", "wb") as file:
            pickle.dump(self.QEstimator, file)


def reward_from_events(self, events: List[str]) -> int:
    game_rewards = {
        e.KILLED_OPPONENT: 100,
        e.COIN_COLLECTED: 60,
        e.CRATE_DESTROYED: 10,
        e.BOMB_DROPPED: -1,
        e.WAITED: -9,
        e.INVALID_ACTION: -10,
        VALID_ACTION: -1
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    # self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
