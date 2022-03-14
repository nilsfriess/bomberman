from collections import deque
from typing import List

import pickle
import datetime
import random

from settings import COLS, ROWS
import events as e
from .callbacks import state_to_features

from .base_helpers import ACTIONS, cityblock_dist, find_path, action_from_direction

from .training_helpers_leif import *
#rom .training_helpers_nils import *

import numpy as np

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # experience buffer
    self.transitions = []

    setup_custom_vars(self)
    
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
    if old_game_state is None:
        return
    
    self.old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)

    additional_events = compute_custom_events(self,old_game_state,
                                              self_action,
                                              new_game_state,
                                              events)

    for e in additional_events:
        events.append(e)
    
    self.transitions.append((self.old_features,
                             self_action,
                             new_features,
                             reward_from_events(events)))

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
                            reward_from_events(events)))
    
    # Update Q by a Gradient Boost step
    self.QEstimator.update(self.transitions)

    # Print stats
    print_progress(self, last_game_state, last_action, events)
        
    # Store the model
    if last_game_state['round']%10 == 0 and last_game_state['round'] > 49:
        dt = datetime.datetime.now()
        st = dt.strftime('%Y-%m-%d %H:%M:%S')
        # with open(f"models/model_{st}.pt", "wb") as file:
        #     pickle.dump(self.QEstimator, file)
        with open(f"models/model.pt", "wb") as file:
            pickle.dump(self.QEstimator.regressor, file)
