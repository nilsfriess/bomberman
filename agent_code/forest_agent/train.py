from collections import deque
from typing import List

import pickle
import datetime
import random

from settings import COLS, ROWS
import events as e
from .callbacks import state_to_features

from .helpers import ACTIONS, index_of_action, cityblock_dist, find_path

import numpy as np

MOVED_TOWARDS_COIN = 'MOVED_TOWARDS_COIN'
MOVED_AWAY_FROM_COIN = 'MOVED_AWAY_FROM_COIN'
VALID_ACTION = 'VALID_ACTION'
NO_COIN_COLLECTED = 'NO_COIN_COLLECTED'
#TOOK_DIRECTION_TO_CLOSEST_COIN = 'TOOK_DIRECTION_TO_CLOSEST_COIN'

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # experience buffer
    self.transitions = []
    
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
    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)
        
    if old_game_state is not None:
        self.transitions.append((old_features,
                                 self_action,
                                 new_features,
                                 reward_from_events(self, events)))

        ''' 
        Check, if we walked in the direction of the closest coin.
        This is done by computing the path from our position in
        `old_game_state` to the closest coin using A*. If our position
        in `new_game_state` is the first coordinate in the computed
        path, then we took the correct direction.

        *TODO*: Compute A* path for all coins, not just the "closest" 
        one according to the cityblock_distance and take the shortest 
        path (because the path to the "closest" according to the
        cityblock distance might be very long, if crates are present 
        on the field).
        '''
        coins = old_game_state['coins']
        if len(coins) > 0:
            self_pos = old_game_state['self'][3]

            shortest_path_length = float("inf")
            for coin in coins:
                path = find_path(old_game_state['field'], self_pos, coin)

                if path.size > 0 and (len(path) < shortest_path_length):
                    shortest_path_length = len(path)
                    best_dir = path[0]
                    
            if np.array_equal(best_dir, new_game_state['self'][3]):
                events.append(MOVED_TOWARDS_COIN)
                print("Moved towards")
            else:
                events.append(MOVED_AWAY_FROM_COIN)
                
    if e.COIN_COLLECTED not in events:
        events.append(NO_COIN_COLLECTED)

    if e.INVALID_ACTION not in events:
        events.append(VALID_ACTION)

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
    self.transitions.append((state_to_features(last_game_state),
                             last_action,
                             None,
                             reward_from_events(self, events)))

    # Update Q by a Gradient Boost step
    self.QEstimator.update(self.transitions)

    s = 0
    for (_,_,_,reward) in self.transitions:
        s += reward

    # After the gradient boost update, discard the transitions
    self.transitions = []

    print(f"Total reward: {s}")
    print(f"Coins left: {len(last_game_state['coins'])}")
    
    # # Store the model
    # dt = datetime.datetime.now()
    # st = dt.strftime('%Y-%m-%d %H:%M:%S')
    # with open(f"models/model_{st}.pt", "wb") as file:
    #     pickle.dump(self.Q, file)


def reward_from_events(self, events: List[str]) -> int:
    game_rewards = {
        e.COIN_COLLECTED: 20,
        NO_COIN_COLLECTED: -2,
        e.WAITED: -20,
        e.INVALID_ACTION: -10,
        e.KILLED_SELF: -500,
        MOVED_AWAY_FROM_COIN: -3,
        MOVED_TOWARDS_COIN: 2,
        VALID_ACTION: -1
        # e.KILLED_OPPONENT: 5,
        # PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    # self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
