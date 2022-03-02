from collections import deque
from typing import List

import pickle
import datetime
import random

from settings import COLS, ROWS
import events as e
from .callbacks import state_to_features, ACTIONS, index_of_actions

import numpy as np

EXP_BUFFER_SIZE = 100
BATCH_SIZE = 10
GAMMA = 0.9
ALPHA = 0.8

MOVED_TOWARDS_COIN = 'MOVED_TOWARDS_COIN'
MOVED_AWAY_FROM_COIN = 'MOVED_AWAY_FROM_COIN'

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # experience buffer
    self.transitions = deque(maxlen=EXP_BUFFER_SIZE)

def cityblock_dist(x,y):
    return abs(x[0]-y[0]) + abs(x[1]-y[1])

def dist_to_closest_coin(self_pos : tuple, coins_pos: list[tuple]) -> float:
    return min([cityblock_dist(self_pos, coin) for coin in coins_pos])

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
        closest_coin_dist_old = float("inf")
    else:
        closest_coin_dist_old = dist_to_closest_coin(old_game_state['self'][3],
                                                     old_game_state['coins'])
    closest_coin_dist_new = dist_to_closest_coin(new_game_state['self'][3],
                                                 new_game_state['coins'])

    if closest_coin_dist_new < closest_coin_dist_old:
        events.append(MOVED_TOWARDS_COIN)
    else:
        events.append(MOVED_AWAY_FROM_COIN)    
    
    # state_to_features is defined in callbacks.py
    self.transitions.append((state_to_features(old_game_state),
                             self_action,
                             state_to_features(new_game_state),
                             reward_from_events(self, events)))

    batchsize = min(BATCH_SIZE, len(self.transitions))
    
    batch = random.sample(self.transitions, batchsize)
    for (old, action, new, reward) in batch:
        Q_old = self.Q[np.array2string(old)][index_of_actions(action)]
        if new is None:
            Q_SARSA = np.amax(self.Q[np.array2string(old)])
        else:
            Q_SARSA = self.Q[np.array2string(new)][index_of_actions(action)]

        Q_new = Q_old + ALPHA*(reward + GAMMA*Q_SARSA - Q_old)

        self.Q[np.array2string(old)][index_of_actions(action)] = Q_new
    


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
    self.transitions.append((state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    dt = datetime.datetime.now()
    st = dt.strftime('%Y-%m-%d %H:%M:%S')
    with open(f"models/model_{st}.pt", "wb") as file:
        pickle.dump(self.Q, file)


def reward_from_events(self, events: List[str]) -> int:
    game_rewards = {
        e.COIN_COLLECTED: 5,
        e.MOVED_LEFT: 1,
        e.MOVED_RIGHT: 1,
        e.MOVED_DOWN: 1,
        e.MOVED_UP: 1,
        e.WAITED: -1,
        e.INVALID_ACTION: -2,
        e.KILLED_SELF: -50,
        MOVED_AWAY_FROM_COIN: -1,
        MOVED_TOWARDS_COIN: 5
        # e.KILLED_OPPONENT: 5,
        # PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    # self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
