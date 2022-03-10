from collections import namedtuple, deque

import numpy as np

import pickle
from typing import List

import events as e
from .callbacks import state_to_features

from .lin_q_policy import LinearQPolicy, TrainLinearQPolicy


# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

LEARNING_RATE = 0.01
UPDATE_PERIOD = 400
EXPLORATION_PARAMETER = 0.1
DISCOUNT_FACTOR = 0.9


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    self.trainer = TrainLinearQPolicy(self.params, LEARNING_RATE, UPDATE_PERIOD, EXPLORATION_PARAMETER, DISCOUNT_FACTOR, self.num_features)





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
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    rewards = reward_from_events(self, events) + rewards_custom(old_game_state, new_game_state)

    old_state_features = state_to_features(old_game_state)

    self.trainer.record(old_state_features, self_action, rewards)

    if self.trainer.update_now():
        # return the updated params and also update params stored in trainer.
        self.params = self.trainer.updated_params(state_to_features(new_game_state))


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
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # always update when the game is finished.
    self.params = self.trainer.updated_params(state_to_features(last_game_state))

    # Store the model
    with open("params.pt", "wb") as file:
        pickle.dump(self.params, file)

    if last_game_state["round"]%10 == 0:
        print(self.params)




def reward_from_events(self, events: List[str]) -> float:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 2,
        # e.KILLED_OPPONENT: 5,
        e.WAITED: -0.2,
        e.INVALID_ACTION: -0.2,
        e.GOT_KILLED: -5
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def rewards_custom(old_game_state, new_game_state) -> float:
    reward = 0
    return reward
