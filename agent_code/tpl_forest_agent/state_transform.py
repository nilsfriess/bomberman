import numpy as np

from settings import ROWS,COLS

from helpers_leif import *

def state_to_features(game_state: dict) -> np.array:
    if game_state is None:
        return np.array([])


    (_,_,_, (x,y)) = game_state['self']
    coins = np.array(game_state['coins'])
    field = np.array(game_state["field"])


    coin_positions = direction_to_best_coin(field, x, y, coins, 3)

    #blocked = blocked_neighbourhood(game_state, x, y, 1)

    bombs = neighbouring_bomb_locations_t(game_state, x, y, 1)
    danger = bomb_danger_in_t(game_state, x, y, 2)



    features = np.concatenate([
    bombs,
    danger
    ]).astype(np.int8)

    return features


def train_act(self, game_state: dict) -> str:
    # tweak probability to dodge a bomb:
    NEW_ACTIONS = np.array(["UP", "DOWN", "LEFT", "RIGHT"])
    if train_act.counter != 0:

        train_act.counter += 1
        if train_act.counter == 5:
            train_act.counter = 0

        bomb_risk = bomb_risk_neighbourhood(game_state, game_state["self"][3][0], game_state["self"][3][1], 1)[:-1]

        blocked = blocked_neighbourhood(game_state, game_state["self"][3][0], game_state["self"][3][1], 1)

        return NEW_ACTIONS[np.argmin(bomb_risk + 10*blocked)]

    if np.random.uniform() < 0.0:
        train_act.counter = 1
        return "BOMB"

    if np.random.uniform() < 1-self.initial_epsilon:

        death_actions = death_implying_actions(game_state)
        VALID_ACTIONS = np.setdiff1d(valid_actions(game_state), death_actions)

        if len(VALID_ACTIONS) > 0:
            state = state_to_features(game_state)

            av = np.array([self.QEstimator.estimate(state, action) for action in VALID_ACTIONS])
            best_action = VALID_ACTIONS[np.argmax(av)]

        elif len(VALID_ACTIONS) == 1:
            best_action = VALID_ACTIONS[0]

        else:
            best_action = "WAIT"

        return best_action

    else:
        action = np.random.choice(len(ACTIONS))
        return ACTIONS[action]

train_act.counter = 0
