import numpy as np

from settings import ROWS,COLS

from helpers_leif import *

from time import sleep

def state_to_features(game_state: dict) -> np.array:
    if game_state is None:
        return np.array([])


    (_,_,_, (x,y)) = game_state['self']
    coins = np.array(game_state['coins'])
    field = np.array(game_state["field"])


    #coin_positions = direction_to_best_coin(field, x, y, coins, 3)

    #blocked = blocked_neighbourhood(game_state, x, y, 1)

    bombs = neighbouring_bomb_locations_t(game_state, x, y, 2)
    danger = bomb_danger_in_t(game_state, x, y, 2, (1,3))
    crates = crates_in_neighbourhood(game_state, x, y, 4)

    edges = is_at_edge(game_state)

    features = np.concatenate([
    #coin_positions,
    bombs,
    danger,
    crates,
    edges
    ]).astype(np.int8)

    return features


def train_act(self, game_state: dict) -> str:
    #################
    # tweak probability to dodge a bomb:
    NEW_ACTIONS = np.array(["UP", "DOWN", "LEFT", "RIGHT"])
    if train_act.counter != 0:

        train_act.counter += 1
        if train_act.counter == 5:
            train_act.counter = 0

        bomb_risk = bomb_risk_neighbourhood(game_state, game_state["self"][3][0], game_state["self"][3][1], 1)[:-1]

        blocked = blocked_neighbourhood(game_state, game_state["self"][3][0], game_state["self"][3][1], 1)

        best_action = NEW_ACTIONS[np.argmin(bomb_risk + 10*blocked)]

        if best_action is None:
            return "WAIT"
        else:
            return best_action

    if np.random.uniform() < self.show_dodging and game_state["self"][2]:
        train_act.counter = 1
        self.count_show += 1
        return "BOMB"
    ##################


    ##################
    # act:
    death_actions = death_implying_actions(game_state)
    VALID_ACTIONS = np.setdiff1d(valid_actions(game_state), death_actions)

    #VALID_ACTIONS = np.setdiff1d(VALID_ACTIONS, np.array(("BOMB")))

    if len(VALID_ACTIONS) > 0:
        if np.random.uniform() < 1-self.epsilon:
            state = state_to_features(game_state)
            av = np.array([self.QEstimator.estimate(state, action) for action in VALID_ACTIONS])
            best_action = VALID_ACTIONS[np.argmax(av)]
        else:
            action = np.random.choice(len(VALID_ACTIONS))
            return VALID_ACTIONS[action]

    elif len(VALID_ACTIONS) == 1:
        best_action = VALID_ACTIONS[0]

    else:
        best_action = "WAIT"

    return best_action


train_act.counter = 0
