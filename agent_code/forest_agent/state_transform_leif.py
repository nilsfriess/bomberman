import numpy as np

from settings import ROWS,COLS

from .helpers_leif import *

def state_to_features(game_state: dict) -> np.array:
    if game_state is None:
        return np.array([])

    own_position    = np.zeros((ROWS, COLS), dtype=np.int8)
    crates_position = np.zeros((ROWS, COLS), dtype=np.int8)
    walls_position  = np.zeros((ROWS, COLS), dtype=np.int8)
    enemy_positions = np.zeros((ROWS, COLS), dtype=np.int8)
    bomb_positions  = np.zeros((ROWS, COLS), dtype=np.int8)

    explosion_positions = (game_state['explosion_map'] > 0).astype(np.int8)
    field = np.array(game_state['field'])
    bombs = [(x,y) for ((x,y),_) in game_state['bombs']]
    timers = [t for ((_,_),t) in game_state['bombs']]
    bombs_and_timers = game_state["bombs"]
    others = [(x,y) for (_,_,_,(x,y)) in game_state['others']]
    (_,_,_, (x,y)) = game_state['self']
    coins = np.array(game_state['coins'])

    coin_positions = direction_to_best_coin(field, x, y, coins, 3)
    blocked = blocked_neighbourhood(field, others, bombs, x, y, 1)
    bomb_risk = bomb_risk_neighbourhood(field, bombs_and_timers, x, y, 1)
    explosions = explosion_neighbourhood(x, y, explosion_positions, 1)


    features = np.concatenate([
        #coin_positions,
        bomb_risk,
        explosions,
        blocked
    ]).astype(np.int8)

    return features

def train_act(self, game_state: dict) -> str:
    # tweak probability to dodge a bomb:
    NEW_ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
    if train_act.counter != 0:

        train_act.counter += 1
        if train_act.counter == 5:
            train_act.counter = 0

        bomb_risk = bomb_risk_neighbourhood(np.array(game_state["field"]), game_state["bombs"], game_state["self"][3][0], game_state["self"][3][1], 1)[:-1]

        blocked = blocked_neighbourhood(np.array(game_state["field"]), [(x,y) for (_,_,_,(x,y)) in game_state['others']], [(x,y) for ((x,y),_) in game_state['bombs']], game_state["self"][3][0], game_state["self"][3][1], 1)

        return NEW_ACTIONS[np.argmin(bomb_risk + 10*blocked)]

    # if np.random.uniform() < 0.01:
    #     train_act.counter = 1
    #     return "BOMB"

    if np.random.uniform() < 1-self.initial_epsilon:
        state = state_to_features(game_state)
        av = np.array([self.QEstimator.estimate(state, action) for action in ACTIONS])
        best_action = ACTIONS[np.argmax(av)]
        return best_action

    else:
        action = np.random.choice(len(ACTIONS))
        return ACTIONS[action]

train_act.counter = 0
