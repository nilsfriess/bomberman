import numpy as np

from settings import ROWS,COLS

from .helpers_leif_local import *

from time import sleep

def state_to_features(game_state: dict) -> np.array:
    if game_state is None:
        return np.array([])

    n_steps_bombs = 3
    n_steps_crates = 4
    (x,y) = game_state["self"][3]


    # OWN POSITION

    # integer, indicating the position relative to the intermediate walls, is
    #  2 if walls are up and down
    #  1 if walls are left and right
    #  0 else (walls are only blocking if they are edges)
    # this feature is used to decide which regression forest is used for the game_state, the rest of the features are all depending on it. also the size of the feature vector and the meaning of its entries may vary among different values of state_index
    state_index = get_is_constrained(game_state)

    # one-hot-encoding wheter the agent stands next to an edge
    edges = is_at_edge(x, y, state_index)


    # array of length 5, with the first 4 entries one-hot encoding the direction towards the closest (cityblock metric) reachable coin and the last measuring the distance to this coin (higher values correspond to a closer agent)
    coin_positions = direction_to_best_coin(game_state)


    # one-hot encode whether a neighbouring field is affected by a bomb with timer greater than zero (these fields are forbidden anyway, the agent does not need to know about them), including the current position. The last entry quantifies the minimal timer of the bombs with effect on the neighbourhood
    danger = bomb_danger(game_state, get_step_neighbourhood(x, y, n_steps_bombs, state_index), n_steps_bombs)


    # one hot encodes, which tiles reachable in less than or in n_steps are crates.
    crates = crates_in_neighbourhood(game_state, get_step_neighbourhood(x, y, n_steps_crates, state_index))


    # the zeroth entry of this vector is used to decide which regressor is called, it should always be state_index
    features = np.concatenate([
    np.array([state_index]),
    coin_positions,
    danger,
    crates,
    edges
    ]).astype(np.int8)

    return features


def train_act(self, game_state: dict) -> str:

    #################
    # tweak probability to dodge a bomb:
    NEW_ACTIONS = np.array(["UP", "DOWN", "LEFT", "RIGHT"])
    if train_act.counter > 1:

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

    if train_act.counter == 1:
        train_act.counter += 1
        death_actions = death_implying_actions(game_state)
        VALID_ACTIONS = np.setdiff1d(valid_actions(game_state), death_actions)

        #VALID_ACTIONS = np.setdiff1d(VALID_ACTIONS, np.array(("BOMB")))

        if len(VALID_ACTIONS) > 0:
            action = np.random.choice(len(VALID_ACTIONS))
            return VALID_ACTIONS[action]
        else:
            return "WAIT"

    if np.random.uniform() < self.show_dodging and game_state["self"][2]:
        if np.intersect1d(np.array(["BOMB"]), death_implying_actions(game_state)).shape[0] == 0:
            train_act.counter = 1
            self.count_show += 1
            return "BOMB"
    ##################


    ##################
    # act:
    VALID_ACTIONS = possible_actions(game_state)

    VALID_ACTIONS = np.setdiff1d(VALID_ACTIONS, np.array(("BOMB")))
    #VALID_ACTIONS = np.array(["UP", "DOWN", "LEFT", "RIGHT"])

    if VALID_ACTIONS.shape[0] > 0:
        if np.random.uniform() < 1-self.epsilon:
            state = state_to_features(game_state)
            av = np.array([self.QEstimator.estimate(state, action) for action in VALID_ACTIONS])
            best_action = VALID_ACTIONS[np.argmax(av)]
        else:
            action = np.random.choice(len(VALID_ACTIONS))
            return VALID_ACTIONS[action]

    elif VALID_ACTIONS.shape[0] == 1:
        best_action = VALID_ACTIONS[0]

    else:
        best_action = "WAIT"

    return best_action


train_act.counter = 0
