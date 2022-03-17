import numpy as np

from settings import ROWS, COLS

from base_helpers import *

import os
import pickle



def store_model(last_game_state, data, name = "model"):
    if last_game_state['round']%1 == 0 and last_game_state['round'] > 1:
        # dt = datetime.datetime.now()
        # st = dt.strftime('%Y-%m-%d %H:%M:%S')
        # with open(f"models/model_{st}.pt", "wb") as file:
        #     pickle.dump(self.QEstimator, file)
        with open(f"models/" + name + ".pt", "wb") as file:
            pickle.dump(data, file)
        # save some state to automatically check if model has to be overwritten or can be kept
        if not os.path.isfile("models/some_state.pt"):
            with open(f"models/some_state.pt", "wb") as file2:
                pickle.dump(last_game_state, file2)

# check whether the stored model params can be used with state_to_features. If so, loads the .regressor member of the QEstimator class, otherwise overwrites the parameters.
def load_model(self, state_to_features):
    if os.path.isfile("models/model.pt"):
        with open("models/model.pt", "rb") as file:
            try:
                stored_regressor = pickle.load(file)
                with open("models/some_state.pt", "rb") as file2:
                    some_state = pickle.load(file2)
                self.QEstimator.regressor = stored_regressor

                # test two member functions:
                temp = self.QEstimator.estimate(state_to_features(some_state), "UP")
                self.QEstimator.update([(state_to_features(some_state), "UP", state_to_features(some_state), 0),
                                        (state_to_features(some_state), "UP", state_to_features(some_state), 0)])

                # reset after testing update:
                self.QEstimator.regressor = stored_regressor
                print("Using stored regression parameters")

            except ValueError:
                print("Stored regression parameters have another shape, beginning to train new parameters, will overwrite old model after 50 steps.")
                self.QEstimator = QEstimator(learning_rate = 0.1,
                                             discount_factor = 0.8)


''' GRID '''
# returns a list of indices and tuples, describing the fields in the square-neighbourhood of x and y, in a distance measure where a disc of radius 1 is a square of 3x3 fields
def get_neighbourhood(x, y, radius):
    return 0

# returns a list of indices and tuples, describing the fields that can be reached within n_steps or less, excluding the own position (x,y)
# if n_steps == 1, the order is DOWN UP LEFT RIGHT
def get_step_neighbourhood(x, y, n_steps):
    neighb = []
    counter = 0
    for i in range(n_steps+1):
        for j in range(n_steps - i+1):
            if i == 0 and j == 0:
                continue
            # if one is zero, do not count +-zero twice:
            elif i == 0:
                for sign_j in [-1,1]:
                    neighb.append((counter, (x, y+sign_j*j)))
                    counter += 1
            elif j == 0:
                for sign_i in [-1,1]:
                    neighb.append((counter, (x+sign_i*i, y)))
                    counter += 1
            # case none is zero:
            else:
                for sign_i in [-1,1]:
                    for sign_j in [-1,1]:
                        neighb.append((counter, (x+sign_i*i, y+sign_j*j)))
                        counter += 1
    return neighb



def find_next_step_to_closest_coin(field, self_pos, coins):
    if len(coins) == 0:
        return self_pos

    shortest_path_length = float("inf")
    best_coord = (0,0)

    for coin in coins:
        path = find_path(field, self_pos, coin)

        if len(path) == 0:
            return self_pos

        if len(path) < shortest_path_length:
            shortest_path_length = len(path)
            best_coord = path[0]

    return best_coord



''' FEATURES '''
# return an array of length 4, one-hot encoding the direction towards the closest (cityblock metric) reachable coin
def direction_to_best_coin(field, x, y, coins, n_coins_considered):
    coin_positions  = np.zeros(4, dtype=np.int8)
    n_closest_coins = min(len(coins), n_coins_considered)
    coins = coins[np.argpartition(np.array([cityblock_dist((x,y), coin)
                                            for coin in coins]),
                                  n_closest_coins-1)]

    # coord of the step towards the closest coin
    coord_to_closest_coin = find_next_step_to_closest_coin(field, (x,y), coins[:n_closest_coins])

    # exclude the case that the closest coin is on top of the agent
    if not ((coord_to_closest_coin[0] == x) and (coord_to_closest_coin[1] == y)):
        dist = coord_to_closest_coin - [x,y]

        if dist[0] == 0:
            if dist[1] == 1:
                coin_positions[0] = 1
            else:
                coin_positions[1] = 1
        else:
            if dist[0] == 1:
                coin_positions[2] = 1
            else:
                coin_positions[3] = 1

        assert(np.count_nonzero(coin_positions) == 1)

    return coin_positions

# one hot encodes, which tiles reachable in less than n_steps are blocked. If n_steps == 1, the order is UP DOWN LEFT RIGHT
# 1 means blocked.
def blocked_neighbourhood(game_state, x, y, n_steps=1) -> np.array:
    field = game_state["field"]
    others = [(x,y) for (_,_,_,(x,y)) in game_state['others']]
    bombs = [(x,y) for ((x,y),_) in game_state['bombs']]

    neighbourhood = get_step_neighbourhood(x, y, n_steps)
    blocked = np.zeros(len(neighbourhood))

    for index, (x_test, y_test) in neighbourhood:

        if not (x_test<COLS and y_test<ROWS and x_test>=0 and y_test>=0):
            # when the neighbourhood is not in the field, count this as blocked.
            blocked[index] = 1

        else:

            if field[x_test,y_test] != 0:
                blocked[index] = 1
                continue
            else:

                for (_,x_a,y_a) in others:
                    if x_a == x_test and y_a == y_test:
                        blocked[index] = 1
                        break # out of the others loop
                if blocked[index] != 1:
                    for (x_b,y_b) in bombs:
                        if x_b == x_test and y_b == y_test:
                            blocked[index] = 1
                            break # out of the bombs loop

    return blocked

# one-hot encode whether a neighbouring field is affected by a bomb in t_to_explosion steps, including the current positions. If t == None, all times are considered. t_to_explosion steps can also be a tuple of min and max times
# If n_steps == 1, the order is UP DOWN LEFT RIGHT CURRENT_POSITION
def bomb_danger_in_t(game_state, x, y, n_steps=1, t_to_explosion=None) -> np.array:

    neighbourhood = get_step_neighbourhood(x, y, n_steps)
    neighbourhood.append((len(neighbourhood),(x,y)))
    bomb_danger = np.zeros(len(neighbourhood))
    bombs_and_timers = game_state["bombs"]
    field = game_state["field"]

    # set the danger to one outside of the field
    for index, (x_test, y_test) in neighbourhood:
        if not (x_test<COLS and y_test<ROWS and x_test>=0 and y_test>=0):
            bomb_danger[index] = 1

    for ((x_b,y_b),t) in bombs_and_timers:

        good_timer = False

        if t_to_explosion is None:
            good_timer = True
        elif type(t_to_explosion) is tuple:
            if (t >= t_to_explosion[1] and t <= t_to_explosion[0]):
                good_timer = True
        elif t == t_to_explosion:
            good_timer = True

        if not good_timer:
            continue

        for index, (x_test, y_test) in neighbourhood:
            # if danger is already present, skip
            if bomb_danger[index] == 1:
                continue

            dx = x_test - x_b
            dy = y_test - y_b

            if dx == 0 and abs(dy) < 4:
                wall_in_between = False
                # all possible y coord in between:
                for coo in range(min(y_test,y_b) + 1, max(y_test,y_b)):
                    if field[x_test,coo] == -1:
                        wall_in_between = True
                        break
                if not wall_in_between:
                    bomb_danger[index] = 1

            elif dy == 0 and abs(dx) < 4:
                wall_in_between = False
                # all possible x coord in between:
                for coo in range(min(x_test,x_b) + 1, max(x_test,x_b)):
                    if field[coo,y_test] == -1:
                        wall_in_between = True
                        break
                if not wall_in_between:
                    bomb_danger[index] = 1

    return bomb_danger

# one-hot encode whether a bomb with timer = t is located at a field in the neighbourhood. If t == None, all bombs are considered. t can also be a tuple (t_min,t_max)
# If n_steps == 1, the order is UP DOWN LEFT RIGHT CURRENT_POSITION
def neighbouring_bomb_locations_t(game_state, x, y, n_steps=1, t=None) -> np.array:

    neighbourhood = get_step_neighbourhood(x, y, n_steps)
    neighbourhood.append((len(neighbourhood),(x,y)))
    bombs = np.zeros(len(neighbourhood))
    bombs_and_timers = game_state["bombs"]
    field = game_state["field"]

    for ((x_b,y_b),timer) in bombs_and_timers:

        good_timer = False

        if t is None:
            good_timer = True
        elif type(t) is tuple:
            if (timer >= t[1] and timer <= t[0]):
                good_timer = True
        elif t == timer:
            good_timer = True

        if not good_timer:
            continue

        for index, (x_test, y_test) in neighbourhood:
            # outside of the field, there are no bombs
            if not (x_test<COLS and y_test<ROWS and x_test>=0 and y_test>=0):
                bombs[index] = 0

            elif x_test == x_b and y_test == y_b:
                bombs[index] = 1

    return bombs

# nonzero-hot encode a risk score for neighbouring tiles including the current position
# If n_steps == 1, the order is UP DOWN LEFT RIGHT CURRENT_POSITION
def bomb_risk_neighbourhood(game_state, x, y, t=None, n_steps=1) -> np.array:
    field = game_state["field"]
    bombs_and_timers = game_state["bombs"]

    neighbourhood = get_step_neighbourhood(x, y, n_steps)
    neighbourhood.append((len(neighbourhood),(x,y)))
    bomb_risk = np.zeros(len(neighbourhood))

    max_risk = 4 # max dist = 3, max timer = 3, risk = 7 - dist - timer
    def risk(dist, timer):
        return max_risk - dist - timer

    for index, (x_test, y_test) in neighbourhood:
        # outside of the field, the bomb risk is maximal
        if not (x_test<COLS and y_test<ROWS and x_test>=0 and y_test>=0):
            bomb_risk[index] = max_risk

        else:

            for ((x_b,y_b),t) in bombs_and_timers:
                dx = x_test - x_b
                dy = y_test - y_b

                if dx == 0 and abs(dy) < 4:
                    wall_in_between = False
                    # all possible y coord in between:
                    for coo in range(min(y_test,y_b) + 1, max(y_test,y_b)):
                        if field[x_test,coo] == -1:
                            wall_in_between = True
                            break
                    if not wall_in_between:
                        bomb_risk[index] = max(bomb_risk[index], risk(dy,t))

                elif dy == 0 and abs(dx) < 4:
                    wall_in_between = False
                    # all possible x coord in between:
                    for coo in range(min(x_test,x_b) + 1, max(x_test,x_b)):
                        if field[coo,y_test] == -1:
                            wall_in_between = True
                            break
                    if not wall_in_between:
                        bomb_risk[index] = max(bomb_risk[index], risk(dx,t))

    return bomb_risk

# one may implement a faster version here later, taking everything in one loop instead of each function calling its own.
# also indicate whether the agent is currently standing on a bomb
def bomb_feature_set(game_state, x, y) -> np.array:
    bomb_features = np.zeros(1)

    return bomb_features

def explosion_neighbourhood(game_state, x, y, n_steps=1) -> np.array:

    explosion_positions = game_state["explosion_map"]
    neighbourhood = get_step_neighbourhood(x, y, n_steps)
    explosion = np.zeros(len(neighbourhood))

    for index, (x_test, y_test) in neighbourhood:

        if not (x_test<COLS and y_test<ROWS and x_test>=0 and y_test>=0):
            # when the neighbourhood is not in the field, count this as explosion.
            explosion[index] = 1

        elif explosion_positions[x_test, y_test] == 2:
            # if an explosion is valid for one step only, agent may step onto it without effect.
            explosion[index] = 1

    return explosion

# one hot encodes, which tiles reachable in less than n_steps are blocked. If n_steps == 1, the order is UP DOWN LEFT RIGHT
def crates_in_neighbourhood():

    return 0

def valid_actions(game_state) -> np.array:

    ordered_actions = np.array(["UP", "DOWN", "LEFT", "RIGHT"])
    x, y = game_state["self"][3][0], game_state["self"][3][1]

    blocked = blocked_neighbourhood(game_state, x, y, 1)
    if game_state["self"][2]:
        return np.append(ordered_actions[np.where(blocked == 0)], np.array(("WAIT", "BOMB")))
    else:
        return np.append(ordered_actions[np.where(blocked == 0)], np.array(("WAIT")))

def death_implying_actions(game_state) -> np.array:

    ordered_actions = np.array(["UP", "DOWN", "LEFT", "RIGHT", "WAIT"])
    x, y = game_state["self"][3][0], game_state["self"][3][1]

    #
    explosions = np.append(explosion_neighbourhood(game_state, x, y, 1), np.array([0]))
    bombs_with_zero_timer = neighbouring_bomb_locations_t(game_state, 0, 1)
    sum = explosions + bombs_with_zero_timer
    return ordered_actions[np.where(sum == 0)]