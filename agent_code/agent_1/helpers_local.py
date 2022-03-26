import numpy as np

from settings import ROWS, COLS

from .base_helpers import *
from .local_qfunction import *

import os
import pickle



def store_model(last_game_state, data, name = "model"):
    if last_game_state['round']%10 == 0 and last_game_state['round'] > 19:
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
                self.QEstimator.update([(state_to_features(some_state), "UP", 0),
                                        (state_to_features(some_state), "UP", 0)])

                # reset after testing update:
                self.QEstimator.regressor = stored_regressor
                print("Using stored regression parameters")
                #self.logger.info("Loading model from saved state.")

            except ValueError:
                print("Stored regression parameters have another shape. If self.train, beginning to train new parameters, will overwrite old model after 20 rounds.")
                self.QEstimator = SubspaceQEstimator(learning_rate = 0.1,
                                             discount_factor = 0.8)


''' GRID '''
# returns a list of indices and tuples, describing the fields that can be reached within n_steps or less, excluding the own position (x,y) and positions of walls, depending on the state_index, which is determined by get_is_constrained().
# if state_index == None, also walls are included
# if n_steps == 1 and state_index == None, the order is DOWN UP LEFT RIGHT
def get_step_neighbourhood(x, y, n_steps, state_index = None):

    # helper:
    def steps_good(x_step, y_step, state_ind):
        if state_ind is None:
            return True

        if (x_step%2==0 and y_step%2==0):
            return True

        if state_ind == 0:
            if not (x_step%2==1 and y_step%2==1):
                return True
            else:
                return False

        if state_ind == 1:
            if (x_step%2==0) or (x_step%2==1 and y_step%2==1):
                return True
            else:
                return False

        if state_ind == 2:
            if (y_step%2==0) or (x_step%2==1 and y_step%2==1):
                return True
            else:
                return False

    counter = 0
    neighb = []
    for i in range(n_steps+1):
        for j in range(n_steps - i+1):
            if i == 0 and j == 0:
                continue
            elif not steps_good(i,j,state_index):
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

# returns
#  2 if constrained by up down
#  1 if constrained by left right
#  0 if not constrained
def get_is_constrained(game_state) -> int:
    (x,y) = game_state["self"][3]
    if x % 2 == 1:
        if y % 2 == 1:
            # e.g. (1,1) i.e. a corner
            return 0
        else:
            # e.g. (1,2) i.e. between walls at (0,2) and (2,2)
            return 1
    else:
        if y % 2 == 1:
            # e.g. (2,1) i.e. between walls at (2,0) and (2,2)
            return 2
        else:
            # should not happen, generate error
            assert not (x%2==0 and y%2==0), "position should be blocked by wall but is not"




def find_next_step_to_closest_coin(field, self_pos, coins):
    if len(coins) == 0:
        return self_pos, None

    shortest_path_length = float("inf")
    best_coord = (0,0)
    best_coin_coord = None

    for coin in coins:
        path = find_path(field, self_pos, coin)

        if len(path) == 0:
            return self_pos, None

        if len(path) < shortest_path_length:
            shortest_path_length = len(path)
            best_coord = path[0]
            best_coin_coord = coin

    return best_coord, best_coin_coord



''' FEATURES '''
# return an array of length 5, with the first 4 entries one-hot encoding the direction towards the closest (cityblock metric) reachable coin and the last measuring the distance to this coin (higher values correspond to a closer agent)
def direction_to_best_coin(game_state, n_coins_considered = 2):

    (_,_,_, (x,y)) = game_state['self']
    coins = np.array(game_state['coins'])
    field = np.array(game_state["field"])

    coin_positions  = np.zeros(5, dtype=np.int8)
    n_closest_coins = min(len(coins), n_coins_considered)
    coins = coins[np.argpartition(np.array([cityblock_dist((x,y), coin)
                                            for coin in coins]),
                                  n_closest_coins-1)]

    # coord of the step towards the closest coin
    coord_to_closest_coin, coord_coin = find_next_step_to_closest_coin(field, (x,y), coins[:n_closest_coins])

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

    # distance measure:
    distance_info = [15,6,3] #hyper params, make it easier for the tree

    if not coord_coin is None:
        dist_to_coin = np.sqrt(np.square(coord_coin[0] - x) + np.square(coord_coin[1] - y))
        for index, distance in enumerate(distance_info):
            if dist_to_coin >= distance:
                coin_positions[4] = index
                break

    return coin_positions


# one-hot encode whether a neighbouring field is affected by a bomb with timer greater than zero (these fields are forbidden anyway, the agent does not need to know about them), including the current position. The last entry quantifies the minimal timer of the bombs with effect on the neighbourhood
def bomb_danger(game_state, neighbourhood, n_steps, active_outside_of_field = True) -> np.array:

    bombs_and_timers = game_state["bombs"]
    field = game_state["field"]
    (x,y) = game_state["self"][3]
    neighbourhood.append((len(neighbourhood),(x,y)))
    bomb_danger_ = np.zeros(len(neighbourhood)+1)

    # set the danger to one outside of the field
    if active_outside_of_field:
        for index, (x_test, y_test) in neighbourhood:
            if not (x_test<COLS and y_test<ROWS and x_test>=0 and y_test>=0):
                bomb_danger_[index] = 1

    min_timer = 4

    for ((x_b,y_b),t) in bombs_and_timers:
        if abs(x_b-x) + abs(y_b-y) > n_steps + 4:
            continue
        if t == 0:
            continue

        bomb_used = False
        for index, (x_test, y_test) in neighbourhood:
            if not (x_test<COLS and y_test<ROWS and x_test>=0 and y_test>=0):
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
                    bomb_danger_[index] = 1
                    bomb_used = True

            elif dy == 0 and abs(dx) < 4:
                wall_in_between = False
                # all possible x coord in between:
                for coo in range(min(x_test,x_b) + 1, max(x_test,x_b)):
                    if field[coo,y_test] == -1:
                        wall_in_between = True
                        break
                if not wall_in_between:
                    bomb_danger_[index] = 1
                    bomb_used = True

        if bomb_used and t < min_timer:
            min_timer = t

    bomb_danger_[-1] = 4-min_timer
    return bomb_danger_

# one hot encodes, which tiles reachable in less than or in n_steps are crates.
def crates_in_neighbourhood(game_state, neighbourhood):
    field = game_state["field"]
    crates = np.zeros(len(neighbourhood))

    for index, (x_test, y_test) in neighbourhood:

        if not (x_test<COLS and y_test<ROWS and x_test>=0 and y_test>=0):
            # when the neighbourhood is not in the field, count this as no crate.
            crates[index] = 0
        elif field[x_test,y_test] == 1:
            crates[index] = 1

    return crates

# one hot encodes, which tiles reachable in less than or in n_steps are crates.
def enemies_in_neighbourhood(game_state, neighbourhood, n_steps):
    others = [(x,y) for (_,_,_,(x,y)) in game_state['others']]
    enemies = np.zeros(len(neighbourhood))
    (x,y) = game_state["self"][3]

    for (x_a,y_a) in others:
        if abs(x_a-x) + abs(y_a-y) > n_steps:
            continue
        for index, (x_test, y_test) in neighbourhood:
            if x_a == x_test and y_a == y_test:
                enemies[index] = 1
                break #out of the neighb. loop

    return enemies


def is_at_edge(x, y, state_index = None) -> np.array:
    if (state_index is None) or state_index == 0:
        edges = np.zeros(4)
        if x == 1:
            edges[0] = 1
        elif x == 15:
            edges[1] = 1
        if y == 1:
            edges[2] = 1
        elif y == 15:
            edges[3] = 1

    else:
        # in this case agent can only be located at two edges
        edges = np.zeros(2)
        if state_index == 1:
            if x == 1:
                edges[0] = 1
            elif x == 15:
                edges[1] = 1
        elif state_index == 2:
            if y == 1:
                edges[0] = 1
            elif y == 15:
                edges[1] = 1
        else:
            assert False, "state index out of bounds"

    return edges


"""ACTIONS"""

def possible_actions(game_state) -> np.array:
    death_actions = death_implying_actions(game_state)
    return np.setdiff1d(valid_actions(game_state), death_actions)

def valid_actions(game_state) -> np.array:

    ordered_actions = np.array(["UP", "DOWN", "LEFT", "RIGHT"])
    (x,y) = game_state["self"][3]

    blocked = blocked_neighbourhood(game_state, x, y, 1)
    if game_state["self"][2]:
        return np.append(ordered_actions[np.where(blocked == 0)], np.array(("WAIT", "BOMB")))
    else:
        return np.append(ordered_actions[np.where(blocked == 0)], np.array(("WAIT")))

def death_implying_actions(game_state) -> np.array:

    ordered_actions = np.array(["UP", "DOWN", "LEFT", "RIGHT", "WAIT", "BOMB"])
    (x,y) = game_state["self"][3]
    field = game_state["field"]


    # IMMEDIATELY DEADLY ACTIONS
    explosions = np.append(explosion_neighbourhood(game_state, x, y, 1), np.array([0,0]))
    bombs_with_zero_timer = np.append(bomb_danger_in_t(game_state, x, y, 1, 0, False), np.array([0]))
    indicator = explosions + bombs_with_zero_timer
    imm_death_actions = ordered_actions[np.where(indicator != 0)]


    # ACTIONS THAT GUARANTEE A DEATH WITHIN A FEW STEPS
    min_pathlength = steps_needed_on_bomb((x,y), field)
    if min_pathlength > 4:
        bomb_drop_deadly = True
    else:
        bomb_drop_deadly = False

    standing_on_bomb = False
    timer = 0
    for ((b_x,b_y),t) in game_state["bombs"]:
        if x == b_x and y == b_y:
            standing_on_bomb = True
            timer = t
            break

    if standing_on_bomb:
        on_bomb_death_dir = deadly_directions_after_own_bomb(game_state)
        death_actions = np.union1d(imm_death_actions, on_bomb_death_dir)

        # determine whether not moving is deadly:
        if min_pathlength > timer:
            death_actions = np.append(death_actions, np.array(["WAIT", "BOMB"]))
    else:
        death_actions = imm_death_actions

    if bomb_drop_deadly:
        death_actions = np.append(death_actions, np.array(["BOMB"]))

    return death_actions

def steps_needed_on_bomb(drop_coord, field) -> np.int8:
    (x,y) = drop_coord
    max_steps = 5
    if field[drop_coord] != 0:
        return max_steps
    if not (x<COLS and y<ROWS and x>=0 and y>=0):
        return max_steps

    min_steps = max_steps
    for index, (dx,dy) in [(0,(1,0)), (1,(-1,0)), (2,(0,1)), (3,(0,-1))]:
        tile_dist = 1
        for tile_dist in range(1,5):

            if tile_dist >= min_steps:
                break

            # test tile going tile_dist steps in direction dx,dy
            x_t, y_t = x+tile_dist*dx, y+tile_dist*dy

            # direction is blocked
            if field[x_t,y_t] != 0:
                break

            # standing at x+dx,y+dy, check whether we can go sideways from here by swapping dx and dy (one is alwas zero):
            if field[x_t+dy,y_t+dx] == 0 or field[x_t-dy,y_t-dx] == 0:
                min_steps = min(tile_dist + 1, min_steps)

                break


            # if we are not yet broken out of the loop, the direction points towards a narrow path of length 3. If the 4th tile is free, we can avoid the bomb by going there:
            if tile_dist == 4 and field[x+4*dx,y+4*dy] == 0:
                min_steps = min(4, min_steps)


    return min_steps


def deadly_directions_after_own_bomb(game_state):
    (x,y) = game_state["self"][3]
    field = game_state["field"]
    deadly = []

    corresponding_actions = ["RIGHT", "LEFT", "DOWN", "UP"]

    for index, (dx,dy) in [(0,(1,0)), (1,(-1,0)), (2,(0,1)), (3,(0,-1))]:
        bomb_avoidable = False
        tile_dist = 1
        for tile_dist in range(1,5):
            # test tile going tile_dist steps in direction dx,dy
            x_t, y_t = x+tile_dist*dx, y+tile_dist*dy

            # direction is blocked
            if field[x_t,y_t] != 0:
                break

            # standing at x+dx,y+dy, check whether we can go sideways from here by swapping dx and dy (one is alwas zero):
            if field[x_t+dy,y_t+dx] == 0 or field[x_t-dy,y_t-dx] == 0:
                bomb_avoidable = True
                break


            # if we are not yet broken out of the loop, the direction points towards a narrow path of length 3. If the 4th tile is free, we can avoid the bomb by going there:
            if tile_dist == 4 and field[x+4*dx,y+4*dy] == 0:
                bomb_avoidable = True

        if not bomb_avoidable:
            deadly.append(corresponding_actions[index])


    return np.array(deadly)


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

                for (x_a,y_a) in others:
                    if x_a == x_test and y_a == y_test:
                        blocked[index] = 1
                        break # out of the others loop
                if blocked[index] != 1:
                    for (x_b,y_b) in bombs:
                        if x_b == x_test and y_b == y_test:
                            blocked[index] = 1
                            break # out of the bombs loop

    return blocked

def explosion_neighbourhood(game_state, x, y, n_steps=1) -> np.array:

    explosion_positions = game_state["explosion_map"]
    neighbourhood = get_step_neighbourhood(x, y, n_steps)
    explosion = np.zeros(len(neighbourhood))

    for index, (x_test, y_test) in neighbourhood:

        if not (x_test<COLS and y_test<ROWS and x_test>=0 and y_test>=0):
            # when the neighbourhood is not in the field, count this as explosion.
            explosion[index] = 1
        else:
            explosion[index] = explosion_positions[x_test,y_test]

    return explosion


"""
DEPRECATED:
these functions are only used in testing-routines but not in the actual agent implementation
"""
# one-hot encode whether a neighbouring field is affected by a bomb in t_to_explosion steps, including the current positions. If t == None, all times are considered. t_to_explosion steps can also be a tuple of min and max times
# If n_steps == 1, the order is UP DOWN LEFT RIGHT CURRENT_POSITION
def bomb_danger_in_t(game_state, x, y, n_steps=1, t_to_explosion=None, active_outside_of_field = True) -> np.array:

    neighbourhood = get_step_neighbourhood(x, y, n_steps)
    neighbourhood.append((len(neighbourhood),(x,y)))
    bomb_danger = np.zeros(len(neighbourhood))
    bombs_and_timers = game_state["bombs"]
    field = game_state["field"]

    # set the danger to one outside of the field
    if active_outside_of_field:
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
            if not (x_test<COLS and y_test<ROWS and x_test>=0 and y_test>=0):
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


# one-hot encode whether a bomb with timer = t is located at a tile in the neighbourhood including the current position. If t == None, all bombs are considered. t can also be a tuple (t_min,t_max)
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
