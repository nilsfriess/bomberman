import numpy as np

from settings import ROWS, COLS

from .base_helpers import *

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

# one hot encodes, which fields reachable in less than n_steps are blocked. If n_steps == 1, the order is DOWN UP LEFT RIGHT
# 1 means blocked.
def blocked_neighbourhood(field, others, bombs, x, y, n_steps):
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

# nonzero-hot encode wheter a field is in bomb-danger and assign it a risk score depending on timer of the bomb and distance
# If n_steps == 1, the order is DOWN UP LEFT RIGHT CURRENT_POSITION
def bomb_risk_neighbourhood(field, bombs_and_timers, x, y, n_steps):
    neighbourhood = get_step_neighbourhood(x, y, n_steps)
    neighbourhood.append((len(neighbourhood),(x,y)))
    bomb_risk = np.zeros(len(neighbourhood))

    max_risk = 4 # max dist = 3, max timer = 3, risk = 7 - dist - timer
    def risk(dist, timer):
        return max_risk - dist

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

# one hot encodes, which fields reachable in less than n_steps are blocked. If n_steps == 1, the order is DOWN UP LEFT RIGHT
def explosion_neighbourhood(x, y, explosion_positions, n_steps):
    neighbourhood = get_step_neighbourhood(x, y, n_steps)
    explosion = np.zeros(len(neighbourhood))

    for index, (x_test, y_test) in neighbourhood:

        if not (x_test<COLS and y_test<ROWS and x_test>=0 and y_test>=0):
            # when the neighbourhood is not in the field, count this as explosion.
            explosion[index] = 2

        else:
            explosion[index] = explosion_positions[x_test, y_test]

    return explosion

# one hot encodes, which fields reachable in less than n_steps are blocked. If n_steps == 1, the order is DOWN UP LEFT RIGHT
def crates_in_neighbourhood():

    return 0


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
