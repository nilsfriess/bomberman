import numpy as np
import pyastar2d

from settings import ROWS, COLS

from .more_helpers import get_neighbourhood, get_step_neighbourhood

''' ACTIONS '''
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def index_of_action(action: str) -> int:
    return ACTIONS.index(action)

def one_hot_encode_action(action: str) -> np.array:
    out = np.zeros(len(ACTIONS))
    out[index_of_action(action)] = 1
    return out

''' PATHFINDING '''
def cityblock_dist(x,y):
    return abs(x[0]-y[0]) + abs(x[1]-y[1])

coordinates = [[(i,j) for j in range(COLS)] for i in range(ROWS)]
def find_path(field, start, goal):
    # compute manhattan distance from `start` to all the squares in the field
    weights = np.array([[cityblock_dist(start, coord)
                         for coord in row]
                        for row in coordinates], dtype=np.float32)
    weights = weights + 1 # weights must >= 1
    weights[field != 0] = np.inf # walls have infinite weight

    # Compute shortest path from start to goal using A*
    path = pyastar2d.astar_path(weights, start, goal, allow_diagonal=False)
    if path is None:
        return []
    return path[1:] # discard first element in path, since it's the start position


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

def blocked_neighbourhood(field, others, x, y, n_steps):
    neighbourhood = get_step_neighbourhood(x, y, n_steps)
    blocked = np.zeros(len(neighbourhood))

    for index, (x_test, y_test) in neighbourhood:
        if (x_test<COLS and y_test<ROWS and x_test>=0 and y_test>=0):
            if field[x_test,y_test] != 0:
                blocked[index] = 1
                continue
            else:
                for (_,x_a,y_a) in others:
                    if x_a == agent_coo[0] and y_a == agent_coo[1]:
                        blocked[index] = 1
                        break # out of the agent loop
        else:
            # when the neighbourhood is not in the field, count this as blocked.
            blocked[index] = 1
    return blocked

def bomb_danger_neighbourhood(field, bombs, x, y, n_steps):
    return 0
