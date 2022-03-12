import numpy as np
import pyastar2d

from settings import ROWS, COLS

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
