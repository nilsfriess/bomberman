import numpy as np
import pyastar2d

from settings import ROWS, COLS

''' ACTIONS '''
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def one_hot_action(action: str) -> np.array:
    oh_action = np.zeros((6,1))
    oh_action[ACTIONS.index(action)] = 1

    return oh_action

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

'''
Computes for each asset in `assets` the shortest path from `self_pos`
to `asset` using A*.

Returns the first coordinate of the shortest among all paths.
'''
def find_next_step_to_assets(field, others, self_pos, assets):
    if len(assets) == 0:
        return self_pos

    for other in others:
        field[other] = 1
    
    shortest_path_length = float("inf")
    best_coord = (0,0)
    
    for asset in assets:
        path = find_path(field, self_pos, asset)
        
        if len(path) == 0:
            return self_pos

        if len(path) < shortest_path_length:
            shortest_path_length = len(path)
            best_coord = path[0]

    return best_coord

'''
Returns 1-hot encoded 4-array where `self_pos` and `asset_pos`
are two neighboring coordinates on the grid. Each component
of the result corresponds to one direction, i.e., one component
is 1 iff `self_pos` is directly above `asset_pos`, one component
is 1 iff `self_pos` is left of `asset_pos` etc. 

Should be used together with `find_next_step_to_assets`
'''
def direction_from_coordinates(self_pos, asset_pos):
    direction = np.zeros((4,1))

    self_pos = np.array(self_pos)
    asset_pos = np.array(asset_pos)
    
    if not np.array_equal(self_pos, asset_pos):
        dist = self_pos - asset_pos

        if dist[0] == 0:
            if dist[1] == 1:
                direction[0] = 1
            else:
                direction[1] = 1
        else:
            if dist[0] == 1:
                direction[2] = 1
            else:
                direction[3] = 1

    return direction

def action_from_direction(direction):
    if direction[0] == 1:
        return 'UP'

    if direction[1] == 1:
        return 'DOWN'

    if direction[2] == 1:
        return 'LEFT'

    if direction[3] == 1:
        return 'RIGHT'

    return 'WAIT'
