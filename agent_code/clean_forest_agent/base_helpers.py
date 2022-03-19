import pyastar2d
import numpy as np
from scipy.spatial.distance import cdist

from settings import ROWS, COLS

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

coords = [[i,j] for i in range(ROWS) for j in range(COLS)]
def find_path(field, start, goal):
    start = np.array(start)
    weights = cdist(coords, [start], 'cityblock').reshape(ROWS, COLS).astype(np.float32)
    
    weights = weights + 1 # weights must be >= 1
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
    best_coord = self_pos
    
    for asset in assets:
        path = find_path(field, self_pos, asset)

        if (len(path) > 0) and (len(path) < shortest_path_length):
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

def compute_risk_map(game_state):
    '''
    Compute a 'risk factor' for each tile of the game map.
    For instance, if a bomb is about to explode and a tile
    is within the explosion radius, the tile has a high risk.
    Free tiles or tiles with coins have low risk.
    '''

    # Explosions are definitely high-risk regions
    risk_map = 100*np.array(game_state['explosion_map'])
    field = game_state['field']

    #  risk_map[field == -1] = 10
    
    for bomb_pos, bomb_val in game_state['bombs']:
        bomb_risk = 50 - 10*(bomb_val + 1)   # Bombs are riskier, the lower their timer is
        
        risk_map[bomb_pos] = bomb_risk
        
        for k in [-1,-2,-3]:
            coord_on_field = (bomb_pos[0] + k, bomb_pos[1])

            if field[coord_on_field] == -1:
                # Reached a wall, no need to look further
                break
            risk_map[coord_on_field] = bomb_risk - abs(k)

        for k in [1,2,3]:
            coord_on_field = (bomb_pos[0] + k, bomb_pos[1])

            if field[coord_on_field] == -1:
                # Reached a wall, no need to look further
                break
            risk_map[coord_on_field] = bomb_risk - abs(k)

        for k in [-1,-2,-3]:
            coord_on_field = (bomb_pos[0], bomb_pos[1]+k)

            if field[coord_on_field] == -1:
                # Reached a wall, no need to look further
                break
            risk_map[coord_on_field] = bomb_risk - abs(k)

        for k in [1,2,3]:
            coord_on_field = (bomb_pos[0], bomb_pos[1]+k)

            if field[coord_on_field] == -1:
                # Reached a wall, no need to look further
                break
            risk_map[coord_on_field] = bomb_risk - abs(k)

    return risk_map
