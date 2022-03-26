import pyastar2d
import numpy as np
from scipy.spatial.distance import cdist

from settings import ROWS, COLS

from .bomb_helpers import should_drop_bomb, explosion_radius

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

coords = [[i,j] for i in range(ROWS) for j in range(COLS)]
def find_path(field, start, goal):
    start = np.array(start)
    weights = cdist(coords, [start], 'cityblock').reshape(ROWS, COLS).astype(np.float32)
    
    weights = weights + 1 # weights must be >= 1
    weights[field != 0] = 10e9 # walls have very weight

    # Set field at goal to zero if we are looking for something on the field
    # Otherwise the algorithm would not find a path to the target.
    weights[goal] = 1
    
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
    field = game_state['field'].copy() # We change the field below, need to create a copy
    self_pos = game_state['self'][3]
    
    risk_map = np.zeros_like(field)
    
    # # Squares that are deadly now have high risk
    # deadly_map = deadly_in(game_state)
    # risk_map[deadly_map == 0] = 90
    
    # # Squares that are deadly in the next step are also risky
    # risk_map[deadly_map == 1] = 80

    #  # Squares that are deadly in the step after the next are a bit less risky
    # risk_map[deadly_map == 2] = 70
    
    # Squares with bombs have very high risk
    bombs = [pos for (pos, val) in game_state['bombs']]
    for bomb in bombs:
        explosions = explosion_radius(game_state['field'], bomb)

        for (square, risk) in explosions:
            risk_map[square] = risk

        risk_map[bomb] = 10 # The risk of the bomb field itself is high

    # Compute the explosion radius of bombs that could be dropped by enemies
    for (_,_,_,pos) in game_state['others']:
       exp_radius =  explosion_radius(game_state['field'], pos, False)

       for coord in exp_radius:
           risk_map[coord] = 1 # Not that risky but should be avoided if zero risk squares are available

     # Blocked tiles have large risk since they lead to invalid actions
    risk_map[field != 0] = 10

    ''' 
    If we are on a bomb, then walking in a direction with no escape routes,
    this will definitely kill us, so this has maximum risk
    '''
    if self_pos in bombs:
        _, escape_squares_directions = should_drop_bomb(game_state)

        x,y = self_pos
        neighbors = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
        suicide_directions = np.zeros((4,))
        for k, dir_escape_squares in enumerate(escape_squares_directions.values()):
            if dir_escape_squares == 0:
                risk_map[neighbors[k]] = 100
           
    return risk_map


'''
For every square on the game state, compute the number of time steps
until standing on that square will lead to the death of the agent.
A value of 0 means that the square is now deadly (for instance, explosions
are deadly). A value of 1 means that the square will be deadly in the
next time step. A value of -1 means that the square is not deadly in the
next four time steps.
'''
def deadly_in(game_state):
    field = game_state['field']
    deadly_map = -1*np.ones_like(field)

    # For every bomb, compute the explosion radius and set the value there to the bomb's timer
    bombs = game_state['bombs']
    for pos, value in bombs:
        explosion = explosion_radius(field, pos, False)

        for square in explosion:
            deadly_map[square] = value + 1

    return deadly_map

def deadly_now(game_state):
    deadly_map = np.zeros_like(game_state['field'])
    deadly_map[ game_state['explosion_map'] > 0] = 1

    return deadly_map
    

