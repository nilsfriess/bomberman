import numpy as np
import pyastar2d

from scipy.spatial.distance import cdist

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

coords = [[i,j] for i in range(ROWS) for j in range(COLS)]
def find_path(field, start, goal):
    # compute manhattan distance from `start` to all the squares in the field
    # weights = np.array([[cityblock_dist(start, coord)
    #                      for coord in row]
    #                     for row in coordinates], dtype=np.float32)

    start = np.array(start)
    weights = cdist(coords, [start], 'cityblock').reshape(ROWS, COLS).astype(np.float32)
    
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

'''
Returns a new game_state that represents the state
after a n 90 degree rotations in clockwise direction.
Note that this corresponds to *anti*clockwise rotations
of the game field in the GUI, since both game fields
are transposed versions of each other.
'''
def rotate_game_state(game_state: dict, n):
    n = -n
    rot_game_state = game_state.copy()
    
    # rotate the field
    field = game_state['field']
    rot_game_state['field'] = np.rot90(field, n)
    
    # rotate bombs
    bombs = game_state['bombs']
    if len(bombs) > 0:
        bomb_matrix = np.zeros_like(field)
        for position, timer in bombs:
            bomb_matrix[position] = timer

        new_bomb_matrix = np.rot90(bomb_matrix, n)
        indices =  (new_bomb_matrix != 0).nonzero()
        values = new_bomb_matrix[indices]

        new_bombs = []
        for i,[x,y] in enumerate(np.transpose(indices)):
            new_bombs.append(((x,y), values[i]))

        rot_game_state['bombs'] = new_bombs

    # rotate explosion map
    explosion_map = game_state['explosion_map']
    rot_game_state['explosion_map'] = np.rot90(explosion_map, n)

    # rotate coins
    coins = game_state['coins']
    if len(coins) > 0:
        coin_matrix = np.zeros_like(field)
        for coin in coins:
            coin_matrix[coin] = 1
        new_coin_matrix = np.rot90(coin_matrix, n)

        new_coins = []
        for [x,y] in np.transpose((new_coin_matrix != 0).nonzero()):
            new_coins.append((x,y))

        rot_game_state['coins'] = new_coins

    # rotate self and enemies
    def rotate_player(player):
        (name, score, bomb, pos) = player

        player_matrix = np.zeros_like(field)
        player_matrix[pos] = 1
        new_player_matrix = np.rot90(player_matrix, n)
        new_pos = tuple(np.array(new_player_matrix.nonzero()).ravel())

        return (name, score, bomb, new_pos)

    agent = game_state['self']
    rot_game_state['self'] =  rotate_player(agent)

    enemies = game_state['others']
    if len(enemies) > 0:
        new_enemies = []
        for enemy in enemies:
            new_enemies.append(rotate_player(enemy))
        rot_game_state['others'] = new_enemies
    
    return rot_game_state

def rotate_action(action, n):
    if n == 0:
        return action

    if (action == 'WAIT') or (action == 'BOMB'):
        rot_action = action
    elif action == 'UP':
        rot_action = 'LEFT'
    elif action == 'DOWN':
        rot_action = 'RIGHT'
    elif action == 'LEFT':
        rot_action = 'DOWN'
    elif action == 'RIGHT':
        rot_action = 'UP'

    return rotate_action(rot_action, n-1)
