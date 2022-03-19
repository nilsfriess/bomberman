import numpy as np

from .base_helpers import ACTIONS, compute_risk_map
from .bomb_helpers import should_drop_bomb

def one_hot_action(action: str) -> np.array:
    oh_action = np.zeros((6,1))
    oh_action[ACTIONS.index(action)] = 1

    return oh_action

def random_action(allow_bombs = False):
    if allow_bombs:
        return np.random.choice(ACTIONS)
    else:
        return np.random.choice(ACTIONS[:-1])

def generate_stupid_actions(game_state):
    if np.random.uniform() < 0.8: # Only filter sometimes
        return []
    
    risk_map = compute_risk_map(game_state)
    (_,_,_,(x,y)) = game_state['self']
    
    current_risk = risk_map[(x,y)]

    right = (x+1,y)
    left  = (x-1,y)
    up    = (x,y-1)
    down  = (x,y+1)
    
    left_risk = risk_map[left]
    right_risk = risk_map[right]
    up_risk = risk_map[up]
    down_risk = risk_map[down]

    '''
    If we are currently on a high-risk square, but there are neighboring zero-risk 
    squares, then all actions but the ones that lead to zero risk squares are stupid.
    '''
    stupid_actions = ['LEFT', 'RIGHT', 'UP', 'DOWN']
    if current_risk > 0:
        if left_risk == 0:
          stupid_actions.remove('LEFT')
        if right_risk == 0:
          stupid_actions.remove('RIGHT')
        if up_risk == 0:
          stupid_actions.remove('UP')
        if down_risk == 0:
          stupid_actions.remove('DOWN')
    if len(stupid_actions) < 4:
        # We found a good move
        stupid_actions.append('WAIT')
        stupid_actions.append('BOMB')
        
        return stupid_actions

    ''' 
    If no zero-risk action has been found, stupid actions are
    those that are risk-increasing
    '''
    stupid_actions = []
    if risk_map[left] > current_risk:
        stupid_actions.append('LEFT')

    if risk_map[right] > current_risk:
        stupid_actions.append('RIGHT')

    if risk_map[up] > current_risk:
        stupid_actions.append('UP')

    if risk_map[down] > current_risk:
        stupid_actions.append('DOWN')

    # If there are actions that are not stupid, then waiting is stupid
    if len(stupid_actions) < 4:
        stupid_actions.append('WAIT')

    if current_risk > 0:
        stupid_actions.append('BOMB')

    if should_drop_bomb(game_state)[0] <= 10:
        # Dropping a bomb is not safe
        stupid_actions.append('BOMB')
    
    return stupid_actions


'''
Transforms the given game_state according to the transformation `transform`.
This could be, for instance, np.rot90, np.flipud or np.fliplr
'''
def transform_game_state(game_state: dict, transform) -> dict:
    new_game_state = game_state.copy()
    
    # transform the field
    field = game_state['field']
    new_game_state['field'] = transform(field)
    
    # transform bombs
    bombs = game_state['bombs']
    if len(bombs) > 0:
        bomb_matrix = np.zeros_like(field)
        for position, timer in bombs:
            bomb_matrix[position] = timer

        new_bomb_matrix = transform(bomb_matrix)
        indices =  (new_bomb_matrix != 0).nonzero()
        values = new_bomb_matrix[indices]

        new_bombs = []
        for i,[x,y] in enumerate(np.transpose(indices)):
            new_bombs.append(((x,y), values[i]))

        new_game_state['bombs'] = new_bombs

    # transform explosion map
    explosion_map = game_state['explosion_map']
    new_game_state['explosion_map'] = transform(explosion_map)

    # transform coins
    coins = game_state['coins']
    if len(coins) > 0:
        coin_matrix = np.zeros_like(field)
        for coin in coins:
            coin_matrix[coin] = 1
        new_coin_matrix = transform(coin_matrix)

        new_coins = []
        for [x,y] in np.transpose((new_coin_matrix != 0).nonzero()):
            new_coins.append((x,y))

        new_game_state['coins'] = new_coins

    # transform self and enemies
    def transform_player(player):
        (name, score, bomb, pos) = player

        player_matrix = np.zeros_like(field)
        player_matrix[pos] = 1
        new_player_matrix = transform(player_matrix)
        new_pos = tuple(np.array(new_player_matrix.nonzero()).ravel())

        return (name, score, bomb, new_pos)

    agent = game_state['self']
    new_game_state['self'] =  transform_player(agent)

    enemies = game_state['others']
    if len(enemies) > 0:
        new_enemies = []
        for enemy in enemies:
            new_enemies.append(transform_player(enemy))
        new_game_state['others'] = new_enemies
    
    return new_game_state

'''
Returns a rotated game_state where the agent is in the upper left quadrant.
'''
def rotate_game_to_upper_left(game_state):
    '''
    Check which quadrant of game_state['field'] we are in.
    Note that this is *not* the same quadrant as on the actual
    game field rendered in the GUI (upper right and lower left
    are swapped, since the GUI game field is a transposed version
    of the game_state['field']).
    '''
    return game_state, 0
    
    (_,_,_,(x,y)) = game_state['self']
    if (x <= 8) and (y <= 8):
        # upper left
        quad = 0
    elif (x > 8) and (y <= 8):
        # upper right
        quad = 1
    elif (x > 8) and (y > 8):
        # lower right
        quad = 2
    elif (x <= 8) and (y > 8):
        quad = 3

    game_state = rotate_game_state(game_state, quad)
        
    #  Sanity check: make surethat after the rotation, we are in the upper left quadrant
    assert(game_state['self'][3][0] <= 8)
    assert(game_state['self'][3][1] <= 8)
    #print(f"Position after rotation {game_state['self'][3]}")

    return game_state, quad
    
'''
Returns a new game_state that represents the state
after n 90 degree rotations in clockwise direction.
Note that this corresponds to *anti*clockwise rotations
of the game field in the GUI, since both game fields
are transposed versions of each other.
'''
def rotate_game_state(game_state: dict, n):
    n = -n
    rotate_n_90deg = lambda state : np.rot90(state, n)

    return transform_game_state(game_state, rotate_n_90deg)
    

def rotate_action(action, n):
    if n < 0:
        return rotate_action(action, 4-n)
    
    if n == 0:
        return action

    print("NOOOO")

    if (action == 'WAIT') or (action == 'BOMB'):
        rot_action = action
    elif action == 'UP':
        rot_action = 'RIGHT'
    elif action == 'DOWN':
        rot_action = 'LEFT'
    elif action == 'LEFT':
        rot_action = 'UP'
    elif action == 'RIGHT':
        rot_action = 'DOWN'

    return rotate_action(rot_action, n-1)
