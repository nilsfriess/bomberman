import numpy as np

from .base_helpers import ACTIONS, compute_risk_map
from .bomb_helpers import should_drop_bomb, explosion_radius

def one_hot_action(action: str) -> np.array:
    oh_action = np.zeros((6,1))
    oh_action[ACTIONS.index(action)] = 1

    return oh_action

def random_action(allow_bombs = True):
    if allow_bombs:
        # random action but bombs have lower probability
        return np.random.choice(ACTIONS, p=[0.18, 0.18, 0.18, 0.18, 0.18, 0.1])
    else:
        return np.random.choice(ACTIONS[:-1])

def generate_suicidal_actions(game_state):
    explosion_map = game_state['explosion_map']

    (_,_,_,(x,y)) = game_state['self']

    right = (x+1,y)
    left  = (x-1,y)
    up    = (x,y-1)
    down  = (x,y+1)

    # Walking into explosions is deadly
    actions = set()
    if explosion_map[right] > 0:
        actions.add('RIGHT')
    if explosion_map[left] > 0:
        actions.add('LEFT')
    if explosion_map[up] > 0:
        actions.add('UP')
    if explosion_map[down] > 0:
        actions.add('DOWN')

    total_escape_squares, n_escape_squares = should_drop_bomb(game_state)
    if total_escape_squares == 0:
        actions.add('BOMB')

    bombs = [pos for (pos,_) in game_state['bombs']]
    if (x,y) in bombs:
        # If we just dropped a bomb, then walking in a direction with no escape squares is suicidal
        directions = ['RIGHT', 'LEFT', 'DOWN', 'UP']
        for d in directions:
            if n_escape_squares[d] == 0:
                actions.add(d)

    return actions

def generate_stupid_actions(game_state):    
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

    # print("Risks:")
    # print(left_risk)
    # print(right_risk)
    # print(up_risk)
    # print(down_risk)
    
    '''
    If we are currently on a high-risk square, but there are neighboring zero-risk 
    squares, then all actions but the ones that lead to zero risk squares are stupid.
    '''
    stupid_actions = set(['LEFT', 'RIGHT', 'UP', 'DOWN'])
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
        stupid_actions.add('WAIT')
        stupid_actions.add('BOMB')
        
        return stupid_actions

    ''' 
    If no zero-risk action has been found, stupid actions are
    those that are not risk-decreasing
    '''
    stupid_actions = set()
    if risk_map[left] > current_risk:
        stupid_actions.add('LEFT')

    if risk_map[right] > current_risk:
        stupid_actions.add('RIGHT')

    if risk_map[up] > current_risk:
        stupid_actions.add('UP')

    if risk_map[down] > current_risk:
        stupid_actions.add('DOWN')

    # If there are actions that are not stupid, then waiting is stupid
    if len(stupid_actions) < 4:
        stupid_actions.add('WAIT')

    if current_risk > 10:
        stupid_actions.add('BOMB')
        
    # if (should_drop_bomb(game_state)[0] > 2) and (should_drop_bomb(game_state)[0] <= 8):
    #     # Dropping a bomb is not safe
    #     stupid_actions.append('BOMB')
        
    return stupid_actions

'''
Returns a rotated game_state where the agent is in the upper left quadrant.
'''
def rotate_game_to_upper_left(game_state):
    # return game_state, 0
    
    game_state = game_state.copy()
    
    (_,_,_,(x,y)) = game_state['self']
    rotations = 0
    while (x > 8) or (y > 8):
        game_state = rotate_game_state(game_state)

        rotations += 1
        (_,_,_,(x,y)) = game_state['self']

        if rotations > 4:
            raise "Error, too many rotations"

    return game_state, rotations

def mirror_game_state_lr(game_state):
    return transform_game_state(game_state, np.fliplr)

'''
Returns a new game_state that represents the state
after a 90 degree rotation in anticlockwise direction.
'''
def rotate_game_state(game_state: dict):
    game_state = game_state.copy()

    # Rotates a coordinate 90deg anti-clockwise
    rotate_coordinate = lambda coord : (16 - coord[1], coord[0])

    # Rotate field
    field = game_state['field']
    game_state['field'] = np.rot90(field, -1)

    # Rotate explosion map
    explosion_map = game_state['explosion_map']
    game_state['explosion_map'] = np.rot90(explosion_map, -1)

    def rotate_agent(agent):
        (name, score, bomb, pos) = agent
        rot_pos = rotate_coordinate(pos)
        return (name, score, bomb, rot_pos)

    # Rotate our agent
    game_state['self'] = rotate_agent(game_state['self'])

    # Rotate the other agents
    enemies = game_state['others']
    if len(enemies) > 0:
        new_enemies = []
        for enemy in enemies:
            new_enemies.append(rotate_agent(enemy))
        new_game_state['others'] = new_enemies

    # Rotate coins
    coins = game_state['coins']
    rot_coins = []
    for coin in coins:
        rot_coins.append(rotate_coordinate(coin))
    game_state['coins'] = rot_coins

    # Rotate bombs
    bombs = game_state['bombs']
    rot_bombs = []
    for position, timer in bombs:
        rot_bombs.append((rotate_coordinate(position), timer))
    game_state['bombs'] = rot_bombs
        
    return game_state
    

def rotate_action(action, n):
    if n == 0:
        return action
    
    if n < 0:
        return rotate_action(action, 4-n)

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
