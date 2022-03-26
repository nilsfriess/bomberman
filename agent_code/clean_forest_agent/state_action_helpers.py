import numpy as np

from .base_helpers import ACTIONS, compute_risk_map, deadly_in, deadly_now
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
    deadly_in_map  = deadly_in(game_state)
    deadly_now_map = deadly_now(game_state)

    (_,_,_,(x,y)) = game_state['self']

    right = (x+1,y)
    left  = (x-1,y)
    up    = (x,y-1)
    down  = (x,y+1)

    # Walking into explosions is deadly
    actions = set()
    if (deadly_in_map[right] == 1) or (deadly_now_map[right] == 1):
        actions.add('RIGHT')
    if (deadly_in_map[left] == 1) or (deadly_now_map[left] == 1):
        actions.add('LEFT')
    if (deadly_in_map[up] == 1) or (deadly_now_map[up] == 1):
        actions.add('UP')
    if (deadly_in_map[down] == 1) or (deadly_now_map[down] == 1):
        actions.add('DOWN')

    if deadly_in_map[(x,y)] == 1:
        actions.add('WAIT')

    total_escape_squares, _ = should_drop_bomb(game_state)
    
    # Check if dropping a bomb will probably kill us
    if total_escape_squares == 0:
         actions.add('BOMB')

    return actions    

def generate_stupid_actions(game_state):
    stupid_actions = set()
    
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

    # print(f"Risk at {(x,y)}: {[left_risk, right_risk, up_risk, down_risk, current_risk]}")
    
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
            stupid_actions.add('WAIT')
            stupid_actions.add('BOMB')
            return stupid_actions
        else:
            stupid_actions = set()

    ''' 
    If no zero-risk action has been found, stupid actions are
    those that are not risk-decreasing
    '''
    stupid_actions = set()
    if left_risk > current_risk:
        stupid_actions.add('LEFT')

    if right_risk > current_risk:
        stupid_actions.add('RIGHT')

    if up_risk > current_risk:
        stupid_actions.add('UP')

    if down_risk > current_risk:
        stupid_actions.add('DOWN')

    risks = [left_risk, right_risk, up_risk, down_risk]
    # Filter risks that are wall-risks (i.e. == 10) or larger than the current risk
    for risk in risks:
        if risk == 10: # wall
            continue
        if risk <= current_risk:
            stupid_actions.add('WAIT')

            if current_risk > 0:
                stupid_actions.add('BOMB')
            break

    # If we have just dropped a bomb, walking into a direction with no escapes is forbidden
    _, escape_directions = should_drop_bomb(game_state)
    bombs = [pos for (pos,_) in game_state['bombs']]
    if (x,y) in bombs:
        directions = ['RIGHT', 'LEFT', 'DOWN', 'UP']
        for d in directions:
            if escape_directions[d] == 0:
                stupid_actions.add(d)

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
