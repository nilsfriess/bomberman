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

    # If we have just dropped a bomb, walking into a direction with no escapes is forbidden
    _, escape_directions = should_drop_bomb(game_state)
    bombs = [pos for (pos,_) in game_state['bombs']]
    if (x,y) in bombs:
        directions = ['RIGHT', 'LEFT', 'DOWN', 'UP']
        for d in directions:
            if escape_directions[d] == 0:
                actions.add(d)

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
    

def rotate_action(action, n):
    if n == 0:
        return action

    if (action == 'WAIT') or (action == 'BOMB'):
            return action
    
    if n > 0:
        for k in range(n):
            if action == 'UP':
                action = 'LEFT'
            elif action == 'DOWN':
                action = 'RIGHT'
            elif action == 'LEFT':
                action = 'DOWN'
            elif action == 'RIGHT':
                action = 'UP'
    else:
        for k in range(-n):
            if action == 'UP':
                action = 'RIGHT'
            elif action == 'DOWN':
                action = 'LEFT'
            elif action == 'LEFT':
                action = 'UP'
            elif action == 'RIGHT':
                action = 'DOWN'

    return action
