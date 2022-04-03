import numpy as np

def one_hot_action(action: str) -> np.array:
    oh_action = np.zeros((6,1))
    oh_action[ACTIONS.index(action)] = 1

    return oh_action

def random_action(allow_bombs = True):
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
