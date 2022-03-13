import numpy as np

def action_is_stupid(game_state : dict, action) -> bool:
    ''' 
    Compute the explosion_map for the next iteration.
    Then, check for each action, if we would land on an 
    explosion. If yes, then the action is stupid.
    '''

    bomb_avoiding, bomb_near_us = bomb_avoiding_actions(game_state)

    # if bomb_near_us is true, a bomb is the important threat
    if bomb_near_us:
        if action in bomb_avoiding:
            return False
        else:
            return True
        
    old_explosion_map = np.array(game_state['explosion_map'])
    bombs = game_state['bombs']
    field = game_state['field']
    
    new_explosion_map = old_explosion_map

    #new_explosion_map[old_explosion_map != 0] -= 1 # decrease explosion counter

    
    # Compute explosions
    for (bomb, time) in bombs:
        if time > 0:
            continue
        new_explosion_map[bomb] = 1

        for k in [-2,-1,1,2]:
            # Explosion left and right of bomb
            coord = (bomb[0] + k, bomb[1])
            if coord[0] < len(field):
                new_explosion_map[coord] = 1

            # Explosion abov and below of bomb
            coord = (bomb[0], bomb[1] + k)
            if coord[1] < len(field):
                new_explosion_map[coord] = 1

    (x,y) = game_state['self'][3]
    if action == 'UP':
        if new_explosion_map[x,y-1] > 0:
            return True

    if action == 'DOWN':
        if new_explosion_map[x,y+1] > 0:
            return True

    if action == 'LEFT':
        if new_explosion_map[x-1,y] > 0:
            return True

    if action == 'RIGHT':
        if new_explosion_map[x+1,y] > 0:
            return True

    if action == 'WAIT':
        if new_explosion_map[x,y] > 0:
            return True

    return False















                

    # possible_actions = []

    # if len(bombs) == 0:
    #     return False
    # else:
    #     bomb_avoiding = bomb_avoiding_actions(game_state)
    #     if action in bomb_avoiding:
    #         return False
    #     else:
    #         return True
                
    # if explosion_map[x+1,y] > 0:
    #     if 'RIGHT' in possible_actions:
    #         possible_actions.remove('RIGHT')

    # if explosion_map[x-1,y] > 0:
    #     if 'LEFT' in possible_actions:
    #         possible_actions.remove('LEFT')

    # if explosion_map[x,y+1] > 0:
    #     if 'DOWN' in possible_actions:
    #         possible_actions.remove('DOWN')

    # if explosion_map[x,y-1] > 0:
    #     if 'UP' in possible_actions:
    #         possible_actions.remove('UP')

    # # if len(possible_actions) == 0:
    # #     print("CAUTION: FOUND NO USEFUL ACTION")
    # #     return False
    # # else:        
    # #     if (action in possible_actions):
    # #         print(f"Action {action} is useful")
    # #         return False
    # #     else:
    # #         print(f"Action {action} is stupid")
    # #         return True

    # if (action in possible_actions) or (len(possible_actions) == 0):
    #     #print(f"Action {action} is useful (or found none)")
    #     return False
    # else:
    #     #print(f"Action {action} is stupid")
    #     return True


           
    
def bomb_avoiding_actions(game_state):
    field = game_state['field']
    bombs = [pos for (pos,_) in game_state['bombs']]
    (x,y) = game_state['self'][3]

    possible_actions = []
    if ((x,y) in bombs):
        # We are standing on bomb, find free directions
        if field[x,y+1] == 0:
            possible_actions.append('DOWN')
        if field[x,y-1] == 0:
            possible_actions.append('UP')
        if field[x+1,y] == 0:
            possible_actions.append('RIGHT')
        if field[x-1,y] == 0:
            possible_actions.append('LEFT')

        return possible_actions, True

    bomb_near_us = False
    for k in [-3,-2,-1,1,2,3]:        
        if (x+k,y) in bombs:            
            bomb_near_us = True
            # Bomb is left or right of us, check if we can go up or down
            if field[x,y+1] == 0:
                possible_actions.append('DOWN')
            if field[x,y-1] == 0:
                possible_actions.append('UP')
                
            if len(possible_actions) == 0:
                # If we cannot go up or down, move away from bomb
                if k < 0:
                    possible_actions.append('RIGHT')
                else:
                    possible_actions.append('LEFT')
            
        elif (x,y+k) in bombs:
            bomb_near_us = True
            # Bomb is above or below us, try to escape right or left
            if field[x+1,y] == 0:
                possible_actions.append('RIGHT')
            if field[x-1,y] == 0:
                possible_actions.append('LEFT')

            if len(possible_actions) == 0:
                # If we cannot go left or right, move away from bomb
                if k < 0:
                    possible_actions.append('DOWN')
                else:
                    possible_actions.append('UP')
                    
    return possible_actions, bomb_near_us
