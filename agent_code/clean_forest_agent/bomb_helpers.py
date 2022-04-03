import numpy as np

'''
Computes the "explosion radius" of a bomb, i.e., a list of squares
that will be hit by an explosion of a bomb at bomb_pos.

Since this is also used in the risk map calculation,  by setting
`with_risk = True`, this returns a list of tuples containing the
coordinate of the square affected by the explosion and the distance
to the bomb square of this coordinate.
'''
def explosion_radius(field, bomb_pos, with_risk = True):
    explosion_list = [bomb_pos]
    risk_list = [4]
    
    for k in [-1,-2,-3]:
        coord_on_field = (bomb_pos[0] + k, bomb_pos[1])        
        if field[coord_on_field] == -1:
            # Reached a wall, no need to look further
            break
        explosion_list.append(coord_on_field)
        risk_list.append(4 - abs(k))
        
    for k in [1,2,3]:
        coord_on_field = (bomb_pos[0] + k, bomb_pos[1])
        
        if field[coord_on_field] == -1:
            # Reached a wall, no need to look further
            break
        explosion_list.append(coord_on_field)
        risk_list.append(4 - abs(k))

    for k in [-1,-2,-3]:
        coord_on_field = (bomb_pos[0], bomb_pos[1]+k)
        
        if field[coord_on_field] == -1:
            # Reached a wall, no need to look further
            break
        explosion_list.append(coord_on_field)
        risk_list.append(4 - abs(k))
        
    for k in [1,2,3]:
        coord_on_field = (bomb_pos[0], bomb_pos[1]+k)
    
        if field[coord_on_field] == -1:
            # Reached a wall, no need to look further
            break
        explosion_list.append(coord_on_field)
        risk_list.append(4 - abs(k))
        
    if with_risk:
        return zip(explosion_list, risk_list)
    else:
        return explosion_list

def neighbors_of(start, field):
    x,y = start

    all_neighbors = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
    
    reachable = []
    for neighbor in all_neighbors:
        if field[neighbor] == 0:
            reachable.append(neighbor)

    return reachable

def reachable_in_n(start, field, n):
    reachable = set()
    stack = [start]

    dist = lambda x,y : abs(x[0] - y[0]) + abs(x[1] - y[1])

    while len(stack) > 0:
        next = stack.pop()

        if next not in reachable:
            reachable.add(next)

            if dist(start, next) < n:
                for neighbor in neighbors_of(next, field):
                    stack.append(neighbor)

    return reachable
    
def should_drop_bomb(game_state):
    '''
    This function checks if dropping a bomb at our current position is a good idea.
    The return value is a non-negative number, denoting the number of safe squares 
    available. Zero means, dropping a bomb at our current position will probably
    kill us (probably, since in the 4 steps until the bomb explodes, an enemy could
    create an escape route by destroying blocking crates or the game could end).
    '''
    field = game_state['field'].copy()
    self_pos = game_state['self'][3]

    # Make bombs act as walls
    for bomb, _ in game_state['bombs']:
        field[bomb] = -1

    # Make enemies act as walls
    for (_,_,_,pos) in game_state['others']:
        field[pos] = -1

    # The agent itself is also blocking
    field[self_pos] = -1

    '''
    Compute the coordinates to all squares reachable within four steps.
    We also compute how many escape squares are reachable from which neighbor
    to provide the agent with a feature that represents the best escape direction.
    '''
    explosion_squares = explosion_radius(field, self_pos, with_risk = False)

    reachable = set()
    number_of_squares = np.zeros((4,))

    x,y = self_pos
    all_neighbors = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
    #               ['RIGHT', 'LEFT', 'DOWN', 'UP']
    
    for i, neighbor in enumerate(all_neighbors):
        if field[neighbor] != 0:
            continue
            
        squares = reachable_in_n(neighbor, field, 3)

        reachable = reachable.union(squares)
        number_of_squares[i] = len(squares - set(explosion_squares))

    escape_squares = reachable - set(explosion_squares)

    directions = ['RIGHT', 'LEFT', 'DOWN', 'UP']
    escape_squares_directions = dict()
    for i,direction in enumerate(directions):
        escape_squares_directions[direction] = number_of_squares[i]
        
    return len(escape_squares), escape_squares_directions

'''
Returns a tuple (n_destroyable_coins, n_destroyable_enemies)
'''
def bomb_usefulness(game_state):
    field = game_state['field']
    self_pos = game_state['self'][3]
    
    explosions = set(explosion_radius(field, self_pos, False))
    #print(explosions)

    # Destroyable crates
    crates = set(map(tuple, np.argwhere(field == 1)))
    n_destroyable_crates = len(explosions & crates) # Compute intersection

    # Destroyable enemies
    enemies = [pos for (_,_,_,pos) in game_state['others']]
    
    n_destroyable_enemies = len(explosions & set(enemies))

    return n_destroyable_crates, n_destroyable_enemies
