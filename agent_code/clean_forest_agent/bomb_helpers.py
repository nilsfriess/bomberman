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
    distance_list = [0]
    
    for k in [-1,-2,-3]:
        coord_on_field = (bomb_pos[0] + k, bomb_pos[1])        
        if field[coord_on_field] == -1:
            # Reached a wall, no need to look further
            break
        explosion_list.append(coord_on_field)
        distance_list.append(abs(k))
        
    for k in [1,2,3]:
        coord_on_field = (bomb_pos[0] + k, bomb_pos[1])
        
        if field[coord_on_field] == -1:
            # Reached a wall, no need to look further
            break
        explosion_list.append(coord_on_field)
        distance_list.append(abs(k))

    for k in [-1,-2,-3]:
        coord_on_field = (bomb_pos[0], bomb_pos[1]+k)
        
        if field[coord_on_field] == -1:
            # Reached a wall, no need to look further
            break
        explosion_list.append(coord_on_field)
        distance_list.append(abs(k))
        
    for k in [1,2,3]:
        coord_on_field = (bomb_pos[0], bomb_pos[1]+k)
    
        if field[coord_on_field] == -1:
            # Reached a wall, no need to look further
            break
        explosion_list.append(coord_on_field)
        distance_list.append(abs(k))
        
    if with_risk:
        return zip(explosion_list, distance_list)
    else:
        return explosion_list

def reachable_from(start, field):
    x,y = start

    reachable = []
    for delta in [-1,1]:
        coord = (x+delta,y)
        if field[coord] == 0:
            reachable.append(coord)
        coord = (x,y+delta)
        if field[coord] == 0:
            reachable.append(coord)

    return reachable

def reachable_in_n(start, field, n):
    reachable_squares = set(reachable_from(start, field)) # use set to avoid storing duplicates
    visited = []
    for k in range(n): # n-1 iterations = reachable in n steps
        reachable_from_here = set()
        for square in reachable_squares:
            if square in visited: # we have already checked the reachable squares
                continue
            
            reachable = reachable_from(square, field)
            for r in reachable:
                reachable_from_here.add(r)
            visited.append(square)

        reachable_squares = reachable_squares.union(reachable_from_here)

    return reachable_squares

    
def should_drop_bomb(game_state):
    '''
    This function checks if dropping a bomb at our current position is a good idea.
    The return value is a non-negative number, denoting the number of safe squares 
    available. Zero means, dropping a bomb at our current position will probably
    kill us (probably, since in the 4 steps until the bomb explodes, an enemy could
    create an escape route by destroying blocking crates or the game could end).
    '''
    field = game_state['field']
    self_pos = game_state['self'][3]

    # Make bombs act as walls
    for bomb, _ in game_state['bombs']:
        field[bomb] = -1    

    '''
    Compute the coordinates to all squares reachable within four steps.
    We also compute how many escape squares are reachable from which neighbor
    to provide the agent with a feature that represents the best escape direction.
    '''
    x,y = self_pos
    explosion_squares = explosion_radius(field, self_pos, with_risk = False)

    reachable = set()
    number_of_squares = np.zeros((4,))

    all_neighbors = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
    #               ['RIGHT', 'LEFT', 'DOWN', 'UP']
    
    reachable_neighbors = reachable_from(self_pos, field)
    for i, neighbor in enumerate(all_neighbors):
        if field[neighbor] == -1:
            continue
        
        if neighbor not in reachable_neighbors:
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
Returns either 0, 5, 10 or 15:
- 0:  Bomb is useless (does neither destroy crates nor enemies)
- 5:  Bomb destroys crates but no enemy
- 10: Bomb destroys enemy but no crates
- 15: Bomb destroys both enemies and crates
'''
def bomb_usefulness(game_state):
    field = game_state['field']
    self_pos = game_state['self'][3]
    
    explosions = set(explosion_radius(field, self_pos, False))

    # Destroyable crates
    crates = set(map(tuple, np.argwhere(field == 1)))
    n_destroyable_crates = len(explosions & crates) # Compute intersection

    # Destroyable enemies
    enemies = set()
    for _,_,_,pos in game_state['others']:
        enemies.add(pos)

    n_destroyable_enemies = len(explosions & enemies)
    
    usefulness = 0

    if n_destroyable_crates > 0:
        usefulness += 5

    if n_destroyable_enemies > 0:
        usefulness += 10
    
    return usefulness
