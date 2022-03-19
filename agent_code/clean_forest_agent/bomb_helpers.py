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

    # Act as if our position is blocked to prevent computation of escape routes that
    # would walk over bomb
    field[self_pos] = -1

    # First, compute the coordinates of all squares that can be reached within one step
    def reachable_from(start):
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

    def reachable_in_n(start, n):
        reachable_squares = set(reachable_from(start)) # use set to avoid storing duplicates
        visited = []
        for k in range(n): # n-1 iterations = reachable in n steps
            reachable_from_here = set()
            for square in reachable_squares:
                if square in visited: # we have already checked the reachable squares
                    continue
            
                reachable = reachable_from(square)
                for r in reachable:
                    reachable_from_here.add(r)
                visited.append(square)

            reachable_squares = reachable_squares.union(reachable_from_here)

        return reachable_squares

    '''
    Now, compute the coordinates to all squares reachable within four steps.
    We also compute how many escape squares are reachable from which neighbor
    to provide the agent with a feature that represents the best escape direction.
    '''
    x,y = self_pos
    explosion_squares = explosion_radius(field, self_pos, with_risk = False)

    reachable = set()
    number_of_squares = np.zeros((4,))

    all_neighbors = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
    neighbor_directions = ['RIGHT', 'LEFT', 'DOWN', 'UP']
    
    reachable_neighbors = reachable_from(self_pos)
    for i, neighbor in enumerate(all_neighbors):
        if neighbor not in reachable_neighbors:
            continue
            
        squares = reachable_in_n(neighbor, 3)

        reachable = reachable.union(squares)
        number_of_squares[i] = len(squares - set(explosion_squares))

    escape_squares = reachable - set(explosion_squares)

    return len(escape_squares), neighbor_directions[np.argmax(number_of_squares)]

def bomb_usefulness(game_state):
    field = game_state['field']
    self_pos = game_state['self'][3]
    
    explosions = set(explosion_radius(field, self_pos, False))
    crates = set(map(tuple, np.argwhere(field == 1)))
    
    n_destroyable_crates = len(explosions & crates)

    # TODO: Destroyable enemies

    return n_destroyable_crates
