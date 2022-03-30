import numpy as np
from scipy.spatial.distance import cdist

from .base_helpers import find_next_step_to_assets, direction_from_coordinates, compute_risk_map
from .bomb_helpers import bomb_usefulness, should_drop_bomb
from .state_action_helpers import one_hot_action

def state_to_features(game_state: dict) -> np.array:
    if game_state is None:
        return np.array([])
    
    # Assemble features

    ''' DIRECTION TO TARGET '''
    TOTAL_COINS = 50
    # If there are collectable coins, the target is the nearest coin.
    # If there are no collectable coins but still hidden coins, the target is the nearest crate.
    # If none of the above is true, the target is the nearest enemy.
    field = np.array(game_state['field'])
    coins = np.array(game_state['coins'])
    (_,_,_,self_pos) = game_state['self']

    found_coin = False
    if len(coins) > 0:
        dist_to_coins = cdist(coins, [self_pos], 'cityblock')
        n_closest_coins = min(len(coins), 5)
        target_coins = coins[np.argpartition(dist_to_coins.ravel(),
                                             n_closest_coins-1)]

        blocked = []

        # Do not consider explosions around as as possible squares
        x,y = self_pos
        explosion_map = game_state['explosion_map']
        around_agent = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
        for square in around_agent:
            if explosion_map[square] != 0:
                blocked.append(square)

        coord_to_closest_coin = find_next_step_to_assets(field,
                                                         blocked,
                                                         self_pos,
                                                         target_coins)

        if np.any(coord_to_closest_coin != self_pos):
            # If we found a path, the coin is the target. If we found no path,
            # we are stuck behind a wall of crates -> nearest crate is target
            found_coin = True

            target_direction = direction_from_coordinates(self_pos,
                                                          coord_to_closest_coin)

    if not found_coin:
        # No visible coins or not reachable, destroy nearby crates
        enemies = game_state['others']
        crates_coords = np.argwhere(field == 1)
        enemy_positions = [pos for (_,_,_,pos) in enemies]
        if len(crates_coords) > 0:
            dist_to_crates = cdist(crates_coords, [self_pos], 'cityblock')

            n_closest_crates = min(len(crates_coords), 5)
            crates_by_distance = crates_coords[np.argpartition(dist_to_crates.ravel(),
                                                               n_closest_crates-1)]
            closest_crates = crates_by_distance[:n_closest_crates]
                
            coord_to_closest_crate = find_next_step_to_assets(field,
                                                              enemy_positions,
                                                              self_pos,
                                                              closest_crates)        

            target_direction = direction_from_coordinates(self_pos,
                                                              coord_to_closest_crate)
                                                    
        else:
            # No collectable coins and no hidden coins left, find enemy
            enemy_positions = [pos for (_,_,_,pos) in enemies]
            if len(enemy_positions) != 0:
                dist_to_enemies = cdist(enemy_positions, [self_pos], 'cityblock')

                blocked = []

                # Do not consider explosions around us as possible squares
                x,y = self_pos
                explosion_map = game_state['explosion_map']
                around_agent = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
                for square in around_agent:
                    if explosion_map[square] != 0:
                        blocked.append(square)
        
                closest_enemy = enemy_positions[np.argmin(dist_to_enemies)]
                coord_to_closest_enemy = find_next_step_to_assets(field,
                                                                  blocked, # see above
                                                                  self_pos,
                                                                  [closest_enemy])
                target_direction = direction_from_coordinates(self_pos,
                                                              coord_to_closest_enemy)
                
            else:
                # No targets left, choose arbitrary direction
                target_direction = np.array([1,0,0,0])
                
    ''' 4-vector of directions that have lower risk '''
    risk_map = compute_risk_map(game_state)
    x,y = self_pos
    own_risk = risk_map[(x,y)]
    
    lower_risk_directions = np.zeros((4,))

    right = (x+1,y)
    left  = (x-1,y)
    up    = (x,y-1)
    down  = (x,y+1)
    
    left_risk = risk_map[left]
    right_risk = risk_map[right]
    up_risk = risk_map[up]
    down_risk = risk_map[down]

    # If there exist zero risk directions, consider only those
    if up_risk == 0:
        lower_risk_directions[0] = 1
    if left_risk == 0:
        lower_risk_directions[1] = 1
    if down_risk == 0:
        lower_risk_directions[2] = 1
    if right_risk == 0:
        lower_risk_directions[3] = 1

    if not np.any(lower_risk_directions == 1):
        # Did not find zero risk direction, set 1 where risk is strictly lower
        
        lower_than_own_risk = lambda x : 1 if x < own_risk else 0
    
        # Zero means risk in that direction is higher, one means lower
        lower_risk_directions[0] = lower_than_own_risk(risk_map[up])
        lower_risk_directions[1] = lower_than_own_risk(risk_map[left])
        lower_risk_directions[2] = lower_than_own_risk(risk_map[down])
        lower_risk_directions[3] = lower_than_own_risk(risk_map[up])

    ''' USEFUL BOMB '''
    n_destroyable_crates, n_destroyable_enemies = bomb_usefulness(game_state)
    bomb_allowed = int(game_state['self'][2])
    
    if (n_destroyable_crates + n_destroyable_enemies == 0) or (not bomb_allowed):
        bomb_useful = 0
    else:
        bomb_useful = 1

    if not found_coin:
        if len(crates_coords) == 0:
            # No coins and no crates, bomb is useful as soon as we are near an enemy
            for enemy in enemy_positions:
                if abs(enemy[0] - x) - (enemy[1] - y) < 4:
                    bomb_useful = 1
                    break
    
    features = [
        target_direction.ravel(),
        lower_risk_directions.ravel(),
        [int(own_risk > 0)],
        [bomb_useful]
    ]

    return features
