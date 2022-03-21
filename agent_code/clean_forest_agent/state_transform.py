import numpy as np
from scipy.spatial.distance import cdist

from .base_helpers import find_next_step_to_assets, direction_from_coordinates, compute_risk_map
from .bomb_helpers import bomb_usefulness, should_drop_bomb
from .state_action_helpers import one_hot_action

def state_to_features(game_state: dict, with_feature_list = False) -> np.array:
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

        target_direction = direction_from_coordinates(self_pos,
                                                      coord_to_closest_coin)

    else:
        # No visible coins, compute total left coins
        enemies = game_state['others']
        killed_enemies = 4 - len(enemies)

        total_score = -5*killed_enemies
        for (_,score,_,_) in enemies:
            total_score += score

        left_coins = total_score - TOTAL_COINS

        if left_coins > 0:
            # There are still coins left, target is nearest crate
            
            crates_coords = np.argwhere(field == 1)
            if len(crates_coords) > 0:
                dist_to_crates = cdist(crates_coords, [self_pos], 'cityblock')

                n_closest_crates = min(len(crates_coords), 5)
                crates_by_distance = crates_coords[np.argpartition(dist_to_crates.ravel(),
                                                           n_closest_crates-1)]
                closest_crates = crates_by_distance[:n_closest_crates]
                
                coord_to_closest_crate = find_next_step_to_assets(field,
                                                                  enemies,
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
                # No targets left
                target_direction = direction_from_coordinates(self_pos,
                                                              self_pos)
                
    ''' 5-vector of risks around us and at our position '''
    risk_map = compute_risk_map(game_state)
    risk_factors = np.zeros((5,))

    x,y = self_pos
    risk_factors[0] = risk_map[(x+1,y)]
    risk_factors[1] = risk_map[(x-1,y)]
    risk_factors[2] = risk_map[(x,y+1)]
    risk_factors[3] = risk_map[(x,y-1)]
    risk_factors[4] = risk_map[(x,y)]

    lowest_risk_direction = np.zeros((5,))
    lowest_risk_direction[risk_factors == risk_factors.min()] = 1        

    # ''' 5-vector that is one if a direction is a zero-risk direction '''
    # zero_risk_direction = np.zeros_like(risk_factors)
    # zero_risk_direction[risk_factors == 0] = 1

    ''' Is dropping a bomb a valid move '''
    bomb_allowed = int(game_state['self'][2])

    ''' Safety of dropping bomb here '''
    n_escape_squares, _ = should_drop_bomb(game_state)

    if n_escape_squares == 0:
        bomb_safety = -1
    elif n_escape_squares < 3:
        bomb_safety = 0
    else:
        bomb_safety = 1

    ''' USEFUL BOMB '''
    n_destroyable_crates, n_destroyable_enemies = bomb_usefulness(game_state)
        
    if n_destroyable_crates + n_destroyable_enemies == 0:
        bomb_useful = 0
    else:
        # Either destroys crates or enemies
        if n_destroyable_enemies == 0:
            # Bomb destroys some crates, the more, the better
            if n_destroyable_crates < 3:
                bomb_useful = 1
            else:
                bomb_useful = 2
        else:
            # Bomb tries to kill enemy
            bomb_useful = 3
    
    features = [
        target_direction.ravel(),
        lowest_risk_direction.ravel(),
        [bomb_allowed],
        [bomb_safety],
        [bomb_useful]
    ]
    
    if with_feature_list:
        feature_names = {
            'target direction' : 0,
            'lowest risk direction' : 0,
            'bomb allowed' : 0,
            'bomb safety' : 0,
            'bomb useful' : 0
        }

        for i, feature in enumerate(feature_names):
            feature_names[feature] = len(features[i])

        return np.concatenate(features), feature_names

    return np.concatenate(features)

