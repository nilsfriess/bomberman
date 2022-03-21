import numpy as np
from scipy.spatial.distance import cdist

from .base_helpers import find_next_step_to_assets, direction_from_coordinates, compute_risk_map
from .bomb_helpers import bomb_usefulness, should_drop_bomb
from .state_action_helpers import one_hot_action

def state_to_features(game_state: dict, with_feature_list = False) -> np.array:
    if game_state is None:
        return np.array([])
    
    # Assemble features

    ''' Field around as (2 to the left, 2 to the right, etc.)'''
    field_around = np.zeros((8,))
    (_,_,_,self_pos) = game_state['self']
    x,y = self_pos
    field = np.array(game_state['field'])

    field_around[0] = field[(x+1,y)]
    field_around[1] = field[(x-1,y)]
    field_around[2] = field[(x,y+1)]
    field_around[3] = field[(x,y-1)]
    if x+2 < field.shape[0]:
        field_around[4] = field[(x+2,y)]
    if x-2 >= 0:
        field_around[5] = field[(x-2,y)]
    if y+2 < field.shape[1]:
        field_around[6] = field[(x,y+2)]
    if y-2 >= 0:
        field_around[7] = field[(x,y-2)]
        
    ''' OWN POSITION '''
    # own_position = np.zeros((field.shape[0], field.shape[1]))
    # own_position[self_pos] = 1
    own_position = np.array(self_pos)

    ''' DIRECTION TO CLOSEST COIN '''
    # Find 5 closest coins, where `close` is w.r.t. the cityblock distance
    game_coins = np.array(game_state['coins'])
    enemies = [(x,y) for (_,_,_,(x,y)) in game_state['others']]
    
    if len(game_coins) > 0:
        dist_to_coins = cdist(game_coins, [self_pos], 'cityblock')

        n_closest_coins = min(len(game_coins), 5)
        coins = game_coins[np.argpartition(dist_to_coins.ravel(),
                                           n_closest_coins-1)]
        closest_coins = coins[:n_closest_coins]

        #enemies = []
        coord_to_closest_coin = find_next_step_to_assets(field,
                                                         enemies,
                                                         self_pos,
                                                         closest_coins)

        coin_direction = direction_from_coordinates(self_pos,
                                                    coord_to_closest_coin)
    else:
        coin_direction = direction_from_coordinates(self_pos,
                                                    self_pos)

    ''' DIRECTION TO CLOSEST CRATE '''
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

        crates_direction = direction_from_coordinates(self_pos,
                                                      coord_to_closest_crate)
                                                    
        

    else:
        crates_direction = direction_from_coordinates(self_pos,
                                                     self_pos)

    ''' 5-vector of risks around us and at our position '''
    risk_map = compute_risk_map(game_state)
    risk_factors = np.zeros((4,))

    x,y = self_pos
    risk_factors[0] = risk_map[(x+1,y)]
    risk_factors[1] = risk_map[(x-1,y)]
    risk_factors[2] = risk_map[(x,y+1)]
    risk_factors[3] = risk_map[(x,y-1)]

    lowest_risk_direction = np.zeros((4,))
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
    elif n_escape_squares < 8:
        bomb_safety = 0
    else:
        bomb_safety = 1

    ''' USEFUL BOMB '''
    bomb_useful = bomb_usefulness(game_state)

    features = [
        field_around.ravel(),
        coin_direction.ravel(),
        crates_direction.ravel(),
        lowest_risk_direction.ravel(),
        [bomb_allowed],
        [bomb_safety],
        [bomb_useful]
    ]
    
    if with_feature_list:
        feature_names = {
            'field around' : 0,
            'coin direction': 0,
            'crate direction' : 0,
            'lowest risk direction' : 0,
            'bomb allowed' : 0,
            'bomb safety' : 0,
            'bomb useful' : 0
        }

        for i, feature in enumerate(feature_names):
            feature_names[feature] = len(features[i])

        return np.concatenate(features), feature_names

    return np.concatenate(features)

