import numpy as np
from scipy.spatial.distance import cdist

from .base_helpers import find_next_step_to_assets, direction_from_coordinates, compute_risk_map
from .bomb_helpers import bomb_usefulness, should_drop_bomb
from .state_action_helpers import one_hot_action

def state_to_features(game_state: dict) -> np.array:
    if game_state is None:
        return np.array([])
    
    # Assemble features
    ''' OWN POSITION '''
    (_,_,_,self_pos) = game_state['self']
    field = np.array(game_state['field'])
    own_position = np.zeros((field.shape[0], field.shape[1]))
    own_position[self_pos] = 1
    #own_position = np.array(self_pos)

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

    # ''' ENEMY DIRECTIONS '''
    # coord_to_closest_enemy = find_next_step_to_assets(field,
    #                                                   [],
    #                                                   self_pos,
    #                                                   enemies)
    # closest_enemy_direction = direction_from_coordinates(self_pos,
    #                                                      coord_to_closest_enemy)

    # ''' BOMB DIRECTION '''
    # bombs = [pos for (pos, _) in game_state['bombs']]
    # coord_to_closest_bomb = find_next_step_to_assets(field,
    #                                                  [],
    #                                                  self_pos,
    #                                                  bombs)
    # closest_bomb_direction = direction_from_coordinates(self_pos,
    #                                                     coord_to_closest_bomb)


    # ''' RISK FACTOR OF THE FOUR DIRECTIONS '''
    # risk = np.zeros((4,1)) # UP, DOWN, LEFT, RIGHT

    # for i,action in enumerate(['UP', 'DOWN', 'LEFT', 'RIGHT']):
    #         if action_is_stupid(game_state, action):
    #             risk[i] = 1

    ''' USEFUL BOMB '''
    bomb_useful = bomb_usefulness(game_state)   
    
    ''' 4-vector of crates around us '''
    crates_around = np.zeros((4,))
    x,y = self_pos

    if field[(x-1,y)] == 1:
        crates_around[0] = 1
    if field[(x+1,y)] == 1:
        crates_around[1] = 1
    if field[(x,y-1)] == 1:
        crates_around[2] = 1
    if field[(x,y+1)] == 1:
        crates_around[3] = 1

    ''' 5-vector of risks around us and at our position '''
    risk_map = compute_risk_map(game_state)
    risk_factors = np.zeros((5,))

    x,y = self_pos
    risk_factors[0] = risk_map[(x,y)]
    risk_factors[1] = risk_map[(x+1,y)]
    risk_factors[2] = risk_map[(x-1,y)]
    risk_factors[3] = risk_map[(x,y+1)]
    risk_factors[4] = risk_map[(x,y-1)]

    ''' 5-vector that is one if a direction is a zero-risk direction '''
    zero_risk_direction = np.zeros_like(risk_factors)
    zero_risk_direction[risk_factors == 0] = 1

    ''' Safety of dropping bomb here '''
    bomb_safety, bomb_safety_action = should_drop_bomb(game_state)
    bomb_safety_action_one_hot = one_hot_action(bomb_safety_action)
                
    ''' Is dropping a bomb a valid move '''
    bomb_allowed = int(game_state['self'][2])

    features = np.concatenate([
        #crate_direction.ravel(),
        field.ravel(),
        own_position.ravel(),
        coin_direction.ravel(),
        #closest_enemy_direction.ravel(),
        # explosions_around.ravel(),
        # risks.ravel(),
        #crates_window.ravel(),
        #explosion_window.ravel(),#
        #bombs_window.ravel(),
        crates_around.ravel(),
        #closest_bomb_direction.ravel(),
        #coord_to_closest_bomb.ravel()
        #risk.ravel(),
        #[min_bomb_dist]
        risk_factors.ravel(),
        zero_risk_direction.ravel(),
        [bomb_allowed],
        [bomb_safety],
        [bomb_useful]
        #bomb_safety_action_one_hot.ravel()
        #[bomb_useless]
    ])

    #print(len(features))

    return features

