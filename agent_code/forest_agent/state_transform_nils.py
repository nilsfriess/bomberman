import numpy as np
from scipy.spatial.distance import cdist

from base_helpers import find_next_step_to_assets,\
    direction_from_coordinates,\
    cityblock_dist,\
    ACTIONS,\
    rotate_game_state,\
    rotate_action

from action_filter import action_is_stupid

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
    if len(game_coins) > 0:
        dist_to_coins = cdist(game_coins, [self_pos], 'cityblock')

        n_closest_coins = min(len(game_coins), 5)
        field = np.array(game_state['field'])
        coins = game_coins[np.argpartition(dist_to_coins.ravel(),
                                           n_closest_coins-1)]
        closest_coins = coins[:n_closest_coins]

        enemies = [(x,y) for (_,_,_,(x,y)) in game_state['others']]
        coord_to_closest_coin = find_next_step_to_assets(field,
                                                         enemies,
                                                         self_pos,
                                                         closest_coins)

        coin_direction = direction_from_coordinates(self_pos,
                                                    coord_to_closest_coin)
    else:
        coin_direction = np.array(self_pos)

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

    ''' 7x7 window around agent of blocked (wall/crate), free tiles and explosions '''
    crates_window = np.zeros((7,7))
    x,y = self_pos
    for i in [-3,-2,-1,0,1,2,3]:
        for j in [-3,-2,-1,0,1,2,3]:
            coord_on_field = (x+i, y+j)

            if (x+i < 0)\
               or (x+i >= field.shape[0])\
               or (y+j < 0)\
               or (y+j >= field.shape[1]):
                continue
            else:
                if field[coord_on_field] == 1:
                    crates_window[i+2,j+2] = 1

    explosions = game_state['explosion_map']
    explosion_window = np.zeros((7,7))

    for i in [-3,-2,-1,0,1,2,3]:
        for j in [-3,-2,-1,0,1,2,3]:
            coord_on_field = (x+i, y+j)

            if (x+i < 0)\
               or (x+i >= field.shape[0])\
               or (y+j < 0)\
               or (y+j >= field.shape[1]):
                continue
            else:
                if explosions[coord_on_field] > 1:
                    explosion_window[i+2,j+2] = 1

    bomb_pos  = [pos for (pos,_) in game_state['bombs']]
    bomb_vals = [val for (_,val) in game_state['bombs']]
    bombs_window = np.zeros((7,7))

    if len(bomb_pos) > 0:
        for i in [-3,-2,-1,0,1,2,3]:
            for j in [-3,-2,-1,0,1,2,3]:
                coord_on_field = (x+i, y+j)

                if (x+i < 0)\
                   or (x+i >= field.shape[0])\
                   or (y+j < 0)\
                   or (y+j >= field.shape[1]):
                    continue
                else:
                    if coord_on_field in bomb_pos:
                        bomb_index = bomb_pos.index(coord_on_field)
                        bombs_window[i+2,j+2] = bomb_vals[bomb_index]

    # ''' DISTANCE TO CLOSEST BOMB '''
    # bombs_dist = [cityblock_dist(self_pos, bomb) for bomb in bombs]
    # if len(bombs_dist) == 0:
    #     min_bomb_dist = -1
    # else:
    #     min_bomb_dist = min(bombs_dist)

    features = np.concatenate([
        #crate_direction.ravel(),
        own_position.ravel(),
        coin_direction.ravel(),
        #closest_enemy_direction.ravel(),
        # explosions_around.ravel(),
        # risks.ravel(),
        crates_window.ravel(),
        explosion_window.ravel(),#
        bombs_window.ravel(),
        #closest_bomb_direction.ravel(),
        #coord_to_closest_bomb.ravel()
        #risk.ravel(),
        #[min_bomb_dist]
    ])

    #print(len(features))

    return features


def random_action(allow_bombs = True):
    if allow_bombs:
        return np.random.choice(ACTIONS)
    else:
        return np.random.choice(ACTIONS[:-1])

def train_act(self, game_state:dict) -> str:
    filter_prob = 0.9

    # Compute stupid actions
    stupid_actions = []

    if np.random.uniform() < filter_prob:
        for action in ACTIONS:
            if action_is_stupid(game_state, action):
                stupid_actions.append(action)

        if ('BOMB' not in stupid_actions) and (len(stupid_actions) == 5):
            # Too late, every direction is stupid
            stupid_actions = []

        if (len(stupid_actions) == 6):
            stupid_actions = []

    if np.random.uniform() < 1-self.epsilon:
        '''
        Check which quadrant we are in.
        '''
        (_,_,_,(x,y)) = game_state['self']
        if (x <= 8) and (y <= 8):
            # upper left
            quad = 0
        elif (x > 8) and (y <= 8):
            # upper right
            quad = 3
        elif (x > 8) and (y > 8):
            # lower right
            quad = 2
        elif (x <= 8) and (y > 8):
            quad = 1

        game_state = rotate_game_state(game_state, quad)

#        print(f"Position after rotation {game_state['self'][3]}")

        state = state_to_features(game_state)
        av = np.array([self.QEstimator.estimate(state, action) for action in ACTIONS])

        action = ACTIONS[np.argmax(av)]
        action = rotate_action(action, quad)

        # while action in stupid_actions:
        #     action = random_action()

    else:
        action = random_action(False)
        # while action in stupid_actions:
        #     action = random_action()

    # print(f"Chose: {action}")
    return action
