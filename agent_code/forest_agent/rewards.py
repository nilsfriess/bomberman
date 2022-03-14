from typing import List

import events as e

MOVED_TOWARDS_COIN = 'MOVED_TOWARDS_COIN'
MOVED_AWAY_FROM_COIN = 'MOVED_AWAY_FROM_COIN'
VALID_ACTION = 'VALID_ACTION'
NO_COIN_COLLECTED = 'NO_COIN_COLLECTED'

STAYED_IN_RISK_REGION = 'STAYED_IN_RISK_REGION'
MOVED_AWAY_FROM_RISK = 'MOVED_AWAY_FROM_RISK'

MOVED_AWAY_FROM_BOMB = 'DID_MOVE_AWAY_FROM_BOMB'
MOVED_NOT_AWAY_FROM_BOMB = 'DID_NOT_MOVE_AWAY_FROM_BOMB'

USEFUL_DIRECTION = 'USEFUL_DIRECTION'
NOT_USEFUL_DIRECTION = 'NOT_USEFUL_DIRECTION'

AVOIDED_BOMB = 'AVOIDED_BOMB'

DID_NOT_SURVIVE = 'DID_NOT_SURVIVE'

BOMB_IN_CORNER = 'BOMB_IN_CORNER'

USEFUL_BOMB = 'USEFUL_BOMB'
USELESS_BOMB = 'USELESS_BOMB'

WAITED_ON_BOMB = 'WAITED_ON_BOMB'

def reward_from_events(events: List[str]) -> int:
    # if (DID_MOVE_AWAY_FROM_BOMB in events) or\
    #    (DID_NOT_MOVE_AWAY_FROM_BOMB in events) or\
    #    (USEFUL_DIRECTION in events) or\
    #    (NOT_USEFUL_DIRECTION in events):
    #     print("Custom event")
    
    game_rewards = {
        e.KILLED_OPPONENT: 500,
        e.COIN_COLLECTED: 100,
        e.CRATE_DESTROYED: 50,
        e.INVALID_ACTION: -10,
        e.KILLED_SELF: -1000,
        VALID_ACTION: -1,
        MOVED_AWAY_FROM_BOMB: 50,
        MOVED_NOT_AWAY_FROM_BOMB: -80,
        MOVED_TOWARDS_COIN: 1,
        MOVED_AWAY_FROM_COIN: -2,
        BOMB_IN_CORNER: -50,
        USELESS_BOMB: -200,
        USEFUL_BOMB: 10,
        WAITED_ON_BOMB: -100
        # DID_NOT_SURVIVE: -1000,
        # e.SURVIVED_ROUND: 2000
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    # self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
