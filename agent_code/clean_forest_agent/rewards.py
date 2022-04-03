import events as e

VALID_ACTION = 'VALID_ACTION'

DECREASED_RISK = 'DECREASED_RISK'
INCREASED_RISK = 'INCREASED_RISK'

USEFUL_BOMB = 'USEFUL_BOMB'
USELESS_BOMB = 'USELESS_BOMB'

WALKED_AWAY_FROM_TARGET = 'WALKED_AWAY_FROM_TARGET'
WALKED_TOWARDS_TARGET = 'WALKED_TOWARDS_TARGET'

TOOK_ZERO_RISK_DIRECTION = 'TOOK_ZERO_RISK_DIRECTION'
DID_NOT_TAKE_ZERO_RISK_DIRECTION = 'DID_NOT_TAKE_ZERO_RISK_DIRECTION'

WAITED_IN_RISK = 'WAITED_IN_RISK'

def reward_from_events(events):
    game_rewards = {
        e.KILLED_OPPONENT: 50,
        e.COIN_COLLECTED: 10,
        e.INVALID_ACTION: -10,
        VALID_ACTION: -5, # Valid actions are only good if the serve a purpose
        USELESS_BOMB: -400, # must be <= -4*DECREASED_RISK
        USEFUL_BOMB: 20,
        WALKED_TOWARDS_TARGET: 10,
        WALKED_AWAY_FROM_TARGET: -40,
        DECREASED_RISK: 80, # Not taking the correct direction but decreasing risk is ok
        INCREASED_RISK: -100,
        TOOK_ZERO_RISK_DIRECTION: 40,
        DID_NOT_TAKE_ZERO_RISK_DIRECTION: -40,
        e.KILLED_SELF: -100,
        WAITED_IN_RISK: -1000
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    # self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
