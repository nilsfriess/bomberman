from typing import List

import events as e

NO_COIN_COLLECTED = 'NO_COIN_COLLECTED'
VALID_ACTION = 'VALID_ACTION'
DODGED_BOMB = 'DODGED_BOMB'

def setup_custom_vars(self):
    self.waited = 0
    self.invalid = 0
    self.bomb_dodged = 0

def reward_from_events(events: List[str]) -> int:
    game_rewards = {
        #e.COIN_COLLECTED: 10,
        #e.CRATE_DESTROYED: 30,
        #e.BOMB_DROPPED: 50,
        # NO_COIN_COLLECTED: -2,
        e.WAITED: -5,
        e.INVALID_ACTION: -10,
        #MOVED_AWAY_FROM_COIN: -1,
        #MOVED_TOWARDS_COIN: 1,
        VALID_ACTION: -1,
        DODGED_BOMB: 1000
        # e.KILLED_OPPONENT: 5,
        # PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    # self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def compute_custom_events(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):    
    new_events = []
    
    if e.COIN_COLLECTED not in events:
        new_events.append(NO_COIN_COLLECTED)

    if e.INVALID_ACTION not in events:
        new_events.append(VALID_ACTION)

    if e.INVALID_ACTION in events:
        self.invalid += 1

    if e.WAITED in events:
        self.waited += 1

    if e.BOMB_EXPLODED in events and not e.KILLED_SELF in events and not e.GOT_KILLED in events:
        new_events.append("DODGED_BOMB")
        self.bomb_dodged += 1

    return new_events

def print_progress(self, last_game_state, last_action, events):
    print(f"Coins collected: {last_game_state['self'][1]}")
    print(f"Waited {self.waited} times")
    self.waited = 0
    print(f"Invalid moves: {self.invalid}")
    self.invalid = 0
    print(f"Bombs dodged: {self.bomb_dodged}")
    self.bomb_dodged = 0
    print(f"Survived {last_game_state['step']} steps")
    print()
