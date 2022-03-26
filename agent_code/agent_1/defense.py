import numpy as np

from settings import ROWS, COLS

from .base_helpers import *
from .helpers_local import *

# coord is supp to be different from bomb_coord!
def steps_needed_to_safety(coord, bomb_coord, field) -> np.int8:
    # if blocked or no way out or outside field, return 5
    max_steps = 5
    if field[coord] != 0:
        return max_steps
    if not (coord[0]<COLS and coord[1]<ROWS and coord[0]>=0 and coord[1]>=0):
        return max_steps

    # locate bomb
    if coord[0] == bomb_coord[0]:
        di = 1
    elif coord[1] == bomb_coord[1]:
        di = 0
    else:
        return 0


    delta = abs(bomb_coord[di] - coord[di])
    # bomb doesnt affect tile
    if delta > 3:
        return 0
    sign = np.sign(bomb_coord[di] - coord[di])
    blocked = False
    if di == 0:
        for i in range(1,delta):
            if field[coord[0] + sign * i, coord[1]] != 0:
                blocked = True
                break
    else:
        for i in range(1,delta):
            if field[coord[0], coord[1] + sign * i] != 0:
                blocked = True
                break
    if blocked:
        return 0

    # bomb affects tile, now iterate over step lengths
    away_blocked = False
    close_blocked = False
    if di == 0:
        y = coord[1]
        for corridor_length in range(4):
            test_close = coord[0] + sign * corridor_length
            test_away = coord[0] - sign * corridor_length
            if not away_blocked:
                if field[test_away, y] != 0:
                    away_blocked = True
            if corridor_length == delta:
                close_blocked = True

            if (corridor_length == 4-delta) and not away_blocked:
                return corridor_length

            if away_blocked and close_blocked:
                break

            # if not returned or loop broken, path to resp test is free
            if not away_blocked:
                if field[test_away, y+1] == 0 or field[test_away, y-1] == 0:
                    return corridor_length + 1

            if not close_blocked:
                if field[test_close, y+1] == 0 or field[test_close, y-1] == 0:
                    return corridor_length + 1

    else: #di == 1
        x = coord[0]
        for corridor_length in range(4):
            test_close = coord[1] + sign * corridor_length
            test_away = coord[1] - sign * corridor_length
            if not away_blocked:
                if field[x, test_away] != 0:
                    away_blocked = True
            if corridor_length == delta:
                close_blocked = True

            if corridor_length == 4-delta and not away_blocked:
                return corridor_length

            if away_blocked and close_blocked:
                break

            # if not returned or loop broken, path to resp test is free
            if not away_blocked:
                if field[x+1, test_away] == 0 or field[x-1,test_away] == 0:
                    return corridor_length + 1

            if not close_blocked:
                if field[x+1, test_close] == 0 or field[x-1, test_close] == 0:
                    return corridor_length + 1


    # if not returned by now, needs more than 4 steps, i.e. is always deadly.
    return max_steps




# array containing the number of steps_needed_to_safety for each tile in neighbourhood. the last entry is zero if (x,y) is not endangered, otherwise it contains 3 minus the smallest timer of the bombs that are endangering it
def bomb_escape_direction(x, y, neighbourhood, game_state) -> np.array:
    directions = np.zeros(len(neighbourhood)+1).astype(np.int8)

    return directions




def risk_score(coord, field, others) -> np.int8:

    return 0


# array containing a risk score for each tile in neighbourhood and x,y.
def general_risk(x, y, neighbourhood, others) -> np.array:
    directions = np.zeros(len(neighbourhood)+1).astype(np.int8)

    return directions


# sum the offensive and defensive scores from the current states to calculate rewards
def state_score(game_state) -> np.int8:

    return 0


# returns a list of actions
def valid_nondeadly_actions(game_state):

    valid = valid_actions(game_state)
    easy_death_actions = death_implying_actions(game_state)

    leftovers = np.setdiff1d(valid, easy_death_actions)

    if leftovers.shape[0] == 0:
        return []

    actions = []
    x, y = game_state["self"][3]
    field = game_state["field"]
    for index, a in enumerate(leftovers):
        a_is_deadly = False

        if len(game_state["bombs"]) > 0:
            if a == "BOMB" or a == "WAIT":
                check_here = (x,y)
            elif a == "RIGHT":
                check_here = (x+1,y)
            elif a == "LEFT":
                check_here = (x-1,y)
            elif a == "DOWN":
                check_here = (x,y+1)
            elif a == "UP":
                check_here = (x,y-1)

            for ((x_b,y_b),t) in game_state["bombs"]:
                if (x_b == check_here[0] and y_b == check_here[1]) or t == 0:
                    # already included in easy_death_actions
                    continue

                if t < steps_needed_to_safety(check_here, (x_b,y_b), field):
                    a_is_deadly = True
                    break

        if not a_is_deadly:
            actions.append(a)

    return actions
