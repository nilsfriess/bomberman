import numpy as np

from settings import ROWS, COLS

from .base_helpers import *


def bomb_score(drop_coord, field, others) -> np.int8:
    

    return 0


# array containing the bomb_score for each tile in neighbourhood and at (x,y)
def bomb_score_nb(x, y, neighbourhood, field, others) -> np.array:
    directions = np.zeros(len(neighbourhood)+1).astype(np.int8)

    return directions


# array one-hot-encoding one tile of neighbourhood that has maximum the maximum reachable bomb score in n_steps.
def potential_bomb_score(neighbourhood, field, others, n_steps = 10, discount = 0.9) -> np.array:
    directions = np.zeros(len(neighbourhood)).astype(np.int8)

    return directions
