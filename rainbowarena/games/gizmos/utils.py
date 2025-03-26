import os
import json
import numpy as np
from collections import OrderedDict

# import rlcard

from rainbowarena.games.splendor.card import SplendorCard as Card

# # Read required docs
# ROOT_PATH = rlcard.__path__[0]

# from card import SplendorCard as Card

noble_cards_info = [[[0, 0, 0, 4, 4], 3],
                    [[0, 0, 3, 3, 3], 3],
                    [[3, 0, 3, 0, 3], 3],
                    [[0, 0, 4, 0, 4], 3],
                    [[3, 3, 3, 0, 0], 3],
                    [[0, 4, 0, 4, 0], 3],
                    [[0, 3, 0, 3, 3], 3],
                    [[3, 3, 0, 3, 0], 3],
                    [[4, 4, 0, 0, 0], 3],
                    [[4, 0, 4, 0, 0], 3]]

level1_cards_info = [[[0, 1, 0, 3, 1], 0, 4],
                     [[0, 0, 2, 2, 0], 0, 4],
                     [[2, 0, 0, 0, 2], 0, 3],
                     [[2, 1, 0, 0, 0], 0, 3],
                     [[3, 0, 0, 0, 0], 0, 3],
                     [[1, 0, 1, 1, 2], 0, 1],
                     [[0, 0, 1, 1, 3], 0, 3],
                     [[0, 0, 0, 4, 0], 1, 0],
                     [[0, 3, 0, 0, 0], 0, 2],
                     [[0, 1, 1, 2, 1], 0, 0],
                     [[0, 1, 2, 0, 2], 0, 3],
                     [[3, 1, 1, 0, 0], 0, 2],
                     [[0, 0, 0, 0, 3], 0, 0],
                     [[0, 2, 1, 2, 0], 0, 0],
                     [[0, 2, 0, 1, 0], 0, 4],
                     [[1, 0, 1, 1, 1], 0, 1],
                     [[1, 1, 1, 0, 1], 0, 3],
                     [[1, 1, 2, 0, 1], 0, 3],
                     [[2, 0, 2, 1, 0], 0, 4],
                     [[1, 1, 0, 1, 1], 0, 2],
                     [[1, 0, 3, 0, 1], 0, 0],
                     [[1, 2, 0, 1, 1], 0, 2],
                     [[2, 2, 0, 0, 1], 0, 2],
                     [[0, 0, 3, 0, 0], 0, 4],
                     [[2, 1, 1, 1, 0], 0, 4],
                     [[0, 4, 0, 0, 0], 1, 2],
                     [[0, 0, 0, 0, 4], 1, 1],
                     [[1, 1, 1, 1, 0], 0, 4],
                     [[4, 0, 0, 0, 0], 1, 4],
                     [[0, 0, 4, 0, 0], 1, 3],
                     [[0, 2, 2, 0, 0], 0, 2],
                     [[0, 2, 0, 0, 2], 0, 0],
                     [[0, 0, 0, 3, 0], 0, 1],
                     [[1, 3, 0, 1, 0], 0, 1],
                     [[0, 0, 0, 2, 1], 0, 2],
                     [[1, 0, 0, 2, 2], 0, 1],
                     [[1, 0, 2, 0, 0], 0, 1],
                     [[2, 0, 0, 2, 0], 0, 1],
                     [[0, 1, 1, 1, 1], 0, 0],
                     [[0, 0, 1, 0, 2], 0, 0]]


level2_cards_info = [[[5, 0, 0, 0, 0], 2, 2],
                     [[0, 0, 0, 6, 0], 3, 3],
                     [[0, 0, 0, 5, 0], 2, 3],
                     [[2, 3, 0, 0, 3], 1, 0],
                     [[6, 0, 0, 0, 0], 3, 0],
                     [[3, 0, 0, 2, 3], 1, 3],
                     [[0, 0, 0, 0, 6], 3, 4],
                     [[0, 0, 0, 0, 5], 2, 4],
                     [[0, 3, 0, 2, 2], 1, 0],
                     [[0, 3, 3, 0, 2], 1, 4],
                     [[1, 4, 0, 2, 0], 2, 4],
                     [[0, 0, 6, 0, 0], 3, 2],
                     [[2, 2, 0, 3, 0], 1, 4],
                     [[0, 0, 0, 5, 3], 2, 4],
                     [[0, 6, 0, 0, 0], 3, 1],
                     [[0, 1, 0, 4, 2], 2, 2],
                     [[5, 3, 0, 0, 0], 2, 2],
                     [[4, 2, 1, 0, 0], 2, 3],
                     [[0, 5, 0, 0, 0], 2, 0],
                     [[0, 0, 3, 0, 5], 2, 0],
                     [[0, 0, 3, 2, 2], 1, 3],
                     [[2, 0, 2, 0, 3], 1, 1],
                     [[2, 0, 4, 0, 1], 2, 1],
                     [[3, 0, 2, 3, 0], 1, 2],
                     [[0, 0, 2, 1, 4], 2, 0],
                     [[0, 0, 5, 0, 0], 2, 1],
                     [[0, 2, 3, 3, 0], 1, 1],
                     [[3, 0, 5, 0, 0], 2, 1],
                     [[3, 2, 2, 0, 0], 1, 2],
                     [[0, 5, 0, 3, 0], 2, 3]]


level3_cards_info = [[[7, 0, 0, 0, 0], 4, 1],
                     [[0, 0, 0, 7, 3], 5, 4],
                     [[3, 0, 5, 3, 3], 3, 1],
                     [[0, 7, 0, 3, 0], 5, 3],
                     [[5, 3, 3, 0, 3], 3, 3],
                     [[3, 6, 0, 3, 0], 4, 3],
                     [[0, 0, 0, 0, 7], 4, 2],
                     [[0, 7, 0, 0, 0], 4, 3],
                     [[0, 0, 7, 0, 0], 4, 0],
                     [[3, 0, 6, 0, 3], 4, 0],
                     [[7, 3, 0, 0, 0], 5, 1],
                     [[3, 5, 3, 3, 0], 3, 4],
                     [[0, 3, 3, 3, 5], 3, 0],
                     [[0, 3, 0, 5, 3], 4, 4],
                     [[3, 3, 0, 5, 3], 3, 2],
                     [[3, 0, 7, 0, 0], 5, 0],
                     [[0, 0, 3, 3, 6], 4, 2],
                     [[6, 3, 3, 0, 0], 4, 1],
                     [[0, 0, 0, 7, 0], 4, 4],
                     [[0, 0, 3, 0, 7], 5, 2]]

def init_cards():
    """
        初始化卡组：共四种级别
        第一级别：
        第二级别：
        第三级别：
        贵族卡级别：
    """
    noble_cards = [Card(0, card[0], card[1], -1) for card in noble_cards_info]
    level1_cards = [Card(1, card[0], card[1], card[2]) for card in level1_cards_info]
    level2_cards = [Card(2, card[0], card[1], card[2]) for card in level2_cards_info]
    level3_cards = [Card(3, card[0], card[1], card[2]) for card in level3_cards_info]
    cards = [noble_cards, level1_cards, level2_cards, level3_cards]
    return cards

def init_tokens(num_players):
    """
        初始化宝石数目
        顺序：黑，白，红，蓝，绿，金
    """
    gems = [num_players + 2, num_players + 2, num_players + 2, num_players + 2, num_players + 2, 5]
    return gems
