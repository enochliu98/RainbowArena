import os
import json
import numpy as np
import copy
from collections import OrderedDict
from rainbowarena.games.ticket2ride_small.card import Ticket2RideTicketCard

"""
颜色:
【0】彩
【1】黄
【2】绿
【3】蓝
【4】红
【5】不限制
"""
"""
站点:
【0】
【1】
【2】
【3】
【4】
【5】
"""

# 线路
LINES = [
    [0, 3, 2, 0, 6, 0, 0],
    [1, 1, 4, 0, 2, 0, 0],
    [2, 2, 1, 0, 3, 0, 0],
    [3, 2, 3, 0, 1, 0, 0],
    [4, 1, 5, 1, 3, 0, 0],
    [5, 3, 4, 1, 5, 0, 0],
    [6, 1, 2, 2, 3, 0, 0],
    [7, 2, 1, 2, 6, 0, 0],
    [8, 3, 5, 3, 5, 0, 0],
    [9, 2, 4, 3, 4, 0, 0],
    [10, 3, 3, 3, 6, 0, 0],
    [11, 1, 3, 4, 5, 0, 0],
    [12, 1, 5, 4, 6, 0, 0],
    [13, 2, 1, 4, 7, 0, 0],
    [14, 2, 2, 5, 7, 0, 0],
    [15, 2, 5, 6, 7, 0, 0],
]

# 线路索引
LINE_INDEX = []  # 每条线路idx对应第一个线路索引
ACTION_LINE = []  # 每个动作对应哪个线路索引
idx = 0
for line in LINES:
    if line[2] < 5:
        LINE_INDEX.append(idx)
        ACTION_LINE.append(line[0])
        idx += 1
    else:
        LINE_INDEX.append(idx)
        ACTION_LINE.extend([line[0]] * 5)  # 4种可选颜色 + 1万能色
        idx += 5

# 地图
CITY_MAP = [[[]] * 8] * 8
for line in LINES:
    CITY_MAP[line[3]][line[4]].append(line[0])
    CITY_MAP[line[4]][line[3]].append(line[0])

# 车票
TICKET_CARDS = [
    [0, 5, 4],
    [0, 7, 5],
    [1, 4, 3],
    [1, 7, 5],
    [1, 6, 4],
    [2, 4, 3],
    [3, 7, 4],
    [5, 6, 2]
]


def init_board_map():
    '''
    地图 包含两部分：
    ① 每条线路记录: 0.索引, 1.长度, 2.颜色, 3.起点, 4.终点, 5.是否被占用, 6.是否双线
    ② map: 不同站点之间的连接
    '''
    board_map_part1 = LINES
    board_map_part2 = CITY_MAP
    board_map = [board_map_part1, board_map_part2]
    return board_map


def init_lines():
    '''
    初始化玩家的初始线路
    '''
    lines = copy.deepcopy(LINES)
    line_index = LINE_INDEX
    action_line = ACTION_LINE

    return lines, line_index, action_line


def init_train_car_cards():
    '''
    车厢卡
    '''
    train_car_cards = [0] * 6 + [1] * 6 + [2] * 6 \
                      + [3] * 6 + [4] * 6
    return train_car_cards


def init_ticket_cards():
    '''
    车票卡
    '''
    ticket_cards = [Ticket2RideTicketCard(ticket_card[0], ticket_card[1], ticket_card[2]) \
                    for ticket_card in TICKET_CARDS]
    return ticket_cards
