import os
import json
import numpy as np
import copy
from collections import OrderedDict
from rainbowarena.games.ticket2ride_small_v2.card import Ticket2RideTicketCard

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

"""
线路：3共16条线路
每条线路的信息：
【0】编号 【1】长度 【2】颜色 【3】起点 【4】终点 【5】是否占用 【6】是否双线
"""
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

"""
线路索引:由于在进行动作选择过程中，当选择修建不限制颜色线路时可以用多种颜色的车厢，所以可选动作不再一对一而是多对一
两部分信息：
LINE_INDEX：线路编号 -> 第一个动作编号
ACTION_LINE：动作编号 -> 线路编号
"""
LINE_ACTION = []  # 线路编号 -> 第一个动作编号
ACTION_LINE = []  # 动作编号 -> 线路编号
idx = 0
for line in LINES:
    if line[2] < 5:
        LINE_ACTION.append(idx)
        ACTION_LINE.append(line[0])
        idx += 1
    else:
        LINE_ACTION.append(idx)
        ACTION_LINE.extend([line[0]] * 4)  # 4种可选颜色 + (1万能色,前面4种可选颜色可以覆盖这种情况) = 4
        idx += 4

"""
地图：是一个三维数组，是对称的
1-2维：8个站点互相连接构成的二维矩阵
3维：每对站点连接中包含的线路（可能不止一条）
"""
CITY_MAP = []
for _ in range(8):
    list_tmp = []
    for _ in range(8):
        list_tmp.append([])
    CITY_MAP.append(list_tmp)

for line in LINES:
    CITY_MAP[line[3]][line[4]].append(line[0])
    CITY_MAP[line[4]][line[3]].append(line[0])

"""
车票：共8张车票
[0] 编号 【1】起点 【2】终点 【3】分数
"""
TICKET_CARDS = [
    [0, 0, 4, 4],
    [1, 0, 7, 5],
    [2, 1, 4, 3],
    [3, 1, 7, 5],
    [4, 1, 6, 4],
    [5, 2, 4, 3],
    [6, 3, 7, 4],
    [7, 5, 6, 2]
]


def init_board_map():
    """
    地图 包含两部分：
    ① 线路记录
    ② 地图
    """
    board_map_part1 = copy.deepcopy(LINES)
    board_map_part2 = copy.deepcopy(CITY_MAP)
    board_map = [board_map_part1, board_map_part2]
    return board_map


def init_lines():
    """
    初始化其他线路相关信息：
    ① 线路
    ② 线路动作对应关系
    ③ 动作线路对应关系
    """
    lines = copy.deepcopy(LINES)
    line_index = copy.deepcopy(LINE_ACTION)
    action_line = copy.deepcopy(ACTION_LINE)
    return lines, line_index, action_line


def init_train_car_cards():
    """
    车厢卡
    """
    train_car_cards = [0] * 8 + [1] * 6 + [2] * 6 + [3] * 6 + [4] * 6
    return train_car_cards


def init_ticket_cards():
    """
    车票卡
    """
    ticket_cards = [Ticket2RideTicketCard(ticket_card[0], ticket_card[1], ticket_card[2], ticket_card[3]) \
                    for ticket_card in TICKET_CARDS]
    return ticket_cards
