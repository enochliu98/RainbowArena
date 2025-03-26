# -*- coding: utf-8 -*-
""" Implement Ticket2Ride Player class
"""
import functools


class Ticket2RidePlayer:

    def __init__(self, player_id=-1):
        """ Give the player an id in one game
        """
        self.player_id = player_id  # 玩家编号

        # 【车厢】
        self.train_cars = 10  # 火车车厢数
        self.train_car_cards = [0] * 5  # 火车车厢卡
        # 【车票】
        self.ticket_cards = []  # 车票卡
        # 【线路】
        self.line_map_valid = [0] * 16  # 记录当前玩家已经连通的线路
        self.longest_line = 0  # 最长线路
        self.board_map_valid = [[0] * 8] * 8  # 记录当前玩家目前已经连通的地图
        # 【分数】
        self.point = 0  # 分数
