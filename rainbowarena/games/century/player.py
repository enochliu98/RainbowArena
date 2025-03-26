# -*- coding: utf-8 -*-
import functools


class CenturyPlayer:

    def __init__(self, player_id=-1):
        """
            玩家信息：
            ① 编号
            ② 功能卡
            ③ 分数卡
            ④ 香料
            ⑤ 分数
        """
        # 【编号】
        self.player_id = player_id  # 玩家编号
        # 【功能卡】
        self.function_cards = []  # 手里的功能卡
        self.played_function_cards = []  # 打出的功能卡
        self.map_function_cards = [-1] * 53  # 每张功能卡都有对应编号，需要维护一个由该固定编号映射到实际玩家手牌的字典
        # 【分数卡】
        self.point_cards = []  # 分数卡
        # 【香料】
        self.tokens = [0] * 4  # 每种方块的数目
        # 【分数】
        self.point = 0  # 分数
