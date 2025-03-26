"""
整体的牌局 （自己的环：1，自己的棋子2，对手的环：3，对手的棋子：4, 空：0）
玩家已完成的圆环数目
"""

class WizardPlayer:

    def __init__(self, player_id=-1):
        self.player_id = player_id
        self.finished_circles = 0
        self.board = [[0] * 10] * 11


