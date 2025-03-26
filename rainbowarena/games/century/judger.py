class CenturyJudger:
    @staticmethod
    def judge_winner(players, round_over_player):

        winner_id = -1  # 赢家id，如果为-1代表目前没分出胜负
        winner_player = None

        # 首先判断本局游戏是否结束
        is_end = False  # 初始化为没结束
        target_number = 6 if len(players) <= 3 else 5  # 当玩家数2-3(6张)，否则5张
        for i, player in enumerate(players):
            if len(player.point_cards) >= target_number:  # 分数牌数目达到要求达到要求
                is_end = True
                break

        if is_end is False:  # 当前玩家是回合结束的玩家，但是牌数不对
            return winner_id

        for i, player in enumerate(players):
            player.point += sum(player.tokens[1:])  # 加上token对应的分数
            if winner_id == -1 or player.point > winner_player.point:
                winner_id = i
                winner_player = player
            elif player.point == winner_player.point:
                if (round_over_player - i) % len(players) < (round_over_player - winner_id) % len(players):  # 分数相同排位靠后
                    winner_id = i
                    winner_player = player

        return winner_id
