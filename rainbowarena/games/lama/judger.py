class LamaJudger:

    @staticmethod
    def judge_winner(players):
        """
        整局游戏结束的判断
        结束条件：
        ① 有玩家扣分40以上
        """
        winner_id = -1  # 赢家id，如果为-1代表目前没分出胜负
        winner = None
        is_over = False

        # 结束条件①:扣分40以上
        for player in players:
            if player.point >= 40:
                is_over = True
                break

        if is_over is False:
            return winner_id
        else:
            for player in players:
                if winner_id == -1 or player.point < winner.point:
                    winner_id = player.player_id
                    winner = player

        return winner_id

    @staticmethod
    def judge_round_over(players):
        """
        结束条件：
        ① 有玩家出光手牌
        ② 所有玩家都退出
        """
        is_over_1 = False
        is_over_2 = True

        # 结束条件①:出光手牌
        for player in players:
            if sum(player.cards) == 0:
                is_over_1 = True
                break

        # 结束条件②：都退出
        for player in players:
            if player.quit == 0:
                is_over_2 = False
                break

        is_over = is_over_1 or is_over_2

        return is_over
