class GongzhuJudger:

    @staticmethod
    def judge_winner(players):
        """
        整局游戏胜利者判断
        标准：得分高的为胜者
        """
        winner_id = -1

        for player in players:
            if winner_id == -1 or player.point > players[winner_id].point:
                winner_id = player.player_id

        return winner_id

    @staticmethod
    def judge_dun_winner(dun_cards, dun_color):
        """
        当前墩胜利者判断
        标准：
        color为dun_color的最大牌
        A最大，2最小
        """
        winner_id = -1  # 赢家id
        winner_card = -1  # 赢家的牌

        for dun_card in dun_cards:
            if winner_id == -1:
                winner_id = dun_card[0]
                winner_card = dun_card[1]
            else:
                if dun_card[1] % 4 == dun_color:
                    if dun_card[1] // 4 == 0:
                        winner_id = dun_card[0]
                        winner_card = dun_card[1]
                    else:
                        if winner_card // 4 != 0 and dun_card[1] // 4 > winner_card // 4:
                            winner_id = dun_card[0]
                            winner_card = dun_card[1]

        return winner_id
