class WizardJudger:

    @staticmethod
    def judge_winner(players):
        """
        整局游戏胜利者判断
        标准：得分高的为胜者
        """
        winner_id = -1
        winner = None

        for player in players:
            if winner_id == -1 or player.point > winner.point:
                winner_id = player.player_id
                winner = player

        return winner_id

    @staticmethod
    def judge_dun_winner(dun_cards, ace_color, dun_color):
        """
        当前墩胜利者判断
        标准：
        ① 有人出巫师牌，第一个打出巫师牌的人胜
        ② 没人出巫师牌，打出王牌花色且点数最大的玩家胜
        ③ 没人出王牌花色，打出本墩花色且点数最大的玩家胜
        ④ 所有人都出小丑牌，第一个出的人胜
        """
        winner_id = -1  # 赢家id
        winner_card_type = -1  # 赢家牌的类型：【0：小丑】【1：其他色】【2：本墩色】【3：王牌色】【4：巫师】
        winner_card = -1

        for dun_card in dun_cards:
            player_id = dun_card[0]
            card = dun_card[1]
            if card >= 56:  # 小丑
                card_type = 0
            elif card >= 52:  # 巫师
                card_type = 4
            elif card % 4 == ace_color:  # 王牌色
                card_type = 3
            elif card % 4 == dun_color:  # 本墩色
                card_type = 2
            else:
                card_type = 1

            if winner_id == -1 or card_type > winner_card_type:
                winner_id = player_id
                winner_card_type = card_type
                winner_card = card
            elif card_type == winner_card_type:
                # 小丑和巫师不用管
                if card < 52 and card // 4 > winner_card // 4:
                    winner_id = player_id
                    winner_card_type = card_type
                    winner_card = card

        return winner_id
