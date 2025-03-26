class WizardJudger:

    @staticmethod
    def judge_winner(players):
        """
        整局游戏胜利者判断
        """
        winner_id = -1

        for player in players:
            if player.finished_circles == 3:
                winner_id = player.player_id
                break

        return winner_id

    @staticmethod
    def judge_5_connect(players):
        """
        判断是否连成了5子
        """
        pass
