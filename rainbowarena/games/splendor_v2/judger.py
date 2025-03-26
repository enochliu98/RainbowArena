class SplendorJudger:


    @staticmethod
    def judge_winner(players, round_over_player):
        ''' Judge the winner of the game

        Args:
            players (list): The list of players who play the game

        Returns:
            (list): The player id of the winner
        '''
        
        winner = -1
        winner_player = None

        for i, player in enumerate(players):
            if player.point >= 15:  # 胜利的基本条件
                if winner_player is None or winner_player.point < player.point:  # 分更高
                    winner_player = player
                    winner = i
                elif winner_player.point == player.point:  # 分相等
                    if len(winner_player.current_cards) > len(player.current_cards):
                        winner_player = player
                        winner = i
                    elif len(winner_player.current_cards) == len(player.current_cards):
                        if (round_over_player + len(players) - winner) % len(players) > (round_over_player + len(players) - i) % len(players):
                            winner_player = player
                            winner = i

        return winner

