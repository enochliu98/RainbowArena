class Ticket2RideJudger:

    @staticmethod
    def judge_winner(players, cur_player_id, round_over_player):

        winner = -1  # 赢家id，如果为-1代表目前没分出胜负
        winner_ticket_cards_count = -1  # 赢家的目的地车牌数目
        winner_player = None

        if cur_player_id != round_over_player:  # 如果当前玩家不是回合结束的玩家
            return winner

        # 首先判断本局游戏是否结束
        is_end = False  # 初始化为没结束

        for i, player in enumerate(players):
            if player.train_cars <= 2:  # 当可用火车车厢数为0,1,2时，退出
                is_end = True
                break

        if is_end is False:  # 当前玩家是回合结束的玩家，但没到达终止条件
            return winner

        for i, player in enumerate(players):
            # 第一步分数计算：判断车票是否达成
            ticket_cards_count = 0
            for ticket_card in player.ticket_cards:
                ticket_card_success = check_ticket_card(ticket_card, player.board_map_valid)
                if ticket_card_success:
                    player.point += ticket_card.point
                    ticket_cards_count += 1
                else:
                    player.point -= ticket_card.point

            if winner == -1 or player.point > winner_player.point:
                winner = i
                winner_player = player
                winner_ticket_cards_count = ticket_cards_count
            elif player.point == winner_player.point:
                if winner_ticket_cards_count < ticket_cards_count:
                    winner = i
                    winner_player = player
                    winner_ticket_cards_count = ticket_cards_count

        return winner


def check_ticket_card(ticket_card, board_map_valid):
    start_pos = ticket_card.start_pos
    end_pos = ticket_card.end_pos

    rows = len(board_map_valid)
    cols = len(board_map_valid[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]

    def dfs(x, y):
        if x < 0 or x >= rows or y < 0 or y >= cols or visited[x][y] or board_map_valid[x][y] == 0:
            return False
        visited[x][y] = True
        if (x, y) == end_pos:
            return True
        for dx, dy in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
            if dfs(x + dx, y + dy):
                return True
        visited[x][y] = False
        return False

    return dfs(start_pos[0], start_pos[1])
