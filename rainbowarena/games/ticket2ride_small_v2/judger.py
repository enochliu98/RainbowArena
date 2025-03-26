class Ticket2RideJudger:

    def __init__(self, np_random, num_players):
        self.random_player = np_random.randint(0, num_players)  # 分不出胜负的情况

    def judge_winner(self, players):
        """
            判断游戏赢家
            ① 初始化
            ② 判断胜者
        """

        # 【初始化】
        winner_player = None
        winner_id = -1  # 赢家id，如果为-1代表目前没分出胜负
        winner_ticket_cards_count = -1  # 赢家的目的地车票数目

        # 【判断胜者】
        for i, player in enumerate(players):
            # 第一步分数计算：判断车票是否达成
            ticket_cards_count = 0
            for ticket_card in player.ticket_cards:
                ticket_card_success = check_ticket_card(player.board_map_valid, ticket_card.start_pos, ticket_card.end_pos)  # 判断车票是否达成
                if ticket_card_success:
                    player.point += ticket_card.point
                    ticket_cards_count += 1
                else:
                    player.point -= ticket_card.point

            if winner_id == -1 or player.point > winner_player.point:
                winner_id = i
                winner_player = player
                winner_ticket_cards_count = ticket_cards_count
            elif player.point == winner_player.point:
                if winner_ticket_cards_count < ticket_cards_count:
                    winner_id = i
                    winner_player = player
                    winner_ticket_cards_count = ticket_cards_count
                elif winner_ticket_cards_count == ticket_cards_count:
                    winner_id = self.random_player
                    winner_player = players[winner_id]


        return winner_id


def check_ticket_card(board_map_valid, source, destination):

    def find(x):
        if p[x] != x:
            p[x] = find(p[x])
        return p[x]

    edges = []
    n = len(board_map_valid)
    for i in range(n):
        for j in range(n):
            if i < j and board_map_valid[i][j] == 1:
                edges.append([i, j])
    p = list(range(n))
    for u, v in edges:
        p[find(u)] = find(v)
    return find(source) == find(destination)