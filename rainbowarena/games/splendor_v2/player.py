class SplendorPlayer:
    def __init__(self, player_id=-1):
        self.player_id = player_id

        self.current_cards = []  # 卡
        self.current_flip_cards = []  # 盖住的卡
        self.current_noble_cards = []  # 贵族卡
        self.current_tokens = [0] * 6  # 每种宝石的数目
        self.current_card_tokens = [0] * 5  # 每种宝石的数目
        self.point = 0


