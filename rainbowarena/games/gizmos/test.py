# from judger import SplendorJudger
# from player import SplendorPlayer
# from card import SplendorCard
# from dealer import SplendorDealer
# from round import SplendorRound
# from game import SplendorGame
# import numpy as np

"""
judger测试
"""
# card_0 = SplendorCard()
# card_1 = SplendorCard()
# card_2 = SplendorCard()
# card_3 = SplendorCard()
# card_4 = SplendorCard()

# player_0 = SplendorPlayer(0)
# player_1 = SplendorPlayer(1)
# player_2 = SplendorPlayer(2)
# player_3 = SplendorPlayer(3)

# player_0.point = 13
# player_1.point = 16
# player_2.point = 16
# player_3.point = 16

# player_1.current_cards = [card_0, card_1, card_2, card_3]
# player_2.current_cards = [card_0, card_1]
# player_3.current_cards = [card_0, card_1]

# players = [player_0, player_1, player_2, player_3]

# judger = SplendorJudger()

# print(judger.judge_winner(players, 1))


"""
dealer测试
"""

# dealer = SplendorDealer(np.random, 3)

# print(len(dealer.cards[0]), len(dealer.cards[1]), len(dealer.cards[2]), len(dealer.cards[3]))

# print(dealer.tokens)

# print(len(dealer.deck[0]), len(dealer.deck[1]), len(dealer.deck[2]), len(dealer.deck[3]))


"""
round测试
"""
# game = SplendorGame(2)

# game.init_game()

# print(game.round.get_legal_actions(game.players[0]))
# print(game.round.get_legal_actions(game.players[1]))


"""
other test
"""

# orgin = np.zeros(10)

# a = np.array([1,2,3])

# orgin[:3] = a

# print(orgin)



# import torch

# embedding = torch.nn.Embedding(3, 5)

# print(embedding(torch.LongTensor([1]))[0].detach().numpy())


# import torch
 
# GRU=torch.nn.GRU(input_size=10,
#                  hidden_size=20,
#                  num_layers=20)
 
# input_tensor=torch.randn(5,3,10)
# '''
# 输入的sequence长5
# batch_size为3
# 输入sequence每一个元素的维度为10
# '''
# h0=torch.randn(1*20,3,20)
# '''
# 第一个参数：单方向GRU（1），20层GRU（20）
# 第二个参数：batch_size
# 第三个参数：hidden_size的大小
# '''
# output,hn=GRU(input_tensor,h0)
# output.shape,hn.shape
# #(torch.Size([5, 3, 20]), torch.Size([20, 3, 20]))


# import copy

# a = [1,2,4]

# b = a[:]

# c = b

# d = c

# c[0]=10

# print(a)


print(3//4)
