import trueskill
from trueskill import Rating, rate, TrueSkill
import numpy as np
import itertools
from collections import deque
from tqdm import tqdm

num_policies = 4
num_players = 2

# Names = ['random',
#          'sp_1', 'sp_2', 'sp_3', 'sp_4', 'sp_5', 'sp_6', 'sp_7', 'sp_8', 'sp_9', 'sp_10',
#          'fp_1', 'fp_2', 'fp_3', 'fp_4', 'fp_5', 'fp_6', 'fp_7', 'fp_8', 'fp_9', 'fp_10',
#          'psro_1', 'psro_2', 'psro_3', 'psro_4', 'psro_5', 'psro_6', 'psro_7', 'psro_8', 'psro_9', 'psro_10',
#          ]
# Names = ['random',
#          'sp_1', 'sp_2', 'sp_3', 'sp_4', 'sp_5', 'sp_6',
#          'fp_1', 'fp_2', 'fp_3', 'fp_4', 'fp_5', 'fp_6',
#          'psro_1', 'psro_2', 'psro_3', 'psro_4', 'psro_5', 'psro_6',
#          ]
Names = ['sp', 'rl', 'llm', 'random']
env = TrueSkill(backend='mpmath')
Ratings = [env.create_rating() for i in range(num_policies)]
print(Ratings)

# results = np.load("SP/ticket2ride.npy")
# result_1 = np.array([[0., 0.2, 0.96, 0.92],
#                      [-0.2, 0., 0.76, 0.12],
#                      [-0.96, -0.76, 0., -0.84],
#                      [-0.92, -0.12, 0.84, 0.]])

result_1 = np.array([[0., 1., 0.88, 0.8],
                     [-1., 0., -0.04, 0.52],
                     [-0.88, 0.04, 0., 0.68],
                     [-0.8, -0.52, -0.68, 0.]])

results = [result_1, -result_1]

ranges = [range(num_policies)] * num_players
ranges = list(itertools.product(*ranges))


def rank_elements(lst):
    # 获取元素按大小排序后的索引
    sorted_indices = sorted(range(len(lst)), key=lambda x: lst[x], reverse=True)

    # 创建排名列表
    ranks = [0] * len(lst)

    # 将排序后的元素赋予从 0 开始的排名
    for rank, index in enumerate(sorted_indices):
        ranks[index] = rank

    return ranks


for indices in tqdm(ranges):
    payoffs = [results[i][indices] for i in range(num_players)]
    ranks = rank_elements(payoffs)

    rating_groups = [(Ratings[indices[i]],) for i in range(num_players)]
    rating_results = env.rate(rating_groups=rating_groups, ranks=ranks)

    for i in range(num_players):
        Ratings[indices[i]] = rating_results[i][0]

print(Ratings)

Ratings_Names = [(Names[i], Ratings[i].mu - 3 * Ratings[i].sigma) for i in range(num_policies)]

Ratings_Names = sorted(Ratings_Names, key=lambda x: x[1])

for i in range(num_policies):
    print(Ratings_Names[i])
