import numpy as np
import pygambit as gbt

dim = 2
level_results = []
policies = ['random',
            'SP/ticket2ride_sp/1.pt', 'SP/ticket2ride_sp/2.pt', 'SP/ticket2ride_sp/3.pt', 'SP/ticket2ride_sp/4.pt', 'SP/ticket2ride_sp/5.pt',
            'SP/ticket2ride_sp/6.pt', 'SP/ticket2ride_sp/7.pt', 'SP/ticket2ride_sp/8.pt', 'SP/ticket2ride_sp/9.pt', 'SP/ticket2ride_sp/10.pt',
            'SP/ticket2ride_fp/1.pt', 'SP/ticket2ride_fp/2.pt', 'SP/ticket2ride_fp/3.pt', 'SP/ticket2ride_fp/4.pt', 'SP/ticket2ride_fp/5.pt',
            'SP/ticket2ride_fp/6.pt', 'SP/ticket2ride_fp/7.pt', 'SP/ticket2ride_fp/8.pt', 'SP/ticket2ride_fp/9.pt', 'SP/ticket2ride_fp/10.pt',
            'SP/ticket2ride_psro/1.pt', 'SP/ticket2ride_psro/2.pt', 'SP/ticket2ride_psro/3.pt', 'SP/ticket2ride_psro/4.pt', 'SP/ticket2ride_psro/5.pt',
            'SP/ticket2ride_psro/6.pt', 'SP/ticket2ride_psro/7.pt', 'SP/ticket2ride_psro/8.pt', 'SP/ticket2ride_psro/9.pt', 'SP/ticket2ride_psro/10.pt',
            ]

infos = [(policy, idx) for idx, policy in enumerate(policies)]

# meta_payoff = np.load("SP/ticket2ride.npy")
meta_payoff = np.load("SP/ticket2ride.npy")
print(meta_payoff)

while meta_payoff.shape[1] > 0:
    g = gbt.Game.from_arrays(*meta_payoff)
    result = gbt.nash.logit_solve(g)
    result = result.equilibria[0][g.players[0]]
    p0_sol = np.array([item[1] for item in result])

    # print(p0_sol)
    indices = np.argwhere(p0_sol >= 0.01)

    current_level_result = [(infos[indice[0]][0], infos[indice[0]][1], p0_sol[indice[0]]) for indice in indices]
    level_results.append(current_level_result)

    # print(current_level_result)

    for indice in reversed(indices):
        indice = indice[0]
        del infos[indice]
        for i in range(dim):
            meta_payoff = np.delete(meta_payoff, indice, axis=i + 1)

    # print(meta_payoff.shape)

for r in level_results:
    print(r)

# # 删除第一个维度中索引为 4 的内容
# arr = np.delete(arr, 4, axis=0)
#
# # 删除第二个维度中索引为 4 的内容
# arr = np.delete(arr, 4, axis=1)
#
# # 删除第三个维度中索引为 4 的内容
# arr = np.delete(arr, 4, axis=2)
#
# print(arr.shape)  # 打印最终矩阵的维度
