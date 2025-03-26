# act2des = {}
# colors = {
#                 "Bamboo": 0,
#                 "Characters": 9,
#                 "Dots": 18,
#             }
#             # Generate number cards and action cards for each color
# for color, offset in colors.items():
#     act2des.update({
#         **{i + offset: f"{color}-{i+1}" for i in range(10)},
#     })
# act2des.update({
#     27: "Dragons-green",
#     28: "Dragons-red",
#     29: "Dragons-white",
#     30: "Winds-east",
#     31: "Winds-west",
#     32: "Winds-north",
#     33: "Winds-south",
#     34: "Pong",
#     35: "Chow",
#     36: "Gong",
#     37: "Stand",
#                         })
# pos2act = {value:key for (key,value) in act2des.items()}

# print(act2des[37])

# with open("examples/game_results.txt", "w") as results_file:
#     results_file.write("111111")

# card_encoding_dict = {}
# num = 0
# for _type in ['bamboo', 'characters', 'dots']:
#     for _trait in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
#         card = _type+"-"+_trait
#         card_encoding_dict[card] = num
#         num += 1
# for _trait in ['green', 'red', 'white']:
#     card = 'dragons-'+_trait
#     card_encoding_dict[card] = num
#     num += 1

# for _trait in ['east', 'west', 'north', 'south']:
#     card = 'winds-'+_trait
#     card_encoding_dict[card] = num
#     num += 1
# card_encoding_dict['pong'] = num
# card_encoding_dict['chow'] = num + 1
# card_encoding_dict['gong'] = num + 2
# card_encoding_dict['stand'] = num + 3

# card_decoding_dict = {card_encoding_dict[key]: key for key in card_encoding_dict.keys()}

# print(card_decoding_dict)
# tiles = ['characters-8', 'characters-4', 'dragons-green', 'winds-south', 'characters-4', 'winds-west', 'characters-7', 'dots-9', 'characters-7', 'bamboo-1', 'dragons-red', 'characters-1', 'characters-5']



# # 自定义的排序函数
# def tile_sort_key(tile):
#     # 分离类别和数字/颜色

#     # 定义类别的排序优先级
#     category_order = {
#         'bamboo': 0,
#         'characters': 1,
#         'dots': 2,
#         'dragons': 3,
#         'winds': 4
#     }
#     parts = tile.split('-')
#     category = parts[0]
#     value = parts[1]
    
#     # 数字化数字部分，如果不是数字（如winds-east），则赋值为一个高值以确保其排序在后
#     if value.isdigit():
#         value = int(value)
#     else:
#         value = float('inf')
    
#     # 返回 (类别优先级, 数字/颜色)
#     return (category_order[category], value)

# # 按照自定义排序规则排序
# sorted_tiles = sorted(tiles, key=tile_sort_key)

# print(sorted_tiles)

# from openai import OpenAI
# client = OpenAI(
#     base_url='http://10.129.143.25:11434/v1/',
#     api_key='ollama', # required but ignored
# )
# chat_completion=client.chat.completions.create(
# messages=[{'role':'user','content':'你好，请介绍下你自己'}],
# model='llama3',
# )

# print(chat_completion.choices[0].message.content)


import ray
import time,datetime

# Start Ray.
ray.init()

import numpy as np

# 定义两个远程函数。
# 这些函数的调用创建了远程执行的任务

@ray.remote
def create_matrix(size):
    return np.random.normal(size=size)
@ray.remote
def multiply_matrices(x, y):
    return np.dot(x, y)

result_ids = []
for i in range(400):
    # 开始两个并行的任务，这些会立即返回futures并在后台执行
    x_id = create_matrix.remote([1000, 1000])
    print(datetime.datetime.now())
    y_id = create_matrix.remote([1000, 1000])
    print(datetime.datetime.now())
    # 开始第三个任务，但这并不会被提前计划，直到前两个任务都完成了.
    result_ids.append(multiply_matrices.remote(x_id, y_id))
    print(datetime.datetime.now())
# 获取结果。这个结果直到第三个任务完成才能得到。只有get创建以后所有的任务才开始创建执行。
z_id = ray.get(result_ids)
print(z_id)

