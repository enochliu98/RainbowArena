import torch
import copy


def load_model(model_path):
    from rainbowarena.agents import DQNAgent
    agent = DQNAgent.from_checkpoint(checkpoint=torch.load(model_path))

    return agent

models = [
          load_model('SP/ticket2ride_psro/2.pt'),
          load_model('SP/ticket2ride_psro/3.pt'),
          load_model('SP/ticket2ride_psro/4.pt'),
          ]  # Model1, Model2, ..., ModelN 是已经加载并放在列表中的模型
weights = [0.425, 0.058, 0.517] # w1, w2, ..., wN 是每个模型对应的权重

# 假设你已经有了n个模型和一个与它们对应的权重列表
# weights 应该是一个与 models 相同长度的列表，包含了每个模型的权重
# models = [load_model('SP/gongzhu_psro_new/7.pt'),
#           load_model('SP/gongzhu_psro_new/8.pt'),
#           load_model('SP/gongzhu_psro_new/9.pt'),
#           ]  # Model1, Model2, ..., ModelN 是已经加载并放在列表中的模型
# weights = [0.114, 0.442, 0.444]  # w1, w2, ..., wN 是每个模型对应的权重

# 确保权重总和为1
weights_sum = sum(weights)
weights = [w / weights_sum for w in weights]

# 创建一个新的空模型，其结构应该与其它模型相同
merged_model = copy.deepcopy(models[0])

# 遍历新模型的参数
for param in merged_model.q_estimator.qnet.parameters():
    param.data = torch.zeros_like(param.data)

# 计算所有模型参数的加权平均值
for i, model in enumerate(models):
    weight = weights[i]
    for merged_param, param in zip(merged_model.q_estimator.qnet.parameters(), model.q_estimator.qnet.parameters()):
        merged_param.data += weight * param.data

# 现在merged_model包含了所有模型权重的加权平均值
merged_model.save_checkpoint(path='SP/ticket2ride_psro/', filename='sum.pt')