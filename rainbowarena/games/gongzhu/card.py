"""
在gongzhu中，共52张牌
四个人玩，每个人13张
52张牌
① 共4种花色
② 按照大小分别为：A＞K＞Q＞J＞10＞9＞8＞7＞6＞5＞4＞3＞2
具体牌的计分：
① 猪牌：黑桃Q称为猪牌，分值为-100分，亮牌后，分值加倍为-200分。
② 羊牌：方块J称为羊牌，分值为100分，亮牌后，分值加倍为200分。
③ 变压器：梅花10称为变压器，得到该牌的玩家，最后的得分无论正负都将加倍计算。如果变压器亮过牌，最后的得分就要按四倍计算。但是，如果只是得到变压器，其他什么分牌都没有得到，则变压器按50分算，如果亮过牌，按100分计算。
④ 血：所有的红桃牌称为血。其中2、3、4没有分；5、6、7、8、9、10分值为-10；J分值为-20；Q分值为-30；K分值为-40；A分值为-50。
⑤ 满红：如果一方在一轮游戏结束时将全部红桃收到自己手中，称为满红。一副牌的情况下：此时得到200分。如果红桃A亮过牌，则得到400分。
⑥ 满贯：如果一方在一轮游戏结束时将全部含有分值的牌都收齐了，称为满贯。一副牌的情况下，得800分。如果有亮牌，则相应加倍计算。
"""
