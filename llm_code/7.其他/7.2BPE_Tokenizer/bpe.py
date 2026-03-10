text = 'This is CookLLM, and keeping study with me!'
# 获取所有唯一字符
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(chars)
print(','.join(chars))# Python 的 join() 方法，用于将序列拼接成字符串
# 输出:  !,CLMTadeghikmnopstuwy
print(vocab_size)
# 输出: 23

# 创建字符 <-> 整数的映射
stoi = {ch: i for i, ch in enumerate(chars)}  # string to integer
itos = {i: ch for i, ch in enumerate(chars)}  # integer to string

# 编码函数：字符串 -> 整数列表
encode = lambda s: [stoi[c] for c in s]

# 解码函数：整数列表 -> 字符串
decode = lambda l: ''.join([itos[i] for i in l])

# 测试
print(encode("this is"))
# 输出: [19, 11, 12, 18, 0, 12, 18]

print(decode(encode("this is")))
# 输出: this is


import torch

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
# 输出: torch.Size([43]) torch.int64

print(data[:20])
# 前 20 个 tokens