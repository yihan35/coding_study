text = 'This is CookLLM, and keeping study with me!'
# 获取所有唯一字符
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
# 输出:  !,CLMTadeghikmnopstuwy
print(vocab_size)
# 输出: 23