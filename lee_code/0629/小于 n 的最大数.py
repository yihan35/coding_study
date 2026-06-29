'''
题目：小于 n 的最大数
题目描述：给定一个数 n（如 23121），给定一组数字 A（如 {2,4,9}），求由 A 中元素组成的、小于 n 的最大数（例如小于 23121 的最大数为 22999）。
23121, [2, 4, 9]
'''

def max_less_than_n(n, A):
    digits = sorted(set(A))# 去重并排序，这一行排序的复杂度是 O(k log k)

    if n <= 0 or not digits:
        return None

    s = str(n)# n换成字符串，方便按位处理
    m = len(s)# 字符串的长度

    max_digit = digits[-1]# 拿到 A 列表中最大的数，后面一旦确定结果已经小于 n，剩下位全部填最大数字。
    nonzero = [d for d in digits if d != 0]# 取出 A 列表中所有非零的数字，因为多位数第一位不能是0

    def biggest_shorter(max_len):#定义函数，当无法构造和n一样长的数字时，构造一个少一位的最大数字
        if max_len <= 0:
            return None
        if nonzero:
            first = max(nonzero)#第一位取最大的非零数字
            return int(str(first) + str(max_digit) * (max_len - 1))
    ans = []

    for i, ch in enumerate(s):#从左到右遍历n的每一位
        cur = int(ch)

        candidates = digits

        if i == 0 and m > 1:#如果是n的第一位，而且 n 是多位数，那么第一位不能选 0
            candidates = nonzero

        if cur in candidates:#如果当前位可以和 n 的当前位保持一样，就先保持一致
            ans.append(cur)
            continue

        smaller = [d for d in candidates if d < cur]#如果候选后中没有当前遍历到的数字，也就是无法保持一样，就找比当前位小的数字

        if smaller:
            d = max(smaller)#取里面最大的那个
            ans.append(d)#把这个数字放到当前位
            ans += [max_digit] * (m - i - 1)# 后面所有的位都填 A 里面的最大数字
            return int(''.join(map(str, ans)))#把列表转为整数返回
        # 如果当前位既不能相等，也不能变小，就说明这条路走不通，需要回退
        break

    for i in range(len(ans) - 1, -1, -1):
        candidates = digits

        if i == 0 and m > 1:
            candidates = nonzero

        smaller = [d for d in candidates if d < ans[i]]

        if smaller:
            d = max(smaller)
            res = ans[:i] + [d] + [max_digit] * (m - i - 1)
            return int(''.join(map(str, res)))
    # 如果相同位完全做不到，就构造少一位的最大数
    return biggest_shorter(m - 1)


print(max_less_than_n(23121, [2, 4, 9]))  # 22999
print(max_less_than_n(500, [1, 3, 7]))    # 377
print(max_less_than_n(222, [2]))          # 22
print(max_less_than_n(1000, [0, 1, 9]))   # 999