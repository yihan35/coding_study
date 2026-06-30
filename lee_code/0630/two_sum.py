def two_sum(nums,target):
    dic = {}
    for i,element in enumerate(nums):
        if target - element in dic:
            return([dic[target - element],i])
        dic[element] = i
    return []
print(two_sum([2,7,11,15],9))
# 两数之和，用哈希的思路
# 时间复杂度 o(n),空间复杂度 o(n)
# 力扣第一题