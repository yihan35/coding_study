# ====================== 核心函数：回溯试错法 ======================
# 功能：试着把数字一个一个放进桶里，不行就撤回，判断能不能分成功
# 参数解释（一个一个讲清楚）：
# nums    : 原始的整数数组（比如 [3,2,4,3,6]）
# index   : 当前正在处理第几个数字（从 0 开始，放完一个就 +1）
# bucket  : 列表，代表 m 个桶，里面存每个桶当前的和
# target  : 每个桶最终必须达到的目标和（固定的数）
def backtrack(nums, index, bucket, target):
    
    # 一、递归终止条件：所有数字都放完了 → 分成功了！返回 True
    if index == len(nums):
        return True
    
    # 二、取出当前要放的数字
    current_num = nums[index]

    # 三、遍历所有的桶，尝试把数字放进每一个“能放下”的桶里
    for i in range(len(bucket)):
        # 如果这个桶放了当前数字会超过目标 → 不能放，跳过
        if bucket[i] + current_num > target:
            continue
        
        # 1. 尝试把数字放进第 i 个桶
        bucket[i] += current_num

        # 2. 递归：放下一个数字（index + 1）
        # 如果下一个数字也能放完 → 直接返回成功
        if backtrack(nums, index + 1, bucket, target):
            return True

        # 3. 回溯关键点：
        # 走到这里 = 刚才放错了！把数字拿出来，恢复原状
        bucket[i] -= current_num

    # 四、所有桶都试过了，都放不进去 → 失败，返回 False
    return False


# ====================== 主函数：求最大 m ======================
# 功能：找到数组最多能分成几份，每份和相等
# 参数：nums = 输入的整数数组
# 返回：最大的 m
def max_m(nums):
    # 1. 计算数组所有数字的总和
    total_sum = sum(nums)
    
    # 2. 数组长度 n
    n = len(nums)

    # 3. 从【最大可能的份数】开始试：最多能分 n 份（每个数字1份）
    for m in range(n, 0, -1):
        # 条件1：总和必须能被 m 整除，否则不可能平分
        if total_sum % m != 0:
            continue
        
        # 每份的目标和
        target = total_sum // m
        
        # 创建 m 个桶，一开始都是空的（和为0）
        bucket = [0] * m

        # 条件2：真的能把数组分成 m 份，每份和 = target
        if backtrack(nums, 0, bucket, target):
            # 能成功 → 这个 m 就是最大值，直接返回
            return m

    # 保底：最少也能分成 1 份
    return 1


# ====================== 测试 ======================
if __name__ == "__main__":
    arr = [3, 2, 4, 3, 6]
    print("最大可分成的份数：", max_m(arr))