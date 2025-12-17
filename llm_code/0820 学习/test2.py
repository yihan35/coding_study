# -*- coding: utf-8 -*-
from typing import List
import math

class Solution:
    """
    求解最大子数组和问题
    """
    def FindGreatestSumOfSubArray(self, array: List[int]) -> int:
        """
        使用动态规划（卡登算法）找到最大子数组和。

        :param array: 输入的整数数组。
        :return: 子数组的最大和。
        """
        # 如果数组为空，根据题目约束其实不会发生，但作为代码健壮性可以加上
        if not array:
            return 0
            
        # global_max 用于存储全局最大和，初始化为数组的第一个元素
        # current_max 用于存储以当前元素结尾的子数组的最大和
        global_max = array[0]
        current_max = array[0]
        
        # 从数组的第二个元素开始遍历
        for i in range(1, len(array)):
            # 核心逻辑：对于当前元素，我们有两个选择
            # 1. 将它加入之前的子数组： current_max + array[i]
            # 2. 从它自己开始一个新的子数组： array[i]
            # 我们选择两者中较大的一个作为以当前元素结尾的子数组的最大和
            current_max = max(array[i], current_max + array[i])
            
            # 更新全局最大和：比较当前的全局最大值和刚计算出的以当前元素结尾的子数组最大和
            global_max = max(global_max, current_max)
            
        return global_max

# --- 测试代码 ---
# 创建 Solution 类的实例
solver = Solution()

# 示例 1
input_1 = [1, -2, 3, 10, -4, 7, 2, -5]
output_1 = solver.FindGreatestSumOfSubArray(input_1)
print(f"输入: {input_1}")
print(f"输出: {output_1}")
print("说明: 子数组 [3, 10, -4, 7, 2] 可以求得最大和为 18")
print("-" * 20)

# 示例 2
input_2 = [2]
output_2 = solver.FindGreatestSumOfSubArray(input_2)
print(f"输入: {input_2}")
print(f"输出: {output_2}")
print("-" * 20)

# 示例 3
input_3 = [-10]
output_3 = solver.FindGreatestSumOfSubArray(input_3)
print(f"输入: {input_3}")
print(f"输出: {output_3}")
print("-" * 20)

# 其他测试用例
input_4 = [-2, -8, -1, -5, -9]
output_4 = solver.FindGreatestSumOfSubArray(input_4)
print(f"输入: {input_4}")
print(f"输出: {output_4}") # 预期输出 -1
print("-" * 20)