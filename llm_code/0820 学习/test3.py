# -*- coding: utf-8 -*-
from typing import List

class Solution:
    def FindContinuousSequence(self, tsum: int) -> List[List[int]]:
        """
        使用滑动窗口（双指针法）找出所有和为 tsum 的连续正数序列。

        :param tsum: 目标和
        :return: 一个包含所有满足条件的序列的列表
        """
        # 根据题意，序列至少有两个数，所以 tsum 至少要大于 2
        if tsum <= 2:
            return []

        # 初始化左右指针（窗口边界）和当前窗口内数字的和
        left, right = 1, 2
        current_sum = left + right
        
        # 结果列表
        result = []
        
        # 循环的终止条件：left 不可能大于 tsum 的一半
        while left <= tsum // 2:
            if current_sum < tsum:
                # 和太小，扩大窗口：右指针向右移动
                right += 1
                current_sum += right
            elif current_sum > tsum:
                # 和太大，缩小窗口：左指针向右移动
                current_sum -= left
                left += 1
            else: # current_sum == tsum
                # 找到了一个解，记录下来
                sequence = list(range(left, right + 1))
                result.append(sequence)
                
                # 继续寻找下一个可能的解，缩小窗口
                current_sum -= left
                left += 1
                
        return result

# --- 测试代码 ---
# 创建 Solution 类的实例
solver = Solution()

# 示例 1
input_1 = 9
output_1 = solver.FindContinuousSequence(input_1)
print(f"输入: {input_1}")
print(f"输出: {output_1}")
print("-" * 20)

# 示例 2
input_2 = 0
output_2 = solver.FindContinuousSequence(input_2)
print(f"输入: {input_2}")
print(f"输出: {output_2}")
print("-" * 20)

# 题目描述中的例子
input_3 = 100
output_3 = solver.FindContinuousSequence(input_3)
print(f"输入: {input_3}")
print(f"输出: {output_3}")
print("-" * 20)

input_4 = 15
output_4 = solver.FindContinuousSequence(input_4)
print(f"输入: {input_4}")
print(f"输出: {output_4}")
print("-" * 20)