# -*- coding: utf-8 -*-
from typing import List

class Solution:
    def FindContinuousSequence(self, tsum: int) -> List[List[int]]:
        """
        使用优化的滑动窗口（双指针法）找出所有和为 tsum 的连续正数序列。

        优化点：
        1. 提前终止条件优化
        2. 减少不必要的计算
        3. 更精确的边界条件

        :param tsum: 目标和
        :return: 一个包含所有满足条件的序列的列表
        """
        # 边界条件：序列至少有两个数，所以 tsum 至少要大于等于 3 (1+2)
        if tsum < 3:
            return []

        result = []
        left, right = 1, 2
        current_sum = 3  # 1 + 2 = 3
        
        # 更精确的终止条件：当 left > (tsum-1)//2 时就不可能找到解了
        # 因为最小的两个连续数是 left 和 left+1，它们的和是 2*left+1
        # 如果 2*left+1 > tsum，即 left > (tsum-1)//2，就不可能有解
        while left < (tsum + 1) // 2:  # 等价于 left <= (tsum-1)//2
            if current_sum < tsum:
                # 和太小，扩大窗口
                right += 1
                current_sum += right
            elif current_sum > tsum:
                # 和太大，缩小窗口
                current_sum -= left
                left += 1
                # 确保 right >= left + 1，保证序列至少有两个数
                if right < left + 1:
                    right = left + 1
                    current_sum = left + right
            else:  # current_sum == tsum
                # 找到解，直接用范围生成列表（比 list(range()) 稍快）
                result.append([i for i in range(left, right + 1)])
                
                # 继续寻找下一个解，缩小窗口
                current_sum -= left
                left += 1
                
        return result

    def FindContinuousSequence_v2(self, tsum: int) -> List[List[int]]:
        """
        另一种优化方法：基于数学公式的枚举法
        
        对于连续序列 [start, start+1, ..., start+length-1]
        其和为：length * start + length * (length-1) // 2 = tsum
        即：start = (tsum - length * (length-1) // 2) // length
        
        这种方法在某些情况下可能更快，特别是当 tsum 很大时。
        """
        if tsum < 3:
            return []
            
        result = []
        
        # 枚举序列长度，从2开始到可能的最大长度
        # 最大长度的估算：当序列从1开始时，length*(length+1)//2 <= tsum
        max_length = int((-1 + (1 + 8 * tsum) ** 0.5) // 2) + 1
        
        for length in range(2, max_length + 1):
            # 计算起始数字
            # length * start + length * (length-1) // 2 = tsum
            # start = (tsum - length * (length-1) // 2) / length
            numerator = tsum - length * (length - 1) // 2
            
            if numerator <= 0:
                break
                
            if numerator % length == 0:  # start 必须是正整数
                start = numerator // length
                if start > 0:  # 确保起始数字是正数
                    sequence = [start + i for i in range(length)]
                    result.append(sequence)
        
        # 按起始数字排序（虽然通常已经是有序的）
        result.sort()
        return result


# --- 性能测试和比较 ---
import time

def test_performance():
    solver = Solution()
    
    test_cases = [9, 15, 100, 1000, 5000]
    
    for tsum in test_cases:
        print(f"\n测试 tsum = {tsum}:")
        
        # 测试原始滑动窗口方法
        start_time = time.perf_counter()
        result1 = solver.FindContinuousSequence(tsum)
        time1 = time.perf_counter() - start_time
        
        # 测试数学公式方法
        start_time = time.perf_counter()
        result2 = solver.FindContinuousSequence_v2(tsum)
        time2 = time.perf_counter() - start_time
        
        print(f"滑动窗口方法: {time1:.6f}s, 结果数量: {len(result1)}")
        print(f"数学公式方法: {time2:.6f}s, 结果数量: {len(result2)}")
        print(f"结果一致性: {result1 == result2}")
        
        if len(result1) <= 10:  # 只打印较短的结果
            print(f"结果: {result1}")

# --- 基本功能测试 ---
def basic_test():
    solver = Solution()
    
    test_cases = [
        (9, "示例1"),
        (0, "边界测试1"),
        (15, "示例2"),
        (100, "示例3")
    ]
    
    for tsum, desc in test_cases:
        result1 = solver.FindContinuousSequence(tsum)
        result2 = solver.FindContinuousSequence_v2(tsum)
        
        print(f"{desc} - 输入: {tsum}")
        print(f"滑动窗口结果: {result1}")
        print(f"数学公式结果: {result2}")
        print(f"结果一致: {result1 == result2}")
        print("-" * 40)

if __name__ == "__main__":
    print("=== 基本功能测试 ===")
    basic_test()
    
    print("\n=== 性能对比测试 ===")
    test_performance()