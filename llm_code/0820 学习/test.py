# -*- coding: utf-8 -*-

class Solution:
    """
    判断一个字符串是否是回文字符串
    """
    def isPalindrome(self, s: str) -> bool:
        """
        使用双指针法判断字符串是否为回文。

        :param s: 输入的字符串，仅由小写字母组成。
        :return: 如果是回文则返回 True，否则返回 False。
        """
        # 初始化左右指针
        left, right = 0, len(s) - 1

        # 当左指针在右指针左边时，循环继续
        while left < right:
            # 比较左右指针指向的字符
            if s[left] != s[right]:
                # 如果不相等，则不是回文
                return False
            # 如果相等，则将指针向中间移动
            left += 1
            right -= 1

        # 如果循环正常结束，说明所有对称位置的字符都相等，是回文
        return True

# --- 测试代码 ---
# 创建 Solution 类的实例
solver = Solution()

# 示例 1
input_1 = "absba"
output_1 = solver.isPalindrome(input_1)
print(f"输入: \"{input_1}\"")
print(f"输出: {str(output_1).lower()}") # 输出 true/false 以匹配示例
print("-" * 20)

# 示例 2
input_2 = "ranko"
output_2 = solver.isPalindrome(input_2)
print(f"输入: \"{input_2}\"")
print(f"输出: {str(output_2).lower()}")
print("-" * 20)

# 示例 3
input_3 = "yamatomaya"
output_3 = solver.isPalindrome(input_3)
print(f"输入: \"{input_3}\"")
print(f"输出: {str(output_3).lower()}")
print("-" * 20)

# 其他测试用例
input_4 = "level"
output_4 = solver.isPalindrome(input_4)
print(f"输入: \"{input_4}\"")
print(f"输出: {str(output_4).lower()}")
print("-" * 20)

input_5 = "a"
output_5 = solver.isPalindrome(input_5)
print(f"输入: \"{input_5}\"")
print(f"输出: {str(output_5).lower()}")
print("-" * 20)