"""
题目：两数之和
给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。
示例 1：
输入：nums = [2,7,11,15], target = 9
输出：[0,1]
解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。

"""
def twoSum(nums,target):
    n = len(nums)
    dic = {}
    if not nums:
        return []
    for i,num in enumerate(nums):
        if target - num in dic:
            return [dic[target-num],i]
        dic[num]  = i  
# print(twoSum(nums=[2,7,11,15],target=9))
# print(twoSum(nums=[3,2,4],target=6))

'''
题目：字母异位词分组
给你一个字符串数组，请你将 字母异位词 组合在一起。可以按任意顺序返回结果列表。
示例 1:
输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
输出: [["bat"],["nat","tan"],["ate","eat","tea"]]
'''
def solution(strs):
    dic = {}
    for word in strs:
        # 1. 字符串排序后为列表 'eat' --> ['a','e','t']
        list_str = sorted(word)
        # 2. 构造键 ['a','e','t'] --> 'aet'
        key = ''.join(list_str)
        # 3. 组合异位词
        if key not in dic:
            dic[key] = []
        dic[key].append(word)
    # 4.dic.values()返回dict_values，最终转为 list 形式 
    return list(dic.values())
# print(solution(strs=["eat", "tea", "tan", "ate", "nat", "bat"]))

'''
题目：最长连续序列
给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。
请你设计并实现时间复杂度为 O(n) 的算法解决此问题。
示例 1：
输入：nums = [100,4,200,1,3,2]
输出：4
解释：最长数字连续序列是 [1, 2, 3, 4]。它的长度为 4。
'''
def longestSeq(nums):
    # 下列 in nums 是在列表中查找，时间复杂度为 O(n)，无法满足题目要求。
    # 应该先转换成集合，使查找复杂度平均为 O(1)。
    nums_set = set(nums)
    res = 1
    for num in nums_set:
        # 首先确定起点
        if num-1 not in nums_set:
            count = 1
            while num+1 in nums_set: 
                num += 1
                count +=1 
        res = max(res,count)
    return res
# print(longestSeq(nums=[100,4,200,1,3,2]))

'''
题目：移动零
给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
请注意 ，必须在不复制数组的情况下原地对数组进行操作。
示例 1:
输入: nums = [0,1,0,3,12]
输出: [1,3,12,0,0]
'''
# 思路：双指针，right 寻找非零元素，left 指向下一个非零元素应该放置的位置
def moveZero(nums):
    n = len(nums)
    left = 0
    for right in range(n):
        # right 每找到一个非零元素，都交换二者，left 在左侧，每交换一次向右移动一位
        # right 没找到，则不交换
        if nums[right]:
            nums[left],nums[right] = nums[right],nums[left]
            left += 1
    return nums
# print(moveZero(nums=[0,1,0,3,12]))

'''
题目：盛最多水的容器
给定一个长度为 n 的整数数组 height 。有 n 条垂线，第 i 条线的两个端点是 (i, 0) 和 (i, height[i]) 。
找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
返回容器可以储存的最大水量。
输入：[1,8,6,2,5,4,8,3,7]
输出：49 
'''
def maxWater(nums):
    left ,right = 0,len(nums)-1
    res = (right-left) * min(nums[left],nums[right])
    while left < right:
        if nums[left] > nums[right]:
            right -= 1
        else:
            left += 1
        curr = (right-left) * min(nums[left],nums[right])
        res = max(curr,res)
    return res
# print(maxWater(nums=[1,8,6,2,5,4,8,3,7]))

'''
题目：三数之和
给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ，同时还满足 nums[i] + nums[j] + nums[k] == 0 。请你返回所有和为 0 且不重复的三元组。
注意：答案中不可以包含重复的三元组。
输入：nums = [-1,0,1,2,-1,-4]
输出：[[-1,-1,2],[-1,0,1]]
'''
def threeSum(nums):
    nums_new = sorted(nums)
    res = []
    for i in range(0,len(nums_new)-2):
        if i>0 and nums_new[i] == nums_new[i-1]:
            continue
        left = i+1
        right = len(nums_new)-1
        while left < right :
            if nums_new[left] + nums_new[right] == -nums_new[i]:
                res.append([nums_new[i],nums_new[left],nums_new[right]])
                while left < right and nums_new[left+1] == nums_new[left]:
                    left += 1
                left += 1
                while left < right and nums_new[right-1] == nums_new[right]:
                    right -= 1
                right -= 1
            elif nums_new[left] + nums_new[right] > -nums_new[i]:
                right -= 1
            else: 
                left += 1
    return res
# print(threeSum(nums=[-1,0,1,2,-1,-4]))

'''
题目：接雨水
给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
'''
# 计算每个位置 min( 前缀最大(包含自己），后缀最大(包含自己) )- 当前值
# 遇到的问题：leftmax[i] = max(nums[i],leftmax[i-1])要用 i-1 的值
def rainMax(nums):
    if not nums:
        return 0
    ans = 0
    n = len(nums)
    leftmax = [0] * n
    rightmax = [0] * n
    leftmax[0] = nums[0]
    rightmax[-1] = nums[-1]
    for i in range(1,n):
        leftmax[i] = max(nums[i],leftmax[i-1])
    for i in range(n-2,-1,-1):
        rightmax[i] = max(nums[i],rightmax[i+1])
        
    for i in range(n):
        ans += min(leftmax[i],rightmax[i]) - nums[i]
    return ans
print(rainMax(nums=[0,1,0,2,1,0,1,3,2,1,2,1]))