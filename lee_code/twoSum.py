from typing import List


def twoSum(nums:List[int],target:int)-> List[int]:
    n = len(nums)
    map = {}
    for index, value in enumerate(nums):
        if target - value in map:
            return [map[target- value],index]
        map[value] = index
    return []

print(twoSum([2,5,6,7],7))