from typing import List


def twoSum(nums: List[int], target: int) -> List[int]:
    n = len(nums)
    map = {}
    for index,res in enumerate(nums):
        if target - res in map:
            return([map[target-res],index])
        map[res] = index
    return []

print(twoSum([2,7,11,15],9))
