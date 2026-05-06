# 有一个有序数组，从某一个位置断开，将前面部分移动到后面，重新拼接，生成一个某一个位置有断层的数组b，数组b 给定，给定一个数x，希望写一个函数，判断x 是否在b里面。
nums = [5, 7, 8, 9, 1, 3, 4]
target = 0
def search(nums,target):
    # 左右指针
    left,right = 0,len(nums)-1
    while left <= right:
        mid = left + (right-left) //2
        if nums[mid] == target:
            return mid
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid -1
            else:
                left = mid+1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid +1
            else:
                right = mid -1
    return  -1
print(search(nums,target))