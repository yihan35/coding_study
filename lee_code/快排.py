import sys
input = sys.stdin.readline

def quick_sort(nums, left, right):
    # 递归终止条件：区间只有一个或零个元素
    if left >= right:
        return

    # 进行分区，获取基准值最终所在位置
    pivot_idx = partition(nums, left, right)

    # 递归排序基准左边的部分
    quick_sort(nums, left, pivot_idx - 1)

    # 递归排序基准右边的部分
    quick_sort(nums, pivot_idx + 1, right)

def partition(nums, left, right):
    # 随机选取基准值下标，避免有序数组导致最坏 O(n²)
    import random
    rand_idx = random.randint(left, right)

    # 将随机基准换到最右边，方便后续处理
    nums[rand_idx], nums[right] = nums[right], nums[rand_idx]

    # 取最右元素作为基准值
    pivot = nums[right]

    # i 指向小于基准区域的最右边界，初始在 left 左侧
    i = left - 1

    # 遍历 left 到 right-1，将小于 pivot 的元素移到左边
    for j in range(left, right):

        # 当前元素小于等于基准值，扩展左区域
        if nums[j] <= pivot:
            i += 1
            # 将当前元素换到左区域末尾
            nums[i], nums[j] = nums[j], nums[i]

    # 将基准值放到最终正确位置（左区域右边一位）
    nums[i + 1], nums[right] = nums[right], nums[i + 1]

    # 返回基准值的最终下标
    return i + 1

# 读取输入
n = int(input().strip())
nums = list(map(int, input().strip().split()))
print(nums)
# 调用快速排序
quick_sort(nums, 0, n - 1)

# 输出结果
print(*nums)