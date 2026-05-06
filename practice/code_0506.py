# 输入数组，输出int 峰值
def find(nums):
    l,r = 0,len(nums)-1
    while l < r:
        mid = (l+r) //2
        if nums[mid] < nums[mid+1]:
            l = mid +1
        else:
            r = mid
    return l

nums = [1,2,3,1]
# print(nums[find(nums)])

# 快排
def quick_sort(nums,left,right):
    if left >= right:
        return
    pivot = nums[left]
    l,r = left,right
    while l < r:
        while l<r and nums[r] >= pivot:
            r -= 1
        nums[l] = nums[r]
        while l<r and nums[l] <= pivot:
            l +=1
        nums[r] = nums[l]
    nums[l] = pivot

    quick_sort(nums,left,l-1)
    quick_sort(nums,l+1,right)

arr = [3,1,4,1,5,9,2,6]
quick_sort(arr,0,len(arr)-1)
# print(arr)

# 基于快排写第 k 大个数
def find_kth(nums,k):
    def quick_select(left,right):
        pivot = nums[left]
        l,r = left,right
        while l < r:
            while l<r and nums[r] <= pivot:
                r -=1
            nums[l] = nums[r]
            while l<r and nums[l] >= pivot:
                l +=1
            nums[r] = nums[l]
        nums[l] = pivot
        if l == k-1:
            return nums[l]
        elif l>k-1:
            return quick_select(left,l-1)
        else:
            return quick_select(l+1,right)
    return quick_select(0,len(nums)-1)
nums = [3,2,1,5,6,4]
k = 2
# print(find_kth(nums,k))


# 编辑距离
def minDistance(word1,word2):
    m,n = len(word1),len(word2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1,m+1):
        for j in range(1,n+1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])+1
    return dp[m][n]
word1 = 'horse',
word2 = 'ros'
print(minDistance(word1,word2))