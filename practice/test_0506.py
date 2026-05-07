# m * n 二维数组，旋转 90 度，打印输出
def rotate(nums):
    m = len(nums)
    n = len(nums[0])
    # res = [[0]*m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            # res[j][m-1-i] = nums[i][j]
            print(nums[m-1-j][i],end='')
    # return res

nums =[[1,2,3],
       [4,5,6]]
print(rotate(nums))