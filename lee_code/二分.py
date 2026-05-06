data = [1,2,2,3,3,3,4,5,5,6]
a,b = 3,5
left = 0
right = len(data) -1
# 第一个大于等于a 的下标
first_a = len(data)
while left <= right:
    mid = (left + right) // 2
    if data[mid] >= a:
        first_a = mid
        right = mid -1
    else:
        left = mid +1
# 最后一个小于等于b的下标
left = 0
right = len(data) -1
last_b = len(data)
while left <= right:
    mid = (left + right) //2
    if data[mid] <= b:
        last_b = mid
        left = mid +1
    else:
        right = mid - 1
print(first_a)
print(last_b)
count_a_b = last_b - first_a +1
print(count_a_b)