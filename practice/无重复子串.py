def lengthOfLongestSubstring(s: str) -> int:
    left = 0 
    right = 0
    max_len = 0
    dic = {}
    for right in range(len(s)):
        print("第",right,"次循环")
        current = s[right]
        print("current:",current)
        # print(dic[current])
        if current in dic and dic[current]>=left:
            left = dic[current] +1
        dic[current] = right
        print("dic=",dic)
        print(max_len)
        max_len = max(max_len,right - left + 1)
    return max_len

s = "abcabcbb"
print(lengthOfLongestSubstring(s))