'''
## 题目描述
给你两个单链表的头节点 `headA` 和 `headB`，找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 `None`。
要求：
1. 使用ACM本地可运行格式，包含链表定义、数组转链表工具函数
2. 自行构造多组测试输入，程序运行后直接打印答案节点的值，无交点输出`-1`
3. 最优解法：双指针，时间O(m+n)，空间O(1)

### 输入说明
输入三组数组：
- 第一组：链表A不相交部分数组
- 第二组：链表B不相交部分数组
- 第三组：两个链表共用的相交尾部数组
若相交数组为空代表无交点

### 输出说明
输出相交节点的值，无交点输出 `-1`
'''
class ListNode:
    def __init__(self,val=0,next=None):
        self.val = val
        self.next  = next
def getIntersectionNode(headA,headB):
    p1,p2 = headA ,headB
    while p1!=p2:
        p1 = p1.next if p1 else headB
        p2 = p2.next if p2 else headA
    return p1
# 由于输入的是列表，需要将数据转换为链表的格式
def list_to_linked(arr):
    if not arr:
        return None
    head = ListNode(arr[0])
    current = head
    for val in arr[1:]:
        current.next = ListNode(val)
        current = current.next
    return head

# 测试
arrA = [4,1]
arrB = [5,0,1]
arr_public = [8,4,5]
public_head = list_to_linked(arr_public)
# 拼接链表A
headA = list_to_linked(arrA)
if headA:
    cur = headA
    while cur.next:
        cur = cur.next
    cur.next = public_head

# 拼接链表B
headB = list_to_linked(arrB)
if headB:
    cur = headB
    while cur.next:
        cur = cur.next
    cur.next = public_head

node= getIntersectionNode(headA,headB)
print(node.val if node else -1)
        