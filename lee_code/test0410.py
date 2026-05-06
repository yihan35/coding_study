# 给你一个链表数组，每个链表都已经按升序排列。

# 请你将所有链表合并到一个升序链表中，返回合并后的链表。

 

# 示例 1：

# 输入：lists = [[1,4,5],[1,3,4],[2,6]]
# 输出：[1,1,2,3,4,4,5,6]
# 解释：链表数组如下：
# [
#   1->4->5,
#   1->3->4,
#   2->6
# ]
# 将它们合并到一个有序链表中得到。
# 1->1->2->3->4->4->5->6
# 示例 2：

# 输入：lists = []
# 输出：[]
# 示例 3：

# 输入：lists = [[]]
# 输出：[]
# Definition for singly-linked list.
import heapq
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
def merge_k(lists):
    dummy = ListNode(0)
    cur = dummy
    heap = []
    for i ,node in enumerate(lists):# 最小堆维护当前最小节点
        if node:
            heapq.heappush(heap,(node.val,i,node))
    while heap:
        val,i,node = heapq.heappop(heap)
        cur.next = node
        cur = cur.next
        if node.next:
            heapq.heappush(heap,(node.next.val,i,node.next))
    return dummy.next
def buid(arr):
    dummy = ListNode(0)
    cur = dummy
    for x in arr:
        cur.next = ListNode(x)
        cur = cur.next
    return dummy.next
def to_list(node):
    res  = []
    while node:
        res.append(node.val)
        node = node.next
    return res
list1 = [buid([1,4,5]),buid([1,3,4]),buid([2,6])]
print(to_list(merge_k(list1)))
