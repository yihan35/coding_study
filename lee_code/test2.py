class ListNode:
    """定义单链表节点类"""
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def get_intersection_node(headA, headB):
    """
    双指针法寻找两个链表的相交节点
    :param headA: 链表A的头节点
    :param headB: 链表B的头节点
    :return: 相交节点（无则返回None）
    """
    pA, pB = headA, headB
    # 双指针遍历，直到相遇或同时走到末尾
    while pA != pB:
        # A指针走到末尾则切换到B链表头
        pA = pA.next if pA else headB
        # B指针走到末尾则切换到A链表头
        pB = pB.next if pB else headA
    return pA


# ---------------------- 输入处理 ----------------------
# 读取链表A的长度和skipA
n, skipA = map(int, input().split())
# 读取链表A的节点值
listA = list(map(int, input().split()))
# 读取链表B的长度和skipB
m, skipB = map(int, input().split())
# 读取链表B的节点值
listB = list(map(int, input().split()))

# ---------------------- 构造链表A ----------------------
headA = None
if n > 0:
    headA = ListNode(listA[0])
    curA = headA
    for val in listA[1:]:
        curA.next = ListNode(val)
        curA = curA.next

# 找到链表A的相交节点（若skipA < n）
intersect_node = None
if skipA < n and headA:
    intersect_node = headA
    for _ in range(skipA):
        intersect_node = intersect_node.next

# ---------------------- 构造链表B ----------------------
headB = None
if m > 0:
    headB = ListNode(listB[0])
    curB = headB
    # 先构造链表B的前skipB个节点
    for i in range(1, skipB):
        curB.next = ListNode(listB[i])
        curB = curB.next
    # 若存在相交节点，让链表B的第skipB个节点后接相交节点
    if skipB < m and intersect_node:
        curB.next = intersect_node
    else:
        # 无相交，继续构造链表B剩余节点
        for val in listB[skipB:]:
            curB.next = ListNode(val)
            curB = curB.next

# ---------------------- 求解并输出结果 ----------------------
result_node = get_intersection_node(headA, headB)
# 输出相交节点值，无则输出0
print(result_node.val if result_node else 0)