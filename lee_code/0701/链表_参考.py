#  # 面试题：相交链表（ACM完整输入输出版）
# ## 题目描述
# 给你两个单链表的头节点 `headA` 和 `headB`，找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 `None`。
# 要求：
# 1. 使用ACM本地可运行格式，包含链表定义、数组转链表、链表转数组工具函数
# 2. 自行构造多组测试输入，程序运行后直接打印答案节点的值，无交点输出`-1`
# 3. 最优解法：双指针，时间O(m+n)，空间O(1)

# ### 输入说明
# 输入三组数组：
# - 第一组：链表A不相交部分数组
# - 第二组：链表B不相交部分数组
# - 第三组：两个链表共用的相交尾部数组
# 若相交数组为空代表无交点

# ### 输出说明
# 输出相交节点的值，无交点输出 `-1`

# ## 完整可运行代码（ACM本地IDE直接跑）
# ```python
# 1. 链表节点定义
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# 2. 数组转链表工具
def array_to_linked_list(arr):
    """将数组转换为单链表
    :param arr: List[int] - 输入数组
    :return: ListNode - 头节点
    """
    if not arr:
        return None
    head = ListNode(arr[0])
    current = head
    for val in arr[1:]:
        current.next = ListNode(val)
        current = current.next
    return head

# 3. 链表转数组工具（打印调试用）
def linked_list_to_array(head):
    """将单链表转换为数组
    :param head: ListNode - 头节点
    :return: List[int] - 转换后的数组
    """
    arr = []
    while head:
        arr.append(head.val)
        head = head.next
    return arr

# 4. 核心算法：双指针求相交节点
def getIntersectionNode(headA: ListNode, headB: ListNode) -> ListNode:
    p1, p2 = headA, headB
    while p1 != p2:
        p1 = p1.next if p1 else headB
        p2 = p2.next if p2 else headA
    return p1

# ====================== ACM构造输入、运行、打印输出 ======================
if __name__ == "__main__":
    # 测试用例1：存在相交
    print("=====测试用例1：存在相交链表=====")
    arrA1 = [4, 1]    # A链表独有部分
    arrB1 = [5, 0, 1] # B链表独有部分
    arrInter = [8,4,5]# 公共相交尾部
    # 拼接链表A
    headA = array_to_linked_list(arrA1 + arrInter)
    # 拼接链表B
    headB = array_to_linked_list(arrB1 + arrInter)
    # 计算交点
    res_node1 = getIntersectionNode(headA, headB)
    print(f"链表A完整序列: {linked_list_to_array(headA)}")
    print(f"链表B完整序列: {linked_list_to_array(headB)}")
    print(f"相交节点值: {res_node1.val if res_node1 else -1}\n")

    # 测试用例2：无相交链表
    print("=====测试用例2：无相交链表=====")
    headA2 = array_to_linked_list([1,2,3])
    headB2 = array_to_linked_list([4,5,6])
    res_node2 = getIntersectionNode(headA2, headB2)
    print(f"链表A完整序列: {linked_list_to_array(headA2)}")
    print(f"链表B完整序列: {linked_list_to_array(headB2)}")
    print(f"相交节点值: {res_node2.val if res_node2 else -1}")
# ```

# ## 程序运行输出结果
# ```
# =====测试用例1：存在相交链表=====
# 链表A完整序列: [4, 1, 8, 4, 5]
# 链表B完整序列: [5, 0, 1, 8, 4, 5]
# 相交节点值: 8

# =====测试用例2：无相交链表=====
# 链表A完整序列: [1, 2, 3]
# 链表B完整序列: [4, 5, 6]
# 相交节点值: -1
# ```

# ## 面试官提问拓展（面试加分）
# 1. 这个双指针解法原理是什么？
# > 两条链表长度和固定，指针p1走完A走B，p2走完B走A，路程相等，有交点一定会在交点相遇，无交点同时走到`None`。
# 2. 有没有其他解法？优缺点？
# > 哈希集合：先存一条链表所有节点，遍历第二条判断存在；时间相同，但空间O(n)，不满足原地O(1)要求。
# 3. 边界情况有哪些？
# - 其中一条链表为空
# - 两条链表完全相同（头节点就是交点）
# - 相交点是链表最后一个节点
# - 完全无交点