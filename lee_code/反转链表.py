# 滴滴

class ListNode:
    def __init__(self,val=0,next=None):
        self.val = val
        self.next = next

def reverseList(head):
    pre = None
    curr = head
    while curr:
        nxt = curr.next
        curr.next = pre
        pre = curr
        curr = nxt
    return pre

head = ListNode(1)
head.next = ListNode(2)
head.next.next = ListNode(3)
head.next.next.next = ListNode(4)
head.next.next.next.next = ListNode(5)

# origin = head

# while origin:
#     print(origin.val,end=' ')
#     origin = origin.next


new_head = reverseList(head)


while new_head:
    print(new_head.val,end=' ')
    new_head = new_head.next
