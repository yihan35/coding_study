class ListNode:
    def __init__(self,val=0,next=None):
        self.val = val
        self.next = next
        
def reverseList(head):
    prev = None
    curr = head
    while curr:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt
    return prev

head = ListNode(1)
head.next = ListNode(2)
head.next.next = ListNode(3)
head.next.next.next = ListNode(4)
head.next.next.next.next = ListNode(5)

new_head = reverseList(head)
while new_head:
    print(new_head.val,end = ' ')
    new_head =new_head.next
