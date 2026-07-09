# 求二叉树的最大深度
from collections import deque
class TreeNode:
    def __init__(self,val,left=None,right=None):
        self.val = val
        self.left = left
        self.right = right
def maxDepth(root):
    if not root:# 1. 结束条件，如果没有节点，深度为 0
        return 0
    left = maxDepth(root.left) # 2. 当前节点要做
    right = maxDepth(root.right)
    return max(left,right)+1 # 3. 返回结果
# 构建树
def is_null(s):
    return s =='null' or s =='NULL' or s == 'None' or s =='#'
def build_tree(level_order):
    if not level_order or is_null(level_order[0]):
        return None
    root = TreeNode(level_order[0])
    queue = deque([root])
    index = 1
    while queue and index < len(level_order):
        node = queue.popleft() #一旦弹出节点，为其分配左右孩子
        if index < len(level_order) and level_order[index] is not None:
            node.left = TreeNode(level_order[index])
            queue.append(node.left)
        index += 1
        if index < len(level_order) and level_order[index] is not None:
            node.right = TreeNode(level_order[index])
            queue.append(node.right)
        index += 1
    return root
level_order= [1, 2, 3, None, None, 4, 5]
#     1
#    / \
#   2   3
#      / \
#     4   5
root = build_tree(level_order)
result = maxDepth(root)
print(result)# 结果是3




    

