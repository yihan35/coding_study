# 中序遍历二叉树
from collections import deque
class TreeNode:
    def __init__(self,val,left=None,right=None):
        self.val = val
        self.left = left
        self.right = right
def minDepth(root):
    if not root:# 1. 结束条件
        return 0
    if not root.left and not root.right:#左右节点都不存在，只有 root
        return 1
    if not root.left:# 左路径不存在，最小深度只能从右侧取
        return minDepth(root.right)+1
    if not root.right: # 同理
        return minDepth(root.right)+1
    # 左右节点都存在
    left = minDepth(root.left)
    right = minDepth(root.right)
    return min(left,right)+1 # 3. 返回结果
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
result = minDepth(root)
print(result)




    

