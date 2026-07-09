# 中序遍历二叉树
from collections import deque
class TreeNode:
    def __init__(self,val,left=None,right=None):
        self.val = val
        self.left = left
        self.right = right
def invertTree(root):
    if not root:# 1. 结束条件
        return None
    left = invertTree(root.left) # 2. 假设左右子树都完成
    right = invertTree(root.right)
    root.left = right # 3.当前层要做
    root.right = left
    return root # 4. 返回结果
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
def inorderTraversal(root):
    if not root:# 1. 结束条件
        return []
    left = inorderTraversal(root.left) # 2. 假设左右节点已完成
    right = inorderTraversal(root.right)
    return left + [root.val] + right # 3. 返回结果
level_order= [1, 2, 3, None, None, 4, 5]
#     1
#    / \
#   2   3
#      / \
#     4   5
root = build_tree(level_order)
result = invertTree(root)
#     1
#    / \
#   3   2
#  / \
# 5   4
print(inorderTraversal(result))#用中序遍历展示结果 [5, 3, 4, 1, 2]




    

