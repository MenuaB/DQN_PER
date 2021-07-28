class Node:
    def __init__(self, left, right, is_leaf=False, idx=None):
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        if not self.is_leaf:
            self.value = self.left.value + self.right.value
        self.parent = None
        self.idx = idx
        if left is not None:
            self.left.parent = self
        if right is not None:
            self.right.parent = self

    @classmethod
    def create_leaf(cls, value, idx):
        leaf = cls(None, None, is_leaf=True, idx=idx)
        leaf.value = value
        return leaf


def create_tree(item_list):
    nodes = [Node.create_leaf(v, i) for i, v in enumerate(item_list)]
    leaf_nodes = nodes
    while len(nodes) > 1:
        inodes = iter(nodes)
        nodes = [Node(*pair) for pair in zip(inodes, inodes)]

    return nodes[0], leaf_nodes


def retrieve(value, node):
    if node.is_leaf:
        return node

    if node.left.value >= value:
        return retrieve(value, node.left)
    else:
        return retrieve(value - node.left.value, node.right)


def update(node, new_value):
    change = new_value - node.value

    node.value = new_value
    propagate_change(node.parent, change)


def propagate_change(node, change):
    node.value += change

    if node.parent is not None:
        propagate_change(node.parent, change)


if __name__ == "__main__":

    root, leaves = create_tree([5, 1, 3, 5, 1, 2, 3, 4, 4, 3, 2, 5, 1, 2, 3, 2])
