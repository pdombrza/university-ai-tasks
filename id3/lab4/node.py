class Node:
    def __init__(self, value=None, attr=None):
        self.value = value
        self.attr = attr
        self.children = {}

    def set_children(self, new_children: dict) -> None:
        self.children = new_children

    def add_child(self, attribute_val: int, new_child) -> None:
        self.children[attribute_val] = new_child

    def is_leaf(self) -> bool:
        if not self.children:
            return True
        return False