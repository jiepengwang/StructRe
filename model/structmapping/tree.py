"""Runtime tree node used by structured-mesh inference outputs."""


class Node:
    """One node in a generated shape hierarchy."""

    def __init__(
        self,
        part_id=0,
        is_leaf=False,
        box=None,
        label=None,
        children=None,
        edges=None,
        full_label=None,
        geo=None,
        geo_feat=None,
        geo_render=None,
        mesh=None,
        mesh_v=None,
        mesh_f=None,
        code_noise=None,
    ):
        self.is_leaf = is_leaf
        self.part_id = part_id
        self.box = box
        self.geo = geo
        self.geo_feat = geo_feat
        self.label = label
        self.full_label = full_label
        self.children = [] if children is None else children
        self.edges = [] if edges is None else edges
        self.geo_render = geo_render
        self.mesh = mesh
        self.mesh_v = mesh_v
        self.mesh_f = mesh_f
        self.code_noise = code_noise

    def depth_first_traversal(self):
        """Return nodes in pre-order, preserving child order."""
        nodes = []
        stack = [self]
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend(reversed(node.children))
        return nodes


def link_tree_nodes(child_nodes, parent_nodes, parent_to_children):
    """Attach child nodes according to a mapping keyed by parent index."""
    for parent_index, child_indices in parent_to_children.items():
        parent = parent_nodes[parent_index]
        if child_indices:
            parent.is_leaf = False
        parent.children.extend(child_nodes[index] for index in child_indices)
    return parent_nodes
