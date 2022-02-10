#coding=utf-8
from asdl.asdl import *
from asdl.asdl_ast import AbstractSyntaxTree
from asdl.transition_system import *

class Hypothesis(object):
    """ Given the (Field, Action) list, construct the target SQL AST,
    and save the necessary infos needed during training, such as frontier_node, t, score
    """
    def __init__(self):
        self.tree = None
        self.field_actions = []
        self.frontier_node = None

        # record the current time step
        self.t = 0
        self.score = 0.

    def apply_field_action(self, field, action, idx=0, score=0.):
        if self.tree is None: # the first action
            assert isinstance(action, ApplyRuleAction), 'Invalid action [%s], only ApplyRule action is valid ' \
                                                        'at the beginning of decoding' % (action)
            self.tree = AbstractSyntaxTree(action.production, created_time=0, score=score)
            self.frontier_node = self.tree
        else:
            frontier_fields = self.frontier_node[field]
            # no golden tree available, directly append the plus 1 index
            ptr_idx = len(self.frontier_node.field_tracker[field])
            realized_field = frontier_fields[ptr_idx]
            self.frontier_node.field_tracker[field].append(idx)
            if isinstance(field.type, ASDLCompositeType):
                field_value = AbstractSyntaxTree(action.production, created_time=self.t, score=score)
                realized_field.add_value(field_value, realized_time=self.t, score=score)
                if len(field_value.fields) > 0: # deep-first-search
                    self.frontier_node = field_value
            else: # fill in a primitive field
                assert isinstance(action, GenTokenAction)
                realized_field.add_value(action.token, realized_time=self.t, score=score)

        self.frontier_node = self.update_frontier_node(self.frontier_node)

        self.t += 1
        self.field_actions.append((field, action))

    def update_frontier_node(self, node: AbstractSyntaxTree):
        if node.decode_finished:
            # go upward to find the next unfinished node
            frontier_field = node.parent_field
            if frontier_field is None: # AST is finished
                return None
            else:
                node = frontier_field.parent_node
                return self.update_frontier_node(node)
        else: # do nothing, still the current node
            return node

    def clone_and_apply_field_action(self, field, action, idx=0, score=0.):
        new_hyp = self.copy()
        new_hyp.apply_field_action(field, action, idx=idx, score=score)
        return new_hyp

    def copy(self):
        new_hyp = type(self)()
        if self.tree:
            new_hyp.tree = self.tree.copy()
            # find and set the frontier_node for the new tree using backtrace
            new_hyp.retrieve_and_set_frontier_node(self)

        new_hyp.field_actions = list(self.field_actions)
        new_hyp.score = self.score
        new_hyp.t = self.t
        return new_hyp

    def retrieve_frontier_node_path(self):
        cur_node, paths = self.frontier_node, []
        while cur_node.parent_field is not None:
            cur_field = cur_node.parent_field
            ptr_idx = len(cur_field.parent_node.field_tracker[cur_field.field]) - 1
            paths.append((cur_field.field, ptr_idx))
            cur_node = cur_field.parent_node
        return paths[::-1]

    def retrieve_and_set_frontier_node(self, ref_hyp):
        bp_paths = ref_hyp.retrieve_frontier_node_path()
        cur_node = self.tree # start with root node
        for field, ptr_idx in bp_paths:
            cur_node = cur_node[field][ptr_idx].value
        self.frontier_node = cur_node

    @property
    def completed(self):
        return self.tree and self.tree.finished
