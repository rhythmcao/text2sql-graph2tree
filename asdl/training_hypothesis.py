#coding=utf-8
from asdl.asdl import *
from asdl.asdl_ast import AbstractSyntaxTree
from asdl.transition_system import *
from asdl.hypothesis import Hypothesis

class TrainingHypothesis(Hypothesis):
    """ Based on Hypothesis, also maintain the pointer to the golden AST tree
    """
    def __init__(self, golden_tree=None):
        super(TrainingHypothesis, self).__init__()
        self.golden_ptr = golden_tree

    def get_remaining_field_ids(self, field):
        already_gen_fields = self.frontier_node.field_tracker[field]
        field_num = self.frontier_node.production.fields[field]
        remaining_ids = []
        for idx in range(field_num):
            if idx not in already_gen_fields:
                remaining_ids.append(idx)
        return remaining_ids

    def apply_field_action(self, field, action, idx=0, score=0.):
        """ Different from Hypothesis, field_tracker must be filled with implemented field index in golden_ptr[field],
        remember to update self.golden_ptr after apply action
        """
        if self.tree is None: # the first action
            assert isinstance(action, ApplyRuleAction), 'Invalid action [%s], only ApplyRule action is valid ' \
                                                        'at the beginning of decoding' % (action)
            self.tree = AbstractSyntaxTree(action.production, created_time=0, score=score)
            self.frontier_node = self.tree
        else:
            frontier_fields = self.frontier_node[field]
            ptr_idx = len(self.frontier_node.field_tracker[field])
            realized_field = frontier_fields[ptr_idx]
            # different from Hypothesis, append the index of the golden tree field
            self.frontier_node.field_tracker[field].append(idx)

            if isinstance(field.type, ASDLCompositeType):
                field_value = AbstractSyntaxTree(action.production, created_time=self.t, score=score)
                realized_field.add_value(field_value, realized_time=self.t, score=score)
                if len(field_value.fields) > 0: # deep-first-search
                    self.frontier_node = field_value
                    # remember to move the pointer of golden tree
                    self.golden_ptr = self.golden_ptr[field][idx].value
            else: # fill in a primitive field
                assert isinstance(action, GenTokenAction)
                realized_field.add_value(action.token, realized_time=self.t, score=score)

        self.frontier_node, self.golden_ptr = self.update_frontier_node(self.frontier_node, self.golden_ptr)

        self.t += 1
        self.actions.append(action)
        # self.actions.append((field, action))

    def update_frontier_node(self, node: AbstractSyntaxTree, ptr: AbstractSyntaxTree):
        if node.decode_finished:
            # go upward to find the next unfinished node
            frontier_field = node.parent_field
            if frontier_field is None: # AST is finished
                return None, None
            else:
                node = frontier_field.parent_node
                ptr = ptr.parent_field.parent_node
                return self.update_frontier_node(node, ptr)
        else: # do nothing, still the current node
            return node, ptr

    def copy(self):
        new_hyp = super(TrainingHypothesis, self).copy()
        new_hyp.golden_ptr = self.golden_ptr
        return new_hyp
