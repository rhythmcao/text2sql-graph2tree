#coding=utf-8
from typing import Tuple, List
from asdl.asdl import Field
from asdl.hypothesis import Hypothesis
from asdl.transition_system import Action


class ActionInfo(object):
    """sufficient statistics for making a prediction of an action at a time step"""

    def __init__(self, field=None, action=None):
        self.t = 0
        self.parent_t = -1
        self.action = action
        self.frontier_prod = None
        self.frontier_field = field
        self.frontier_type = field.type if field is not None else None

    def __repr__(self):
        repr_str = '%s (t=%d, p_t=%d, frontier_prod=[%s], frontier_field=[%s])' % (repr(self.action),
                                                         self.t,
                                                         self.parent_t,
                                                         self.frontier_prod.__repr__() if self.frontier_prod else 'None',
                                                         self.frontier_field.__repr__(True) if self.frontier_field else 'None')
        return repr_str

def get_action_infos(field_action_pairs: List[Tuple[Field, Action]] = []):
    action_infos = []
    hyp = Hypothesis()
    for t, (field, action) in enumerate(field_action_pairs):
        action_info = ActionInfo(field, action)
        action_info.t = t
        if field is not None:
            action_info.parent_t = hyp.frontier_node.created_time
            action_info.frontier_prod = hyp.frontier_node.production
        hyp.apply_field_action(field, action)
        action_infos.append(action_info)
    return action_infos
