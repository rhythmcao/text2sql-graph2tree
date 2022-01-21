#coding=utf-8

from asdl.asdl import *
from asdl.hypothesis import Hypothesis
from asdl.transition_system import *


class DecodeHypothesis(Hypothesis):
    def __init__(self):
        super(DecodeHypothesis, self).__init__()
        self.action_infos = []

    def clone_and_apply_action_info(self, action_info):
        action = action_info.action

        new_hyp = self.clone_and_apply_action(action)
        new_hyp.action_infos.append(action_info)

        return new_hyp

    def copy(self):
        new_hyp = DecodeHypothesis()
        if self.tree:
            new_hyp.tree = self.tree.copy()

        new_hyp.actions = list(self.actions)
        new_hyp.score = self.score
        new_hyp.t = self.t

        new_hyp.update_frontier_info()

        new_hyp.action_infos = list(self.action_infos)
        return new_hyp
