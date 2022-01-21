#coding=utf8
import torch

class ASTComparison():

    def __init__(self) -> None:
        self.comparison_method = {
            'bandit': self.bandit_reward,
            # more advanced method needs tree dynamic programming, not now
        }

    def bandit_reward(self, hyp, ref_ast):
        # for simplicity, just re-use __eq__ function for ast comparison
        # for children fields, use the hash code for quick fix (may be wrong, but faster)
        if hyp.tree == ref_ast:
            return 1 * hyp.score
        else:
            return -1 * hyp.score

    def compute_reward(self, hyps, ref_ast, rl_method='bandit'):
        rewards = [self.comparison_method[rl_method](hyp, ref_ast) for hyp in hyps if hyp.tree]
        if rewards:
            return torch.stack(rewards).mean()
        else: return None
