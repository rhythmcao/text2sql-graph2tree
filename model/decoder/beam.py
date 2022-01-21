#coding=utf8
import torch
from typing import Tuple, List
from asdl.hypothesis import Hypothesis
from asdl.transition_system import ApplyRuleAction, SelectColumnAction, SelectTableAction, TransitionSystem

class Beam():
    """ Maintain a beam of hypothesis during decoding for each example
    """
    def __init__(self, size: Tuple[int, int, int], trans: TransitionSystem, beam_size: int = 5, order_method: str = 'controller', device: torch.device = None) -> None:
        assert beam_size >= 1
        self.trans, self.grammar = trans, trans.grammar
        self.table_num, self.column_num, self.value_num = size
        self.beam_size = beam_size
        self.order_method = order_method
        self.device = device
        # record the current hypothesis and current input fields
        self.hyps = [Hypothesis()]
        self.fields = [None]
        self.live_hyp_ids = [0]
        self.completed_hyps = []

    def get_parent_prod_ids(self) -> torch.LongTensor:
        return torch.tensor([self.grammar.prod2id[hyp.frontier_node.production] for hyp in self.hyps], dtype=torch.long, device=self.device)

    def get_parent_field_ids(self) -> torch.LongTensor:
        return torch.tensor([self.grammar.field2id[f] for f in self.fields], dtype=torch.long, device=self.device)

    def get_parent_type_ids(self) -> torch.LongTensor:
        return torch.tensor([self.grammar.type2id[f.type] for f in self.fields], dtype=torch.long, device=self.device)

    def get_parent_timesteps(self) -> torch.LongTensor:
        return torch.tensor([hyp.frontier_node.created_time for hyp in self.hyps], dtype=torch.long, device=self.device)

    def get_previous_actions(self):
        return [hyp.actions[-1] for hyp in self.hyps]

    def get_previous_hyp_ids(self, offset: int = 0) -> List[int]:
        return [idx + offset for idx in self.live_hyp_ids]

    def advance(self, ar_score, st_score, sc_score, sv_score):
        """ Four types of scores: num_hyps x rule_num, num_hyps x max_tab_num, num_hyps x max_col_num, num_hyps x max_val_num
        """
        assert len(self.hyps) == len(self.fields) == ar_score.size(0)
        metas = [] # (prev_hyp_id, action_id, action_score, hyp_score)
        for idx in range(len(self.hyps)):
            field, hyp = self.fields[idx], self.hyps[idx]
            act_type = self.trans.field_to_action(field)
            if act_type == ApplyRuleAction:
                productions = self.grammar[field.type] if field else self.grammar[self.grammar.root_type]
                for p in productions:
                    pid = self.grammar.prod2id[p]
                    score = ar_score[idx, pid]
                    hyp_score = hyp.score + score
                    metas.append((idx, p, score, hyp_score))
            elif act_type == SelectTableAction:
                for tid in range(self.table_num):
                    score = st_score[idx, tid]
                    hyp_score = hyp.score + score
                    metas.append((idx, tid, score, hyp_score))
            elif act_type == SelectColumnAction:
                for cid in range(self.column_num):
                    score = sc_score[idx, cid]
                    hyp_score = hyp.score + score
                    metas.append((idx, cid, score, hyp_score))
            else:
                for vid in range(self.value_num):
                    score = sv_score[idx, vid]
                    hyp_score = hyp.score + score
                    metas.append((idx, vid, score, hyp_score))

        hyp_scores = torch.stack([x[-1] for x in metas])
        topk_hyp_scores, meta_ids = hyp_scores.topk(min(self.beam_size, hyp_scores.size(0)), -1, True, True)

        # update the hypothesis and fields list
        new_hyps, new_fields, live_hyp_ids = [], [], []
        for score, meta_id in zip(topk_hyp_scores, meta_ids.tolist()):
            entry = metas[meta_id]
            hyp_id = entry[0]
            field, hyp = self.fields[hyp_id], self.hyps[hyp_id]
            action = self.trans.field_to_action(field)(entry[1])
            new_hyp = hyp.clone_and_apply_field_action(field, action, score=entry[2])
            new_hyp.score = score
            if new_hyp.completed:
                self.completed_hyps.append(new_hyp)
                continue
            # find next valid fields
            fields_list = self.trans.get_valid_continuing_fields(new_hyp, method=self.order_method)
            new_fields.extend(fields_list)
            new_hyps.extend([new_hyp] * len(fields_list))
            live_hyp_ids.extend([hyp_id] * len(fields_list))

        if new_hyps:
            self.hyps, self.fields, self.live_hyp_ids = new_hyps, new_fields, live_hyp_ids
        else:
            self.hyps, self.fields, self.live_hyp_ids = [], [], []

    @property
    def done(self):
        # return len(self.completed_hyps) >= self.beam_size or self.hyps == []
        return self.hyps == []

    def sort_finished(self):
        if len(self.completed_hyps) > 0:
            size = min([self.beam_size, len(self.completed_hyps)])
            completed_hyps = sorted(self.completed_hyps, key=lambda hyp: - hyp.score)[:size] # / hyp.tree.size
        else:
            completed_hyps = [Hypothesis()]
        return completed_hyps
