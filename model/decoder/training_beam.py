#coding=utf8
import torch
from typing import List
from model.decoder.beam import Beam
from asdl.asdl_ast import AbstractSyntaxTree
from asdl.training_hypothesis import TrainingHypothesis
from asdl.transition_system import ApplyRuleAction, SelectColumnAction, SelectTableAction, TransitionSystem

class TrainingBeam(Beam):
    """ Maintain a beam of hypothesis during decoding for each example
    """
    def __init__(self, golden: AbstractSyntaxTree, trans: TransitionSystem, beam_size: int = 5, ts_order: str = 'random', uts_order: str = 'enum', device: torch.device = None) -> None:
        assert beam_size >= 1
        self.trans, self.grammar, self.order_controller = trans, trans.grammar, trans.order_controller
        self.beam_size, self.ts_order, self.uts_order = beam_size, ts_order, uts_order
        self.device = device
        # record the current hypothesis and current input fields
        self.hyps = [TrainingHypothesis(golden)]
        self.fields = [None]
        self.live_hyp_ids = [0]
        self.completed_hyps = []


    def advance(self, ar_score, st_score, sc_score, sv_score):
        """ Four types of scores: num_hyps x rule_num, num_hyps x max_tab_num, num_hyps x max_col_num, num_hyps x max_val_num
        """
        assert len(self.hyps) == len(self.fields) == ar_score.size(0)

        # 1. Compare the hyp with golden tree
        # Attention: for field with more than one values, saved all possible scores for ranking
        metas = [] # (prev_hyp_id, action_id, field_idx, action_score, hyp_score)
        for idx in range(len(self.hyps)):
            field, hyp = self.fields[idx], self.hyps[idx]
            # always points to class AbstractSyntaxTree in the golden tree
            cur_ptr = hyp.golden_ptr

            if not field: # starting timestep, field=None
                p = cur_ptr.production
                score = ar_score[idx, self.grammar.prod2id[p]]
                hyp_score = hyp.score + score
                metas.append((idx, p, 0, score, hyp_score))
                continue

            golden_fields_list = cur_ptr[field]
            act_type = self.trans.field_to_action(field)
            # attention that hyp maintains a pointer to the golden ast
            field_indexes = self.order_controller.get_golden_remaining_actions(hyp, field, uts_order=self.uts_order)
            if act_type == ApplyRuleAction:
                for fid in field_indexes:
                    p = golden_fields_list[fid].value.production
                    score = ar_score[idx, self.grammar.prod2id[p]]
                    hyp_score = hyp.score + score
                    metas.append((idx, p, fid, score, hyp_score))
            elif act_type == SelectTableAction:
                for fid in field_indexes:
                    tid = int(golden_fields_list[fid].value)
                    score = st_score[idx, tid]
                    hyp_score = hyp.score + score
                    metas.append((idx, tid, fid, score, hyp_score))
            elif act_type == SelectColumnAction:
                for fid in field_indexes:
                    cid = int(golden_fields_list[fid].value)
                    score = sc_score[idx, cid]
                    hyp_score = hyp.score + score
                    metas.append((idx, cid, fid, score, hyp_score))
            else: # SelectValueAction
                for fid in field_indexes:
                    vid = int(golden_fields_list[fid].value)
                    score = sv_score[idx, vid]
                    hyp_score = hyp.score + score
                    metas.append((idx, vid, fid, score, hyp_score))

        # 2. For memory sake, only reserve topk path
        hyp_scores = torch.stack([x[-1] for x in metas])
        topk_hyp_scores, meta_ids = hyp_scores.topk(min(self.beam_size, hyp_scores.size(0)), -1, True, True)

        # 3. Update the hypothesis and fields list
        new_hyps, new_fields, live_hyp_ids = [], [], []
        for score, meta_id in zip(topk_hyp_scores, meta_ids.tolist()):
            entry = metas[meta_id]
            hyp_id = entry[0]
            field, hyp = self.fields[hyp_id], self.hyps[hyp_id]
            action = self.trans.field_to_action(field)(entry[1])
            new_hyp = hyp.clone_and_apply_field_action(field, action, idx=entry[2], score=entry[3])
            new_hyp.score = score
            if new_hyp.golden_ptr is None: # a faster criterion
                self.completed_hyps.append(new_hyp)
                continue
            # find next valid fields as inputs
            fields_list = self.order_controller.get_valid_continuing_fields(new_hyp, ts_order=self.ts_order)
            new_fields.extend(fields_list)
            new_hyps.extend([new_hyp] * len(fields_list))
            live_hyp_ids.extend([hyp_id] * len(fields_list))
        assert self.completed_hyps == [] or new_hyps == []
        if new_hyps:
            self.hyps, self.fields, self.live_hyp_ids = new_hyps, new_fields, live_hyp_ids
        else:
            self.hyps, self.fields, self.live_hyp_ids = [], [], []