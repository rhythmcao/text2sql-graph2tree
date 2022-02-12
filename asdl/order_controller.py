#coding=utf8
import math, pickle, random
import numpy as np
from typing import List, Dict
from asdl.asdl import ASDLCompositeType, ASDLGrammar, ASDLProduction, Field
from asdl.action_info import get_action_infos
from collections import Counter, defaultdict
from utils.example import SQLDataset


class OrderController():

    """ Used to control the generation order of an AST.
    For typed set, control the generation order of different field types (high-level) for each grammar rule;
    For untyped set, control the generation order of different fields with the same type in the golden tree.
    """

    def __init__(self, grammar: ASDLGrammar) -> None:
        self.prod2id, self.id2prod = grammar.prod2id, grammar.id2prod
        self.field2id, self.id2field = grammar.field2id, grammar.id2field
        # high-level controller for grammar rules, only preserve production rules with number of different Field objects > 1
        self.prod2fields = { prod: sorted(prod.fields.keys(), key=lambda x: repr(x)) for prod in self.prod2id if len(prod.fields) > 1 }
        # print('Number of productions which have more than one different types of Field:', len(self.prod2fields))
        self.history = defaultdict(dict) # first key is the sample id, second key is epoch number
        self.canonical_actions = {} # key is the sample id


    def __getitem__(self, prod):
        return self.prod2fields[prod]


    def shuffle_ts_order(self):
        """ Shuffle the Field generation order for each grammar rule~(high-level)
        """
        for prod in self.prod2fields:
            fields_list = self.prod2fields[prod]
            index = np.arange(len(fields_list))
            np.random.shuffle(index)
            self.prod2fields[prod] = [fields_list[idx] for idx in index]


    def set_canonical_order_per_sample(self, dataset: SQLDataset, ts_control: bool = True, ts_shuffle: bool = False, uts_shuffle: bool = True):
        """ Set the canonical generation order for each training sample, including
        order for typed set, and order for untyped set. If ts_control, use the high-level order for typed set and ignore ts_shuffle.
        """
        def set_canonical_order_for_ast(ast):
            if len(ast.fields) == 0: return
            if not ast.field_order: # initialize order for typed set
                ast.field_order = list(self[ast.production]) if len(ast.fields) > 1 else list(ast.fields.keys())
            if len(ast.fields) > 1:
                if ts_control:
                    ast.field_order = list(self[ast.production])
                elif ts_shuffle:
                    np.random.shuffle(ast.field_order)

            for field in ast.field_tracker:
                if not ast.field_tracker[field]: # initialize order for untyped set
                    ast.field_tracker[field] = list(range(len(ast[field])))
                if len(ast.field_tracker[field]) > 1 and uts_shuffle:
                    np.random.shuffle(ast.field_tracker[field])

                if isinstance(field.type, ASDLCompositeType): # dive into sub-trees
                    for realized_field in ast[field]:
                        sub_ast = realized_field.value
                        set_canonical_order_for_ast(sub_ast)

        for ex in dataset:
            ast = ex.ast
            set_canonical_order_for_ast(ast)
        return dataset


    def get_valid_continuing_fields(self, hyp, ts_order='controller'):
        """ Based on the order method for typed set, return valid fields list as the next input node
        """
        frontier_node = hyp.frontier_node
        if len(frontier_node.fields) == 1: # only one Field, directly return
            return list(frontier_node.fields.keys())

        valid_fields = []
        for field in frontier_node.field_tracker:
            if 0 < len(frontier_node.field_tracker[field]) < len(frontier_node[field]):
                # hierarchically, process this Field first, then switch to other Field objs
                return [field]
            elif len(frontier_node.field_tracker[field]) == len(frontier_node[field]): # finished
                continue
            else: valid_fields.append(field)
        if not valid_fields: raise ValueError('No continuing field found!')

        if ts_order == 'enum':
            return valid_fields
        elif ts_order == 'controller':
            canonical_order = hyp.golden_ptr.field_order if hasattr(hyp, 'golden_ptr') \
                else self[frontier_node.production]
            for field in canonical_order:
                if field in valid_fields:
                    return [field]
            raise ValueError('No continuing field found!')
        elif ts_order == 'random':
            return [random.choice(valid_fields)]
        else:
            raise ValueError('Not recognized order method %s for input field!' % (ts_order))


    def get_golden_remaining_actions(self, hyp, field, uts_order='enum'):
        """ Based on the order method for untyped set, return valid indexes of fields as the target output
        """
        if len(hyp.golden_ptr[field]) == 1: # only one choice
            return [0]
        already_gen_fields = hyp.frontier_node.field_tracker[field]
        all_field_ids = range(len(hyp.frontier_node[field])) if not hyp.golden_ptr.field_tracker[field] \
            else hyp.golden_ptr.field_tracker[field]
        remaining_ids = [idx for idx in all_field_ids if idx not in already_gen_fields]
        if uts_order == 'enum':
            return remaining_ids
        elif uts_order == 'controller':
            return remaining_ids[:1]
        elif uts_order == 'random':
            return [random.choice(remaining_ids)]
        else:
            raise ValueError('Not recognized order method %s for output action!' % (uts_order))


    def sanity_check(self, prod2fields: Dict[ASDLProduction, List[Field]]):
        for prod in prod2fields:
            fields = prod2fields[prod]
            if len(fields) != len(prod.fields):
                raise ValueError('Missing or redundant Field for ASDLProduction %s in prod2fields dict.' % (prod.__repr__()))
            for field in fields:
                if field not in prod.fields:
                    raise ValueError('Invalid %s for ASDLProduction %s in prod2fields dict.' % (field.__repr__(), prod.__repr__()))


    def load_and_set_ts_order(self, order):
        """ Incorporate prior knowledge into the structured generation process
        """
        if type(order) is str:
            prod2fields = pickle.load(open(order, 'rb'))
            print(f'Load canonical order for typed set from path {order}')
        else: prod2fields = order
        self.sanity_check(prod2fields)
        for prod in prod2fields:
            if prod in self.prod2fields:
                self.prod2fields[prod] = list(prod2fields[prod])
            else: raise ValueError(f'[Error]: There exists grammar rule {prod:s} not in the canonical order dictionary.')


    def record_ast_generation_order(self, hyps_list, ids, epoch: int = 0):
        """ Record the generation order for each training sample during training        
        """
        if not hyps_list: return

        def dfs_ast_traversal(ast, path, results):
            """ keys in dict results are tuples of alternating ASDLProduction and (Field, idx),
            if the length of tuple is odd, typed set; o.w. even -> untyped set.
            """
            path = list(path)
            path.append(ast.production)
            if len(ast.fields) > 1: # typed set with fields number > 1
                rf_times = [(f, min([rf.realized_time for rf in ast[f]])) for f in ast.fields]
                results[tuple(path)] = tuple(map(lambda ft: ft[0], sorted(rf_times, key=lambda x: x[1])))
            for f in ast.fields:
                rf_list, indexes = ast[f], ast.field_tracker[f]
                if len(rf_list) > 1:
                    # untyped set with realized fields number > 1
                    results[tuple(path + [f])] = tuple(indexes)
                if isinstance(f.type, ASDLCompositeType):
                    for rf, idx in zip(rf_list, indexes):
                        child_ast = rf.value
                        child_path = path + [(f, idx)]
                        dfs_ast_traversal(child_ast, child_path, results)

        asts = [hyps[0].tree for hyps in hyps_list]
        for idx, ast in zip(ids, asts):
            order_dict = {}
            dfs_ast_traversal(ast, [], order_dict)
            self.history[idx][int(epoch)] = order_dict
        return


    def accumulate_canonical_ts_order(self, history=None, reference_epoch=-1, save_path=None, verbose=True):
        typed_dict = defaultdict(list)
        if history is None: history = self.history
        for ex_id in history.keys():
            max_epoch = max(history[ex_id].keys()) if reference_epoch not in history[ex_id] else reference_epoch
            order_dict = history[ex_id][max_epoch]
            for key in order_dict:
                if len(key) % 2 == 1: # typed set
                    typed_dict[key[-1]].append(tuple(order_dict[key]))
        ts_order_counter = dict([(prod, Counter(typed_dict[prod])) for prod in typed_dict])
        canonical_ts_order = dict([(prod, cnt.most_common(1)[0][0]) for prod, cnt in ts_order_counter.items()])

        if verbose:
            def calculate_ts_entropy(counter):
                total_count = sum(counter.values())
                entropy = 0.
                for field_tuple in counter:
                    prob = counter[field_tuple] / float(total_count)
                    # set the base for log, such that the result is normalized to [0, 1]
                    entropy -= prob * math.log(prob, math.factorial(len(field_tuple)))
                return entropy

            ts_order_entropy = dict([(prod, calculate_ts_entropy(cnt)) for prod, cnt in ts_order_counter.items()])
            self.print_canonical_ts_order(canonical_ts_order, ts_order_entropy)

        if save_path is not None:
            with open(save_path, 'wb') as of:
                pickle.dump(canonical_ts_order, of)
        return canonical_ts_order


    def print_canonical_ts_order(self, order_dict, entropy_dict):
        print('Canonical generation order for each grammar rule:')
        for prod in sorted(order_dict.keys(), key=lambda p: repr(p)):
            print(f'[Entropy:{entropy_dict[prod]:.4f}]: {repr(prod)} ==> ', ', '.join([repr(f) for f in order_dict[prod]]))
        return


    def save_order_history(self, save_path):
        with open(save_path, 'wb') as of:
            pickle.dump(self.history, of)


    def record_canonical_actions(self, hyps_list, ids):
        """ Record canonical field actions, including generation order of both typed sets and untyped sets
        """
        all_field_actions = [hyps[0].field_actions for hyps in hyps_list]
        for ex_id, field_actions in zip(ids, all_field_actions):
            action_infos = get_action_infos(field_actions)
            self.canonical_actions[ex_id] = action_infos
        return


    def save_canonical_actions(self, save_path):
        with open(save_path, 'wb') as of:
            pickle.dump(self.canonical_actions, of)
        return


    def load_and_set_canonical_actions(self, load_path, dataset: SQLDataset):
        with open(load_path, 'rb') as inf:
            self.canonical_actions = pickle.load(inf)
        for ex in dataset:
            ex.canonical_action = list(self.canonical_actions[ex.id])
        return dataset
