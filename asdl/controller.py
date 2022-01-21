#coding=utf8
import pickle
import numpy as np
from typing import Any, List, Tuple, Dict
from asdl.asdl import ASDLProduction, Field
from asdl.asdl_ast import AbstractSyntaxTree
from collections import Counter, defaultdict

class OrderController():
    """ Used to control the generation order of different fields in the production
    """
    def __init__(self, prod2id: Dict[ASDLProduction, int], field2id: Dict[Field, int]) -> None:
        self.prod2id, self.field2id = prod2id, field2id
        # only preserve production rules with number of different Fields > 1
        self.prod2fields = { prod: sorted(prod.fields.keys(), key=lambda x: repr(x)) for prod in prod2id if len(prod.fields) > 1 }
        # print('Number of productions which have more than one different types of Field:', len(self.prod2fields))

    def __getitem__(self, prod):
        return self.prod2fields[prod]

    def sanity_check(self, prod2fields: Dict[ASDLProduction, List[Field]]):
        for prod in prod2fields:
            fields = prod2fields[prod]
            if len(fields) != len(prod.fields):
                raise ValueError('Missing or redundant Field for ASDLProduction %s in prod2fields dict.' % (prod.__repr__()))
            for field in fields:
                if field not in prod.fields:
                    raise ValueError('Invalid %s for ASDLProduction %s in prod2fields dict.' % (field.__repr__(), prod.__repr__()))

    def set_order(self, prod2fields: Dict[ASDLProduction, List[Field]]):
        """ Incorporate prior knowledge into the structured generation process
        """
        prod2fields = { prod: list(prod2fields[prod]) for prod in prod2fields if len(prod.fields) > 1 and prod in self.prod2fields }
        self.sanity_check(prod2fields)
        for prod in prod2fields:
            self.prod2fields[prod] = [f.copy() for f in prod2fields[prod]]

    def shuffle_order(self):
        """ Shuffle the Field generation order for each production which has more than two types of children
        """
        for prod in self.prod2fields:
            fields_list = self.prod2fields[prod]
            if len(fields_list) > 1:
                index = np.arange(len(fields_list))
                np.random.shuffle(index)
                self.prod2fields[prod] = [fields_list[idx] for idx in index]

    def order_history(self, hyps_list, ids, results=None, idx=-1):
        """ Record the order history for each example, for training analysis
        """
        asts = [hyps[0].tree for hyps in hyps_list]
        if results is None:
            results = defaultdict(list)
        for ex_id, ast in zip(ids, asts):
            typed_dct, untyped_dct = record(ast, self.prod2id, self.field2id, dict(), dict(), path=[])
            results[ex_id].append((typed_dct, untyped_dct, idx))
        return results

    def order_change_trajectory(self, history):
        pass

    def order_statistics(self, hyps_list, results=None):
        """ Accumulate the generation order for each production rule given the hyps_list
        """
        if results is None:
            results = defaultdict(list)
        asts = [hyps[0].tree for hyps in hyps_list] # only top of beam tree
        for ast in asts:
            results = update_prod2fields(ast, results)
        return results

    def compute_best_order(self, results, save_path=None, verbose=False):
        results = { p: Counter(results[p]) for p in results }
        if save_path:
            pickle.dump(results, open(save_path, 'wb'))
        most_common_results = { p: list(results[p].most_common(1)[0][0]) for p in results }
        if verbose:
            self.print_best_order(most_common_results)
        return most_common_results

    def print_best_order(self, prod2fields):
        prods = sorted(prod2fields.keys(), key=lambda p: repr(p))
        for p in prods:
            print(repr(p), '==>', ', '.join([repr(f) for f in prod2fields[p]]))

def update_prod2fields(ast: AbstractSyntaxTree, dct: Dict[ASDLProduction, List[Tuple[Field]]] = None):
    """ Record the generation order for each production which has more than one types of Fields as children nodes
    dct: dictionary which maps a ASDLProduction into the list of all occurrences of different typed Field tuples
    """
    if len(ast.fields) > 1: # more than 1 different types
        field_times = [(f, min([rf.realized_time for rf in ast.fields[f]])) for f in ast.fields]
        dct[ast.production].append(tuple(map(lambda ft: ft[0], sorted(field_times, key=lambda x: x[1]))))
    for f in ast.fields:
        for rf in ast.fields[f]:
            value = rf.value
            if isinstance(value, AbstractSyntaxTree):
                dct = update_prod2fields(value, dct)
    return dct

def record(ast: AbstractSyntaxTree, prod2id: Dict[ASDLProduction, int], field2id: Dict[Field, int],
        typed_dct: Dict[Tuple[Any], Tuple[Field]] = None,
        untyped_dct: Dict[Tuple[Any], Tuple[int]] = None, path: List[Any] = []):
    """ Record the generation order of both typed and untyped fields for each example
    """
    path = list(path)
    path.append(prod2id[ast.production])
    if len(ast.fields) > 1: # more than 1 different types
        field_times = [(field2id[f], min([rf.realized_time for rf in ast.fields[f]])) for f in ast.fields]
        typed_dct[tuple(path)] = tuple(map(lambda ft: ft[0], sorted(field_times, key=lambda x: x[1])))
    for f in ast.fields:
        realized_fields, golden_idxs = ast.fields[f], ast.field_tracker[f]
        if len(realized_fields) > 1:
            untyped_dct[tuple(path + [field2id[f]])] = tuple(golden_idxs)
        for rf, idx in zip(realized_fields, golden_idxs):
            value = rf.value
            if isinstance(value, AbstractSyntaxTree):
                child_path = path + [(field2id[f], idx)]
                typed_dct, untyped_dct = record(value, prod2id, field2id, typed_dct, untyped_dct, child_path)
    return typed_dct, untyped_dct
