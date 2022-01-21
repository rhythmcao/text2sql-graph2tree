#coding=utf-8
import random
import numpy as np
from asdl.asdl import ASDLGrammar
from asdl.ast_comparison import ASTComparison
from utils.vocab import Vocab

class Action(object):
    pass


class ApplyRuleAction(Action):
    def __init__(self, production):
        self.production = production

    def __hash__(self):
        return hash(self.production)

    def __eq__(self, other):
        return isinstance(other, ApplyRuleAction) and self.production == other.production

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return 'ApplyRuleAction[%s]' % self.production.__repr__()

class GenTokenAction(Action):
    def __init__(self, token):
        self.token = token

    def __repr__(self):
        return "%s[id=%s]" % (self.__class__.__name__, self.token)

class SelectColumnAction(GenTokenAction):
    @property
    def column_id(self):
        return self.token

class SelectTableAction(GenTokenAction):
    @property
    def table_id(self):
        return self.token

class SelectValueAction(GenTokenAction):
    reserved_spider = Vocab(iterable=['null', 'false', 'true', '0', '1'], default='1')
    reserved_dusql = Vocab(iterable=['否', '是', '0', '1'], default='1')

    @classmethod
    def size(cls, dataset):
        if dataset == 'spider':
            return cls.reserved_spider.vocab_size
        elif dataset == 'dusql':
            return cls.reserved_dusql.vocab_size
        else:
            raise NotImplementedError

    @classmethod
    def vocab(cls, dataset):
        if dataset == 'spider':
            return cls.reserved_spider
        elif dataset == 'dusql':
            return cls.reserved_dusql
        else:
            raise NotImplementedError

    @property
    def value_id(self):
        return self.token


class TransitionSystem(object):

    primitive_type_to_action ={
        'col_id': SelectColumnAction, 'tab_id': SelectTableAction, 'val_id': SelectValueAction
    }

    def __init__(self, grammar: ASDLGrammar):
        self.grammar = grammar
        self.parser, self.unparser = None, None

    def get_field_action_pairs(self, asdl_ast, typed_random=False, untyped_random=False):
        """ Generate (Field, Action) sequence given the golden ASDL Syntax Tree.
        random: control the generation order for the list of RealizedField with the same Field type,
            order for different Field types is controlled by self.grammar.order_controller
        """
        actions = []

        prod = asdl_ast.production
        parent_action = ApplyRuleAction(prod)
        if asdl_ast.parent_field is not None:
            actions.append((asdl_ast.parent_field.field, parent_action))
        else:
            actions.append((None, parent_action))
        ordered_fields = self.grammar.order_controller[prod] if len(prod.fields) > 1 \
            else list(prod.fields.keys()) # just return the dict with only one Field as key
        typed_idxs = np.arange(len(ordered_fields))
        if typed_random and len(ordered_fields) > 1:
            np.random.shuffle(typed_idxs)
        for idx in typed_idxs:
            field = ordered_fields[idx]
            fields_list = asdl_ast.fields[field]
            untyped_idxs = np.arange(len(fields_list))
            if untyped_random and len(untyped_idxs) > 1:
                np.random.shuffle(untyped_idxs)
            for jdx in untyped_idxs:
                real_field = fields_list[jdx]
                # is a composite field
                if self.grammar.is_composite_type(real_field.type):
                    field_actions = self.get_field_action_pairs(real_field.value, typed_random, untyped_random)
                else:  # is a primitive field
                    field_actions = self.get_primitive_field_actions(real_field)
                actions.extend(field_actions)

        return actions


    def ast_to_surface_code(self, asdl_ast, table, value_candidates, *args, **kargs):
        """ table is used to retrieve column and table names by column_id and table_id;
        value_candidates: list of ValueCandidate to extract the corresponding question index span
        """
        return self.unparser.unparse(asdl_ast, table, value_candidates, *args, **kargs)

    def surface_code_to_ast(self, code, sql_values):
        """ sql_values restore pre-retrieved values from the question and sql dict
        """
        return self.parser.parse(code, sql_values)


    def get_primitive_field_actions(self, realized_field):
        return [(realized_field.field, TransitionSystem.primitive_type_to_action[realized_field.type.name](int(realized_field.value)))]

    def field_to_action(self, field):
        if field:
            return TransitionSystem.primitive_type_to_action.get(field.type.name, ApplyRuleAction)
        else: return ApplyRuleAction

    def get_valid_continuing_fields(self, hyp, method='controller'):
        """ Enumerate all valid fields for the current frontier node
        return fields list instead of a single Field object
        """
        frontier_node = hyp.frontier_node
        if len(frontier_node.fields) == 1: # only one Field, directly return
            return list(frontier_node.fields.keys())
        valid_fields = []
        for field in frontier_node.field_tracker:
            if 0 < len(frontier_node.field_tracker[field]) < len(frontier_node.fields[field]):
                return [field]
            elif len(frontier_node.field_tracker[field]) == len(frontier_node.fields[field]): # finished
                continue
            else: valid_fields.append(field)
        if not valid_fields: raise ValueError('No continuing field found!')
        if method == 'all':
            return valid_fields
        elif method == 'controller':
            for field in self.grammar.order_controller[frontier_node.production]:
                if field in valid_fields:
                    return [field]
            raise ValueError('No continuing field found!')
        elif method == 'random':
            return [random.choice(valid_fields)]
        else:
            raise ValueError('Not recognized order method %s!' % (method))

    @staticmethod
    def get_class_by_dataset(lang):
        if lang == 'spider':
            from asdl.spider.sql_transition_system import SQLTransitionSystem
        elif lang == 'dusql':
            from asdl.dusql.sql_transition_system import SQLTransitionSystem
        else:
            raise ValueError('unknown language %s' % lang)
        return SQLTransitionSystem
