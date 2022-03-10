#coding=utf-8
import random
import numpy as np
from asdl.asdl import ASDLGrammar
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
    reserved_cspider_raw = Vocab(iterable=['value'], default='value')
    reserved_cspider = Vocab(iterable=['null', 'false', 'true', '0', '1'], default='1')

    @classmethod
    def size(cls, dataset):
        if dataset == 'spider':
            return cls.reserved_spider.vocab_size
        elif dataset == 'dusql':
            return cls.reserved_dusql.vocab_size
        elif dataset == 'cspider_raw':
            return cls.reserved_cspider_raw.vocab_size
        elif dataset == 'cspider':
            return cls.reserved_cspider.vocab_size
        else:
            raise NotImplementedError

    @classmethod
    def vocab(cls, dataset):
        if dataset == 'spider':
            return cls.reserved_spider
        elif dataset == 'dusql':
            return cls.reserved_dusql
        elif dataset == 'cspider_raw':
            return cls.reserved_cspider_raw
        elif dataset == 'cspider':
            return cls.reserved_cspider
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
        from asdl.order_controller import OrderController
        self.order_controller = OrderController(grammar)
        self.parser, self.unparser = None, None


    def get_field_action_pairs(self, asdl_ast, ts_random=False, uts_random=False):
        """ Generate (Field, Action) sequence given the golden ASDL Syntax Tree.
        random~(bool): control the generation order, attention that enum is not feasible~(lead to explosive number of paths)
        """
        output_pairs = []

        prod = asdl_ast.production
        parent_action = ApplyRuleAction(prod)
        output_pairs.append((asdl_ast.parent_field.field if asdl_ast.parent_field is not None else None, parent_action))

        if len(prod.fields) > 1 and asdl_ast.field_order:
            ordered_fields = asdl_ast.field_order
        elif len(prod.fields) > 1:
            ordered_fields = self.order_controller[prod]
        else: ordered_fields = list(prod.fields.keys())

        typed_idxs = np.arange(len(ordered_fields))
        if ts_random and len(ordered_fields) > 1:
            np.random.shuffle(typed_idxs)
        for idx in typed_idxs:
            field = ordered_fields[idx]
            fields_list = asdl_ast[field]
            untyped_idxs = np.arange(len(fields_list))
            if uts_random and len(fields_list) > 1:
                np.random.shuffle(untyped_idxs)
            for jdx in untyped_idxs:
                real_field = fields_list[jdx]
                # is a composite field
                if self.grammar.is_composite_type(real_field.type):
                    field_actions = self.get_field_action_pairs(real_field.value, ts_random, uts_random)
                else:  # is a primitive field
                    field_actions = self.get_primitive_field_actions(real_field)
                output_pairs.extend(field_actions)
        return output_pairs


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
        if field: # typed constraint for grammar rules
            return TransitionSystem.primitive_type_to_action.get(field.type.name, ApplyRuleAction)
        else: return ApplyRuleAction


    @staticmethod
    def get_class_by_dataset(dataset):
        if dataset == 'spider':
            from asdl.spider.sql_transition_system import SQLTransitionSystem
        elif dataset == 'dusql':
            from asdl.dusql.sql_transition_system import SQLTransitionSystem
        elif dataset == 'cspider_raw':
            from asdl.cspider_raw.sql_transition_system import SQLTransitionSystem
        elif dataset == 'cspider':
            from asdl.cspider.sql_transition_system import SQLTransitionSystem
        else:
            raise ValueError('unknown dataset name %s' % dataset)
        return SQLTransitionSystem
