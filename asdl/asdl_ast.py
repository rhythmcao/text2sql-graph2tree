#coding=utf-8
from io import StringIO
import copy
from asdl.asdl import *
from collections import defaultdict

class AbstractSyntaxTree(object):
    def __init__(self, production, realized_fields=None, created_time=0, parent=None, score=0.):
        self.production = production

        # a child is essentially a *realized_field*
        self.fields = defaultdict(list)

        # record its parent field to which it's attached
        self.parent_field = parent

        # used in decoding, record the time step when this node was created
        self.created_time = created_time
        self.field_tracker = {
            field: [] for field in self.production.fields
        } # record the index of generated RealizedField compared to golden tree
        self.score = score

        if realized_fields:
            for field in realized_fields:
                self.add_child(field)
            self.sanity_check()
        else:
            for field in self.production.fields:
                for _ in range(self.production.fields[field]):
                    self.add_child(RealizedField(field))

    def add_child(self, realized_field):
        self.fields[realized_field.field].append(realized_field)
        realized_field.parent_node = self

    def __getitem__(self, field):
        if field not in self.fields:
            raise KeyError
        return self.fields[field]

    def sanity_check(self):
        assert len(self.production.fields) == len(self.fields), 'Number of different fields must match'
        for field_key in self.production.fields:
            cnt = self.production.fields[field_key]
            if field_key not in self.fields:
                raise KeyError
            assert len(self.fields[field_key]) == cnt
            for child in self.fields[field_key]:
                assert field_key == child.field
                if isinstance(child.value, AbstractSyntaxTree):
                    child.value.sanity_check()

    def copy(self):
        new_tree = AbstractSyntaxTree(self.production, created_time=self.created_time, score=self.score)
        for field_key in self.fields:
            new_field_list = new_tree.fields[field_key]
            for idx, old_field in enumerate(self.fields[field_key]):
                new_field = new_field_list[idx]
                if isinstance(old_field.type, ASDLCompositeType):
                    if old_field.value is not None:
                        new_field.add_value(old_field.value.copy(), old_field.realized_time, old_field.score)
                else:
                    if old_field.value is not None:
                        new_field.add_value(old_field.value, old_field.realized_time, old_field.score)
        new_tree.field_tracker = copy.deepcopy(self.field_tracker)
        return new_tree

    def to_string(self, sb=None):
        is_root = False
        if sb is None:
            is_root = True
            sb = StringIO()

        sb.write('[')
        sb.write(self.production.constructor.name)

        for field_name in self.fields:
            for field in self.fields[field_name]:
                sb.write(' ')
                sb.write('(')
                sb.write(field.type.name)
                sb.write('-')
                sb.write(field.name)

                if field.value is not None:
                    value = field.value
                    sb.write(' ')
                    if isinstance(field.type, ASDLCompositeType):
                        value.to_string(sb)
                    else:
                        sb.write(str(value))
                else:
                    sb.write(' ?')
                sb.write(')')  # of field

        sb.write(']')  # of node

        if is_root:
            return sb.getvalue()

    def __hash__(self):
        code = hash(self.production)
        for field_name in self.fields:
            for field in self.fields[field_name]:
                code = code + 37 * hash(field)
        return code

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        if self.production != other.production:
            return False

        # TODO: FIX set comparison
        for field_name in self.fields:
            if set(self.fields[field_name]) != set(other.fields[field_name]): return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return repr(self.production)

    @property
    def size(self):
        node_num = 1
        for field_name in self.fields:
            for field in self.fields[field_name]:
                value = field.value
                if isinstance(value, AbstractSyntaxTree):
                    node_num += value.size
                else: node_num += 1
        return node_num

    @property
    def finished(self):
        for field_name in self.fields:
            for field in self.fields[field_name]:
                if not field.finished:
                    return False
        return True
    
    @property
    def decode_finished(self):
        for field_name in self.field_tracker:
            if len(self.field_tracker[field_name]) != self.production.fields[field_name]:
                return False
        return True


class RealizedField(Field):
    """ Wrapper of field realized with values """
    def __init__(self, field, value=None, realized_time=0, parent=None):
        super(RealizedField, self).__init__(field.name, field.type)

        # FIXME: hack, return the field as a property
        self.field = field

        # record its parent AST node
        self.parent_node = parent

        # used in decoding, record the time step when this field was realized
        self.realized_time = realized_time
        self.score = 0.

        if value is not None:
            self.add_value(value)
        else:
            self.value = None


    def add_value(self, value, realized_time=0, score=0.):
        if isinstance(value, AbstractSyntaxTree):
            value.parent_field = self
        self.value = value
        self.realized_time = realized_time
        self.score = score

    def __hash__(self):
        code = hash(self.field) ^ hash(self.value)
        return code

    @property
    def finished(self):
        if self.value is None: return False
        if isinstance(self.value, AbstractSyntaxTree):
            return self.value.finished
        else: return True


    def __eq__(self, other):
        if not isinstance(other, self.__class__): return False
        return super(RealizedField, self).__eq__(other) and \
            self.value == other.value