#coding=utf8
import torch
from utils.constants import PAD
from utils.dusql.example import Example, get_position_ids
from model.model_utils import lens2mask, cached_property
from asdl.action_info import get_action_infos

def from_example_list_base(ex_list, device='cpu', train=True):
    """
        question_lens: torch.long, bsize
        questions: torch.long, bsize x max_question_len, include [CLS] if add_cls
        table_lens: torch.long, bsize, number of tables for each example
        table_word_lens: torch.long, number of words for each table name
        tables: torch.long, sum_of_tables x max_table_word_len
        column_lens: torch.long, bsize, number of columns for each example
        column_word_lens: torch.long, number of words for each column name
        columns: torch.long, sum_of_columns x max_column_word_len
    """
    batch = Batch(ex_list, device)
    pad_idx = Example.tokenizer.pad_token_id

    question_lens = [len(ex.question) for ex in ex_list]
    batch.question_lens = torch.tensor(question_lens, dtype=torch.long, device=device)
    batch.table_lens = torch.tensor([len(ex.table) for ex in ex_list], dtype=torch.long, device=device)
    table_word_lens = [len(t) for ex in ex_list for t in ex.table]
    batch.table_word_lens = torch.tensor(table_word_lens, dtype=torch.long, device=device)
    batch.column_lens = torch.tensor([len(ex.column) for ex in ex_list], dtype=torch.long, device=device)
    column_word_lens = [len(c) for ex in ex_list for c in ex.column]
    batch.column_word_lens = torch.tensor(column_word_lens, dtype=torch.long, device=device)

    # prepare inputs for pretrained models
    batch.inputs = {"input_ids": None, "attention_mask": None, "token_type_ids": None, "position_ids": None}
    input_lens = [len(ex.input_id) for ex in ex_list]
    max_len = max(input_lens)
    input_ids = [ex.input_id + [pad_idx] * (max_len - len(ex.input_id)) for ex in ex_list]
    batch.inputs["input_ids"] = torch.tensor(input_ids, dtype=torch.long, device=device)
    attention_mask = [[1] * l + [0] * (max_len - l) for l in input_lens]
    batch.inputs["attention_mask"] = torch.tensor(attention_mask, dtype=torch.float, device=device)
    token_type_ids = [ex.segment_id + [0] * (max_len - len(ex.segment_id)) for ex in ex_list]
    batch.inputs["token_type_ids"] = torch.tensor(token_type_ids, dtype=torch.long, device=device)
    position_ids = [get_position_ids(ex, shuffle=train) + [0] * (max_len - len(ex.input_id)) for ex in ex_list]
    batch.inputs["position_ids"] = torch.tensor(position_ids, dtype=torch.long, device=device)
    # extract representations after plm, remove [SEP]
    question_mask_plm = [ex.question_mask_plm + [0] * (max_len - len(ex.question_mask_plm)) for ex in ex_list]
    batch.question_mask_plm = torch.tensor(question_mask_plm, dtype=torch.bool, device=device)
    table_mask_plm = [ex.table_mask_plm + [0] * (max_len - len(ex.table_mask_plm)) for ex in ex_list]
    batch.table_mask_plm = torch.tensor(table_mask_plm, dtype=torch.bool, device=device)
    column_mask_plm = [ex.column_mask_plm + [0] * (max_len - len(ex.column_mask_plm)) for ex in ex_list]
    batch.column_mask_plm = torch.tensor(column_mask_plm, dtype=torch.bool, device=device)
    # subword aggregation
    question_subword_lens = [l for ex in ex_list for l in ex.question_subword_len]
    batch.question_subword_lens = torch.tensor(question_subword_lens, dtype=torch.long, device=device)
    table_subword_lens = [l for ex in ex_list for l in ex.table_subword_len]
    batch.table_subword_lens = torch.tensor(table_subword_lens, dtype=torch.long, device=device)
    column_subword_lens = [l for ex in ex_list for l in ex.column_subword_len]
    batch.column_subword_lens = torch.tensor(column_subword_lens, dtype=torch.long, device=device)
    return batch

def from_example_list_fixed(ex_list, device='cpu', train=True, **kwargs):
    """ New fields: batch.lens, batch.max_len, batch.relations, batch.relations_mask
    """
    batch = from_example_list_base(ex_list, device, train)
    batch.graph = Example.graph_factory.batch_graphs(ex_list, device, train=train, **kwargs)
    if train:
        batch.tgt_actions = [e.tgt_action for e in ex_list]
        batch.max_action_num = max([len(ex.tgt_action) for ex in ex_list])
    return batch


def from_example_list_random(ex_list, device='cpu', train=True, typed_random=True, untyped_random=True, **kwargs):
    batch = from_example_list_base(ex_list, device, train)
    batch.graph = Example.graph_factory.batch_graphs(ex_list, device, train=train, **kwargs)
    if train:
        batch.tgt_actions = [get_action_infos(Example.trans.get_field_action_pairs(ex.ast, typed_random, untyped_random)) for ex in ex_list]
        batch.max_action_num = max([len(ex.tgt_action) for ex in ex_list])
    return batch


def from_example_list_gtol(ex_list, device='cpu', train=True, **kwargs):
    batch = from_example_list_base(ex_list, device, train)
    batch.graph = Example.graph_factory.batch_graphs(ex_list, device, train=train, **kwargs)
    if train:
        batch.ids = [ex.id for ex in ex_list]
        batch.max_action_num = max([len(ex.tgt_action) for ex in ex_list])
    return batch

def from_example_list_rl(ex_list, device='cpu', train=True, **kwargs):
    batch = from_example_list_base(ex_list, device, train)
    batch.graph = Example.graph_factory.batch_graphs(ex_list, device, train=train, **kwargs)
    if train:
        batch.asts = [ex.ast for ex in ex_list]
    return batch

class Batch():

    def __init__(self, examples, device='cpu'):
        super(Batch, self).__init__()
        self.examples = examples
        self.device = device

    @classmethod
    def from_example_list(cls, ex_list, device='cpu', train=True, method='fixed', **kwargs):
        method_dict = {
            "fixed": from_example_list_fixed,
            "random": from_example_list_random,
            "gtol": from_example_list_gtol,
            "rl": from_example_list_rl
        }
        return method_dict[method](ex_list, device, train=train, **kwargs)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    @cached_property
    def max_question_len(self):
        return torch.max(self.question_lens).item()

    @cached_property
    def max_table_len(self):
        return torch.max(self.table_lens).item()

    @cached_property
    def max_column_len(self):
        return torch.max(self.column_lens).item()

    @cached_property
    def max_table_word_len(self):
        return torch.max(self.table_word_lens).item()

    @cached_property
    def max_column_word_len(self):
        return torch.max(self.column_word_lens).item()

    @cached_property
    def max_question_subword_len(self):
        return torch.max(self.question_subword_lens).item()

    @cached_property
    def max_table_subword_len(self):
        return torch.max(self.table_subword_lens).item()

    @cached_property
    def max_column_subword_len(self):
        return torch.max(self.column_subword_lens).item()

    """ Different types of nodes are seperated instead of concatenated together """
    @cached_property
    def mask(self):
        return torch.cat([self.question_mask, self.table_mask, self.column_mask], dim=1)

    @cached_property
    def question_mask(self):
        return lens2mask(self.question_lens)

    @cached_property
    def table_mask(self):
        return lens2mask(self.table_lens)

    @cached_property
    def column_mask(self):
        return lens2mask(self.column_lens)

    @cached_property
    def table_word_mask(self):
        return lens2mask(self.table_word_lens)

    @cached_property
    def column_word_mask(self):
        return lens2mask(self.column_word_lens)

    @cached_property
    def question_subword_mask(self):
        return lens2mask(self.question_subword_lens)

    @cached_property
    def table_subword_mask(self):
        return lens2mask(self.table_subword_lens)

    @cached_property
    def column_subword_mask(self):
        return lens2mask(self.column_subword_lens)

    def get_frontier_prod_idx(self, t):
        ids = []
        for e in self.examples:
            if t < len(e.tgt_action):
                ids.append(Example.grammar.prod2id[e.tgt_action[t].frontier_prod])
            else:
                ids.append(0)
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def get_frontier_field_idx(self, t):
        ids = []
        for e in self.examples:
            if t < len(e.tgt_action):
                ids.append(Example.grammar.field2id[e.tgt_action[t].frontier_field])
            else:
                ids.append(0)
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def get_frontier_type_idx(self, t):
        ids = []
        for e in self.examples:
            if t < len(e.tgt_action):
                ids.append(Example.grammar.type2id[e.tgt_action[t].frontier_field.type])
            else:
                ids.append(0)
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def get_parent_state_idx(self, t):
        ids = []
        for e in self.examples:
            if t < len(e.tgt_action):
                ids.append(e.tgt_action[t].parent_t)
            else:
                ids.append(0)
        return ids
