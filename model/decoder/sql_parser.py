#coding=utf-8
from __future__ import print_function
import torch, math
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter
from model.decoder.beam import Beam
from model.decoder.training_beam import TrainingBeam
from model.decoder.onlstm import LSTM, ONLSTM
from model.model_utils import Registrable, MultiHeadAttention, TiedLinearClassifier, lens2mask
from asdl.transition_system import ApplyRuleAction, SelectColumnAction, SelectTableAction, SelectValueAction, TransitionSystem

@Registrable.register('decoder')
class StructDecoder(nn.Module):

    def __init__(self, args, transition_system: TransitionSystem):
        super(StructDecoder, self).__init__()
        self.args = args
        self.transition_system = transition_system
        self.grammar = self.transition_system.grammar

        # embedding table for ASDL productions
        self.production_embed = nn.Embedding(len(transition_system.grammar), args.action_embed_size)
        # embedding table for ASDL fields in constructors
        self.field_embed = nn.Embedding(len(transition_system.grammar.fields), args.field_embed_size)
        # embedding table for ASDL types
        self.type_embed = nn.Embedding(len(transition_system.grammar.types), args.type_embed_size)

        # input of decoder lstm: [prev_action, parent_production, parent_field, parent_type, parent_state[, prev_context]]
        input_dim = args.action_embed_size * 2 + args.field_embed_size + args.type_embed_size + args.lstm_hidden_size
        # optional, context feeding
        input_dim += args.gnn_hidden_size * int(args.context_feeding)
        cell_constructor = ONLSTM if args.lstm == 'onlstm' else LSTM
        self.decoder_lstm = cell_constructor(input_dim, args.lstm_hidden_size, num_layers=args.lstm_num_layers,
            chunk_num=args.chunk_size, dropout=args.dropout, dropconnect=args.drop_connect)

        # transform column/table/value embeddings to action embedding space
        self.table_lstm_input = nn.Linear(args.gnn_hidden_size, args.action_embed_size)
        self.value_lstm_input = self.column_lstm_input = self.table_lstm_input

        self.ext_feat_dim = 0 if not args.struct_feeding else args.field_embed_size + args.type_embed_size
        self.context_attn = MultiHeadAttention(args.gnn_hidden_size, args.lstm_hidden_size + self.ext_feat_dim, args.gnn_hidden_size, args.gnn_hidden_size,
            num_heads=args.num_heads, feat_drop=args.dropout)

        # feature vector before ApplyRule or SelectColumn/Table/Value
        self.att_vec_linear = nn.Sequential(nn.Linear(args.lstm_hidden_size + args.gnn_hidden_size, args.att_vec_size), nn.Tanh())
        self.reserved_size = SelectValueAction.size(args.dataset)
        self.reserved_value_memory = Parameter(torch.zeros((1, self.reserved_size, args.gnn_hidden_size), dtype=torch.float))
        nn.init.kaiming_uniform_(self.reserved_value_memory, a=math.sqrt(5))

        # use parent field+type embedding
        self.apply_rule = TiedLinearClassifier(args.att_vec_size + self.ext_feat_dim, args.action_embed_size, bias=False)
        self.select_table = MultiHeadAttention(args.gnn_hidden_size, args.att_vec_size + self.ext_feat_dim, args.gnn_hidden_size, args.gnn_hidden_size,
            num_heads=args.num_heads, feat_drop=args.dropout)
        self.select_column = self.select_value = self.select_table


    def score(self, encodings, value_memory, h0, batch):
        """ Training function
            @input:
                encodings: encoded representations and mask matrix from encoder, bsize x seqlen x gnn_hidden_size
                value_memory: extracted values, bsize x max_value_num x gnn_hidden_size
                batch: see utils.batch, we use fields
                    batch.mask, batch.graph.value_nums, batch.tgt_actions, batch.get_frontier_prod_idx(t),
                    batch.get_frontier_field_idx(t), batch.get_frontier_field_type_idx(t),
                    batch.max_action_num
            output:
                loss: sum of loss for each training batch
        """
        args = self.args
        mask = batch.mask
        padding_action_embed = encodings.new_zeros(args.action_embed_size)
        split_len = [batch.max_question_len, batch.max_table_len, batch.max_column_len]
        _, tab, col = encodings.split(split_len, dim=1)
        _, tab_mask, col_mask = mask.split(split_len, dim=1)
        value_memory = torch.cat([self.reserved_value_memory.expand(len(batch), -1, -1), value_memory], dim=1)
        val_mask = lens2mask(batch.graph.value_nums + self.reserved_size)

        # h0: num_layers, bsize, lstm_hidden_size
        h0 = h0.unsqueeze(0).repeat(args.lstm_num_layers, 1, 1)
        h_c = (h0, h0.new_zeros(h0.size()))
        action_probs = [[] for _ in range(encodings.size(0))]
        history_states = []
        for t in range(batch.max_action_num):
            # x: [prev_action, parent_production, parent_field, parent_type, parent_state[, prev_context]]
            if t == 0:
                x = encodings.new_zeros(encodings.size(0), self.decoder_lstm.input_size)
                offset = args.action_embed_size * 2 + args.field_embed_size
                root_type = torch.tensor([self.grammar.type2id[self.grammar.root_type]] * x.size(0), dtype=torch.long, device=x.device)
                x[:, offset: offset + args.type_embed_size] = self.type_embed(root_type)
                offset += args.type_embed_size
                x[:, offset: offset + args.lstm_hidden_size] = h0[-1]
            else:
                prev_action_embed = []
                for e_id, tgt_action in enumerate(batch.tgt_actions):
                    if t < len(tgt_action):
                        prev_action = tgt_action[t - 1].action
                        if isinstance(prev_action, ApplyRuleAction):
                            prev_action_embed.append(self.production_embed.weight[self.grammar.prod2id[prev_action.production]])
                        elif isinstance(prev_action, SelectTableAction):
                            tab_embed = self.table_lstm_input(tab[e_id, prev_action.table_id])
                            prev_action_embed.append(tab_embed)
                        elif isinstance(prev_action, SelectColumnAction):
                            col_embed = self.column_lstm_input(col[e_id, prev_action.column_id])
                            prev_action_embed.append(col_embed)
                        elif isinstance(prev_action, SelectValueAction):
                            val_embed = self.value_lstm_input(value_memory[e_id, prev_action.value_id])
                            prev_action_embed.append(val_embed)
                        else:
                            raise ValueError('Unrecognized previous action object!')
                    else:
                        prev_action_embed.append(padding_action_embed)
                inputs = [torch.stack(prev_action_embed)]
                inputs.append(self.production_embed(batch.get_frontier_prod_idx(t))) # parent action embed
                inputs.append(self.field_embed(batch.get_frontier_field_idx(t))) # parent field embed
                inputs.append(self.type_embed(batch.get_frontier_type_idx(t))) # parent type embed
                parent_state_ids = batch.get_parent_state_idx(t) # parent hidden state timestep
                parent_state = torch.stack([history_states[t][e_id] for e_id, t in enumerate(parent_state_ids)])
                inputs.append(parent_state)
                if args.context_feeding:
                    inputs.append(context)
                x = torch.cat(inputs, dim=-1)

            # advance decoder lstm and attention calculation
            out, h_c = self.decoder_lstm(x.unsqueeze(1), h_c, start=(t==0))
            out = out.squeeze(1)
            if args.struct_feeding:
                ext_feats = x[:, args.action_embed_size * 2: args.action_embed_size * 2 + self.ext_feat_dim]
                query = torch.cat([out, ext_feats], dim=-1)
            else: query = out
            context, _ = self.context_attn(encodings, query, mask)
            att_vec = self.att_vec_linear(torch.cat([out, context], dim=-1))

            # action logprobs
            if args.struct_feeding:
                att_vec = torch.cat([att_vec, ext_feats], dim=-1)
            apply_rule_logprob = self.apply_rule(att_vec, self.production_embed.weight) # bsize x prod_num
            _, select_tab_prob = self.select_table(tab, att_vec, tab_mask)
            select_tab_logprob = torch.log(select_tab_prob + 1e-32)
            _, select_col_prob = self.select_column(col, att_vec, col_mask)
            select_col_logprob = torch.log(select_col_prob + 1e-32)
            _, select_val_prob = self.select_value(value_memory, att_vec, val_mask)
            select_val_prob = torch.log(select_val_prob + 1e-32)

            for e_id, tgt_action in enumerate(batch.tgt_actions):
                if t < len(tgt_action):
                    action_t = tgt_action[t].action
                    if isinstance(action_t, ApplyRuleAction):
                        logprob_t = apply_rule_logprob[e_id, self.grammar.prod2id[action_t.production]]
                    elif isinstance(action_t, SelectTableAction):
                        logprob_t = select_tab_logprob[e_id, action_t.table_id]
                    elif isinstance(action_t, SelectColumnAction):
                        logprob_t = select_col_logprob[e_id, action_t.column_id]
                    elif isinstance(action_t, SelectValueAction):
                        logprob_t = select_val_prob[e_id, action_t.value_id]
                    else:
                        raise ValueError('Unrecognized action object!')
                    action_probs[e_id].append(logprob_t)

            history_states.append(h_c[0][-1])

        # loss is negative sum of all the action probabilities
        loss = - torch.stack([torch.stack(logprob_i).sum() for logprob_i in action_probs]).sum()
        return loss


    def parse(self, encodings, value_memory, h0, batch, beam_size=5,
            gtl_training=True, ts_order='controller', uts_order='enum', n_best=1, cumulate_method='sum'):
        """ Parse all examples together with beam search
            encodings: bs x (max_q + max_t + max_c) x gnn_hs
            value_memory: bs x max_v_num x gnn_hs
            h0: bs x gnn_hs
        """
        args, device = self.args, encodings.device
        num_examples, mask = len(batch), batch.mask
        split_len = [batch.max_question_len, batch.max_table_len, batch.max_column_len]
        _, table_memory, column_memory = encodings.split(split_len, dim=1)
        _, table_mask, column_mask = mask.split(split_len, dim=1)
        value_memory = torch.cat([self.reserved_value_memory.repeat(num_examples, 1, 1), value_memory], dim=1)
        value_nums = batch.graph.value_nums + self.reserved_size
        value_mask = lens2mask(value_nums, max_len=value_nums.max())
        table2action, column2action, value2action = self.table_lstm_input(table_memory), self.column_lstm_input(column_memory), self.value_lstm_input(value_memory)
        h0 = h0.unsqueeze(0).repeat(args.lstm_num_layers, 1, 1)
        h_c = (h0, h0.new_zeros(h0.size()))
        # prepare data structure to record each sample predictions
        active_idx_mapping, hyp_states = list(range(num_examples)), h0.new_zeros(num_examples, 0, h0.size(-1))
        prev_ids = torch.arange(num_examples, dtype=torch.long, device=device)
        if gtl_training:
            beams = [TrainingBeam(batch[idx].ast, self.transition_system, ts_order=ts_order, uts_order=uts_order,
                beam_size=beam_size, device=device) for idx in range(num_examples)]
            max_action_num = batch.max_action_num
        else:
            table_nums, column_nums, value_nums = batch.table_lens.tolist(), batch.column_lens.tolist(), value_nums.tolist()
            beams = [Beam((table_nums[idx], column_nums[idx], value_nums[idx]), self.transition_system,
                beam_size=beam_size, ts_order=ts_order, device=device) for idx in range(num_examples)]
            max_action_num = args.decode_max_step

        for t in range(max_action_num):
            # prepare structural input: num_hyp x (prev_action + parent_action + parent_field + parent_type + parent_hidden[ + parent_cxt])
            if t == 0: # all initialize to 0 except parent type and parent hidden states
                inputs = encodings.new_zeros(num_examples, self.decoder_lstm.input_size)
                offset = args.action_embed_size * 2 + args.field_embed_size
                root_type = torch.tensor([self.grammar.type2id[self.grammar.root_type]] * num_examples, dtype=torch.long, device=device)
                inputs[:, offset: offset + args.type_embed_size] = self.type_embed(root_type)
                offset += args.type_embed_size
                inputs[:, offset: offset + args.lstm_hidden_size] = h0[-1]

                cur_encodings, cur_table_memory, cur_column_memory, cur_value_memory = encodings, table_memory, column_memory, value_memory
                cur_mask, cur_table_mask, cur_column_mask, cur_value_mask = mask, table_mask, column_mask, value_mask
                num_hyps = [1] * num_examples
            else:
                # previous action embeddings
                prev_action_embeds = []
                prev_actions = [beams[bid].get_previous_actions() for bid in active_idx_mapping]
                for bid, actions in zip(active_idx_mapping, prev_actions):
                    for action in actions:
                        if isinstance(action, ApplyRuleAction):
                            prev_action_embeds.append(self.production_embed.weight[self.grammar.prod2id[action.production]])
                        elif isinstance(action, SelectTableAction):
                            prev_action_embeds.append(table2action[bid, action.table_id])
                        elif isinstance(action, SelectColumnAction):
                            prev_action_embeds.append(column2action[bid, action.column_id])
                        elif isinstance(action, SelectValueAction):
                            prev_action_embeds.append(value2action[bid, action.value_id])
                        else: raise ValueError('Unrecognized action type!')
                prev_action_embeds = torch.stack(prev_action_embeds, dim=0)
                # parent production/field/type embeddings
                prod_ids = torch.cat([beams[bid].get_parent_prod_ids() for bid in active_idx_mapping])
                prod_embeds = self.production_embed(prod_ids)
                field_ids = torch.cat([beams[bid].get_parent_field_ids() for bid in active_idx_mapping])
                field_embeds = self.field_embed(field_ids)
                type_ids = torch.cat([beams[bid].get_parent_type_ids() for bid in active_idx_mapping])
                type_embeds = self.type_embed(type_ids)
                # parent hidden states
                parent_ts = torch.cat([beams[bid].get_parent_timesteps() for bid in active_idx_mapping])
                parent_states = torch.gather(hyp_states, 1, parent_ts.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, hyp_states.size(-1))).squeeze(1)
                inputs = [prev_action_embeds, prod_embeds, field_embeds, type_embeds, parent_states]
                if args.context_feeding:
                    inputs.append(context)
                inputs = torch.cat(inputs, dim=-1)

                select_index = [bid for bid in active_idx_mapping for _ in range(len(beams[bid].hyps))]
                cur_encodings, cur_table_memory, cur_column_memory, cur_value_memory = encodings[select_index], \
                    table_memory[select_index], column_memory[select_index], value_memory[select_index]
                cur_mask, cur_table_mask, cur_column_mask, cur_value_mask = mask[select_index], table_mask[select_index], \
                    column_mask[select_index], value_mask[select_index]
                num_hyps = [len(beams[bid].hyps) for bid in active_idx_mapping]

            # forward and calculate log_scores for each action
            out, (h_t, c_t) = self.decoder_lstm(inputs.unsqueeze(1), h_c, start=(t==0), prev_ids=prev_ids)
            out = out.squeeze(1)
            if args.struct_feeding:
                ext_feats = inputs[:, args.action_embed_size * 2: args.action_embed_size * 2 + self.ext_feat_dim]
                query = torch.cat([out, ext_feats], dim=-1)
            else: query = out
            context, _ = self.context_attn(cur_encodings, query, cur_mask)
            att_vec = self.att_vec_linear(torch.cat([out, context], dim=-1))
            if args.struct_feeding:
                att_vec = torch.cat([att_vec, ext_feats], dim=-1)
            apply_rule_logprob = self.apply_rule(att_vec, self.production_embed.weight)
            ar_score = torch.split(apply_rule_logprob, num_hyps)
            _, select_table_prob = self.select_table(cur_table_memory, att_vec, cur_table_mask)
            st_score = torch.split(torch.log(select_table_prob + 1e-32), num_hyps)
            _, select_column_prob = self.select_column(cur_column_memory, att_vec, cur_column_mask)
            sc_score = torch.split(torch.log(select_column_prob + 1e-32), num_hyps)
            _, select_value_prob = self.select_value(cur_value_memory, att_vec, cur_value_mask)
            sv_score = torch.split(torch.log(select_value_prob + 1e-32), num_hyps)

            # rank and select based on AST type constraints
            active, cum_num_hyps, live_hyp_ids = [], np.cumsum([0] + num_hyps), []
            for idx, bid in enumerate(active_idx_mapping):
                beams[bid].advance(ar_score[idx], st_score[idx], sc_score[idx], sv_score[idx])
                if not beams[bid].done:
                    active.append(bid)
                    live_hyp_ids.extend(beams[bid].get_previous_hyp_ids(cum_num_hyps[idx]))

            if not active: # all beams are finished
                break

            # update each unfinished beam and record active h_c, hidden_states and context vector
            active_idx_mapping = active
            h_c = (h_t[:, live_hyp_ids], c_t[:, live_hyp_ids])
            hyp_states = torch.cat([hyp_states[live_hyp_ids], h_c[0][-1].unsqueeze(1)], dim=1)
            prev_ids = prev_ids[live_hyp_ids]
            if args.context_feeding:
                context = context[live_hyp_ids]

        completed_hyps = [b.sort_finished() for b in beams]
        if gtl_training:
            cum_func = torch.sum if cumulate_method == 'sum' else torch.mean
            loss = - torch.sum(torch.stack([cum_func(torch.stack([hyp.score for hyp in b][:n_best])) for b in completed_hyps]))
            return loss, completed_hyps
        else:
            return completed_hyps
