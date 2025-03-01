import logging
import os.path

import torch
from torch.nn import TransformerEncoderLayer
from transformers import RobertaModel, AutoModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
import faiss
import math
import numpy as np
import time
from Config import Config
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class Attention(torch.nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.attn_proj_q = torch.nn.Linear(hidden_size, hidden_size, bias=True)
        self.attn_q_dropout = torch.nn.Dropout(dropout)

        self.hidden_size = hidden_size

    def forward(self, q, k, v, attention_mask=None):
        q = self.attn_proj_q(q)
        q = self.attn_q_dropout(q)

        a = torch.bmm(q, torch.transpose(k, 1, 2))  # * (self.hidden_size ** -0.5)
        if attention_mask is not None:
            a -= 1e6 * (1 - attention_mask)
        a = torch.nn.functional.softmax(a, dim=-1)
        a = torch.bmm(a, v)

        return a


class IGRen(RobertaPreTrainedModel):
    def __init__(self, config, roberta: RobertaModel, model_args, data_args, op_list, const_list,
                 table_encoder: AutoModel = None):
        super().__init__(config)
        self.roberta = roberta
        self.table_encoder = table_encoder
        self.max_table_size = 0 if table_encoder is None else data_args.max_table_size

        self.generator_retrieval = model_args.generator_retrieval

        self.hidden_size = config.hidden_size
        self.max_seq_length = data_args.max_seq_length

        self.fp16 = model_args.fp16

        self.op_list = op_list
        self.const_list = const_list
        self.reserved_token_size = len(self.op_list) + len(self.const_list)

        self.operator_token_size = len(self.op_list)
        self.constant_token_size = len(self.const_list)

        # Parameters
        self.reserved_token_embeddings = torch.nn.Embedding(self.reserved_token_size, self.hidden_size)

        self.go_representation = torch.nn.Embedding(1, self.hidden_size)

        # layers

        if self.generator_retrieval:
            self.context_fusion_attn = Attention(hidden_size=self.hidden_size, dropout=model_args.dropout_rate)
            # self.generator_retrieval_predict_layer = torch.nn.Sequential(
            #     torch.nn.Linear(self.hidden_size * 2, self.hidden_size), torch.nn.Tanh(),
            #     torch.nn.Linear(self.hidden_size, 1))
            # self.question_fusion_attn = Attention(hidden_size=self.hidden_size, dropout=model_args.dropout_rate)
            # self.generator_retrieval_dual_attention_layer = DualAttentionLayer(hidden_size=self.hidden_size)

        self.seq_prj = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.seq_dropout = torch.nn.Dropout(model_args.dropout_rate)

        self.decoder_history_attn = Attention(hidden_size=self.hidden_size, dropout=model_args.dropout_rate)
        self.question_attn = Attention(hidden_size=self.hidden_size, dropout=model_args.dropout_rate)
        self.question_summary_attn = Attention(hidden_size=self.hidden_size, dropout=model_args.dropout_rate)

        self.document_attn = Attention(hidden_size=self.hidden_size, dropout=model_args.dropout_rate)
        self.retrieval_predict_layer = torch.nn.Sequential(torch.nn.Linear(self.hidden_size * 2, self.hidden_size),
                                                           torch.nn.Tanh(), torch.nn.Linear(self.hidden_size, 1))
        self.merge_prj = torch.nn.Sequential(torch.nn.Linear(self.hidden_size * 2, self.hidden_size), torch.nn.Tanh())
        self.merge_pdt = torch.nn.Linear(self.hidden_size, 1)

        self.input_embeddings_prj = torch.nn.Linear(self.hidden_size * 3, self.hidden_size, bias=True)
        self.input_embeddings_layernorm = torch.nn.LayerNorm([1, model_args.hidden_size])

        self.option_embeddings_prj = torch.nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

        self.decoder_step_proj = torch.nn.Linear(3 * self.hidden_size, self.hidden_size, bias=True)
        self.decoder_step_proj_dropout = torch.nn.Dropout(model_args.dropout_rate)

        all_tmp_list = self.op_list + self.const_list

        if self.use_knn:
            try:
                self.knn = KNN_Dstore(model_args)
            except:
                self.knn = None
        else:
            self.knn = None

        self.step_masks = torch.nn.Parameter(torch.zeros(data_args.max_step_index, self.reserved_token_size),
                                             requires_grad=False)
        for i in range(data_args.max_step_index):
            this_step_mask_ind = all_tmp_list.index("#" + str(i))
            self.step_masks[i, this_step_mask_ind] = 1.0

        self.rnn = torch.nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size,
                                 num_layers=model_args.decoder_layer_num, batch_first=True)
        self.predict_layer = torch.nn.Sequential(torch.nn.Linear(self.hidden_size, self.hidden_size), torch.nn.ReLU(),
                                                 torch.nn.Linear(self.hidden_size, 1))
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-1)

        # self.init_weights()

    def forward(self, ids, input_ids, attention_mask, token_type_ids, query_input_ids, query_attention_mask,
                query_token_type_ids, table_input_ids, table_attention_mask,
                table_token_type_ids,
                table_cell_spans, table_sizes, text_num, labels, program_ids, option_mask,
                program_mask, number_index, input_masks, question_mask):
        device = input_ids.device
        batch_size, seq_length = text_num.size(0), input_ids.size(-1)

        if self.config.model_type == 'roberta':
            token_type_ids = None
            query_token_type_ids = None

        assert text_num.max() == text_num.min()
        example_text_num = text_num[0]

        input_ids = torch.cat([query_input_ids.unsqueeze(1), input_ids.view(batch_size, example_text_num, seq_length)],
                              dim=1).view(batch_size * (example_text_num + 1), seq_length)
        attention_mask = torch.cat(
            [query_attention_mask.unsqueeze(1), attention_mask.view(batch_size, example_text_num, seq_length)],
            dim=1).view(
            batch_size * (example_text_num + 1), seq_length)
        if token_type_ids is not None:
            token_type_ids = torch.cat(
                [query_token_type_ids.unsqueeze(1), token_type_ids.view(batch_size, example_text_num, seq_length)],
                dim=1).view(
                batch_size * (example_text_num + 1), seq_length)

        text_representations = self.roberta(input_ids,
                                            attention_mask=attention_mask,
                                            token_type_ids=token_type_ids)['last_hidden_state']

        if self.table_encoder is not None:
            raw_table_representations = self.table_encoder(table_input_ids, attention_mask=table_attention_mask,
                                                           token_type_ids=table_token_type_ids)['last_hidden_state']
            table_representations = torch.zeros((batch_size, self.max_table_size, self.hidden_size), device=device)
            for batch_idx, table_repre in enumerate(raw_table_representations):
                rows, cols = table_sizes[batch_idx][0], table_sizes[batch_idx][1]
                for r in range(rows):
                    for c in range(cols):
                        linearized_index = r * cols + c
                        cell_token_span = table_cell_spans[batch_idx][linearized_index]
                        if cell_token_span[0] != -1:
                            table_representations[batch_idx][linearized_index] = torch.mean(
                                table_repre[cell_token_span[0]: cell_token_span[1] + 1], dim=0).view(self.hidden_size)

        else:
            table_representations = None

        bert_sequence_output = text_representations.view(batch_size, (example_text_num + 1) * seq_length,
                                                         self.hidden_size)

        split_program_ids = torch.split(program_ids, 1, dim=1)

        sequence_representations = self.seq_prj(bert_sequence_output)
        sequence_representations = self.seq_dropout(sequence_representations)

        op_embeddings = self.reserved_token_embeddings(torch.arange(0, self.reserved_token_size,
                                                                    device=device))  # (self.op_list_size + self.const_list_size) * hidden size
        op_embeddings = op_embeddings.repeat(batch_size, 1,
                                             1)  # batch size x (self.op_list_size + self.const_list_size) * hidden size

        all_logits = []

        init_decoder_output = self.go_representation(torch.tensor([0], device=device))
        decoder_output = init_decoder_output.repeat(batch_size, 1, 1)

        # [batch, op + table + seq len, hidden]
        initial_option_embeddings = torch.cat([op_embeddings, sequence_representations],
                                              dim=1) if self.table_encoder is None else \
            torch.cat([op_embeddings, table_representations, sequence_representations], dim=1)

        decoder_history = decoder_output

        decoder_state_h = torch.zeros(1, batch_size, self.hidden_size, device=device)
        decoder_state_c = torch.zeros(1, batch_size, self.hidden_size, device=device)

        all_token_representations = initial_option_embeddings

        float_input_mask = input_masks.view(batch_size, 1, sequence_representations.size(1))
        if self.generator_retrieval:
            program_context_representations = [[decoder_output[i]] for i in range(batch_size)]

        for cur_step in range(program_ids.size(-1)):

            decoder_history_ctx_embeddings = self.decoder_history_attn(q=decoder_output, k=decoder_history,
                                                                       v=decoder_history)
            question_ctx_embeddings = self.question_attn(q=decoder_output, k=sequence_representations,
                                                         v=sequence_representations)  # , attention_mask=float_input_mask)
            question_summary_embeddings = self.question_summary_attn(q=decoder_output, k=sequence_representations,
                                                                     v=sequence_representations)  # , attention_mask=float_input_mask)

            concat_input_embeddings = torch.cat(
                [decoder_history_ctx_embeddings, question_ctx_embeddings, decoder_output], dim=-1)
            input_embeddings = self.input_embeddings_prj(concat_input_embeddings)
            input_embeddings = self.input_embeddings_layernorm(input_embeddings)

            question_option_vec = all_token_representations * question_summary_embeddings
            this_step_all_token_representations = torch.cat([all_token_representations, question_option_vec], dim=-1)
            this_step_all_token_representations = self.option_embeddings_prj(
                this_step_all_token_representations)  # batch size x (self.op_list_size + self.const_list_size + sequence length) * hidden size
            all_token_logits = torch.matmul(this_step_all_token_representations,
                                            torch.transpose(input_embeddings, 1, 2))
            all_token_logits = torch.squeeze(all_token_logits, dim=2)  # [batch, op + seq_len

            generator_retrieval_scores = self.get_facts_weights(
                batch_size,
                decoder_output,
                example_text_num,
                program_context_representations,
                seq_length,
                this_step_all_token_representations)

            all_token_logits = all_token_logits * generator_retrieval_scores

            all_token_logits -= 1e6 * (1 - option_mask)
            all_logits.append(all_token_logits)

            if self.training:
                program_index = torch.unsqueeze(split_program_ids[cur_step], dim=1)  # teacher forcing
            else:
                # constrain decoding
                if cur_step % 4 == 0:
                    program_index = torch.argmax(all_token_logits[:, :self.operator_token_size], axis=-1, keepdim=True)
                elif (cur_step + 1) % 4 == 0:  # ')' round
                    program_index = torch.full((batch_size, 1), self.op_list.index(')')).to(device)
                else:
                    program_index = torch.argmax(all_token_logits[:, self.operator_token_size:], axis=-1,
                                                 keepdim=True) + self.operator_token_size

                program_index = torch.unsqueeze(program_index, dim=1)

            program_select_index = program_index.view(batch_size)
            for i in range(batch_size):
                program_context_representations[i].append(
                    this_step_all_token_representations[i, program_select_index[i]].unsqueeze(0))

            if (cur_step + 1) % 4 == 0:
                # update op embeddings
                this_step_index = cur_step // 4
                this_step_mask = self.step_masks[this_step_index, :]
                this_step_mask = torch.nn.ConstantPad1d((0, initial_option_embeddings.size(1) - this_step_mask.size(0)),
                                                        0)(this_step_mask)

                decoder_step_vec = self.decoder_step_proj(concat_input_embeddings)
                decoder_step_vec = self.decoder_step_proj_dropout(decoder_step_vec)
                decoder_step_vec = torch.squeeze(decoder_step_vec, dim=1)

                this_step_new_emb = decoder_step_vec  # [batch, hidden]

                this_step_new_emb = torch.unsqueeze(this_step_new_emb, 1)
                this_step_new_emb = this_step_new_emb.repeat(1,
                                                             self.reserved_token_size + self.max_table_size + seq_length * (
                                                                     example_text_num + 1),
                                                             1)  # [batch, op table seq, hidden]

                this_step_mask = torch.unsqueeze(this_step_mask, 0)  # [1, op seq]
                # print(this_step_mask)

                this_step_mask = torch.unsqueeze(this_step_mask, 2)  # [1, op seq, 1]
                this_step_mask = this_step_mask.repeat(batch_size, 1, self.hidden_size)  # [batch, op seq, hidden]
                all_token_representations = torch.where(this_step_mask > 0, this_step_new_emb,
                                                        initial_option_embeddings.half() if self.fp16 else initial_option_embeddings)  # if self.training else initial_option_embeddings)

            # print(program_index.size())
            program_index = torch.repeat_interleave(program_index, self.hidden_size, dim=2)  # [batch, 1, hidden]

            input_program_embeddings = torch.gather(this_step_all_token_representations, dim=1, index=program_index)

            decoder_output, (decoder_state_h, decoder_state_c) = self.rnn(input_program_embeddings,
                                                                          (decoder_state_h, decoder_state_c))
            decoder_history = torch.cat([decoder_history, input_program_embeddings], dim=1)

            if self.knn is not None:
                try:
                    rela_know = self.knn.get_knn_log_prob(decoder_history)
                    decoder_output = 0.7 * decoder_output + 0.3 * rela_know
                except:
                    print("error")

            if self.save_knn:
                with open("key.npc", "w+") as f:
                    f.write(decoder_history)
                with open("value.npc", "w+") as f:
                    f.write(decoder_output)

        all_logits = torch.stack(all_logits, dim=1)
        loss = self.criterion(all_logits.view(-1, all_logits.size(-1)), program_ids.view(-1))
        loss = loss * program_mask.view(-1)
        loss = loss.sum() / program_mask.sum()

        if self.training:
            return loss,

        assert all_logits.size(-1) <= 1500
        assert number_index.size(-1) <= 256 + self.max_table_size
        all_step_scores = torch.nn.ConstantPad1d((0, 1500 - all_logits.size(-1)), -1e4)(all_logits)
        number_index = torch.nn.ConstantPad1d((0, 256 + self.max_table_size - number_index.size(-1)), 0)(number_index)
        return loss, all_step_scores, ids, number_index

    def get_facts_weights(self, batch_size, decoder_output, example_text_num,
                          program_context_representations, seq_length, this_step_all_token_representations):

        _program_context_representations = torch.stack([torch.cat(p, dim=0) for p in program_context_representations],
                                                       dim=0)
        _program_context_representations = self.context_fusion_attn(q=decoder_output,
                                                                    k=_program_context_representations,
                                                                    v=_program_context_representations)

        facts = this_step_all_token_representations[:, self.reserved_token_size + self.max_table_size:].view(batch_size,
                                                                                                             example_text_num + 1,
                                                                                                             seq_length,
                                                                                                             self.hidden_size)
        facts = facts.mean(-2)
        # bs * num * h
        merge_facts = self.merge_prj(
            torch.cat([_program_context_representations.expand(-1, example_text_num + 1, -1), facts], dim=-1))

        logits = merge_facts * decoder_output
        generator_retrieval_scores = self.merge_pdt(logits)
        generator_retrieval_scores = torch.softmax(generator_retrieval_scores, dim=1)

        generator_retrieval_scores_expand = torch.nn.ConstantPad1d((self.reserved_token_size + self.max_table_size, 0),
                                                                   1.0)(
            generator_retrieval_scores.expand(-1, -1, seq_length).reshape(batch_size,
                                                                          (example_text_num + 1) * seq_length))

        return generator_retrieval_scores_expand


class KNN_Dstore(object):
    def __init__(self, args):
        self.half = args.fp16
        self.dimension = args.decoder_embed_dim
        self.k = args.k
        self.dstore_size = args.dstore_size
        self.metric_type = args.faiss_metric_type
        self.sim_func = args.knn_sim_func
        self.dstore_fp16 = args.dstore_fp16
        self.index = self.setup_faiss(args)

    def setup_faiss(self, args):
        if not args.dstore_filename:
            raise ValueError('Cannot build a datastore without the data.')

        start = time.time()
        index = faiss.read_index(args.indexfile, faiss.IO_FLAG_ONDISK_SAME_DIR)
        print('Reading datastore took {} s'.format(time.time() - start))
        index.nprobe = args.probe

        if args.dstore_fp16:
            print('Keys are fp16 and vals are int16')
            if not args.no_load_keys:
                self.keys = np.memmap(args.dstore_filename + '_keys.npy', dtype=np.float16, mode='r',
                                      shape=(self.dstore_size, self.dimension))
            self.vals = np.memmap(args.dstore_filename + '_vals.npy', dtype=np.int16, mode='r',
                                  shape=(self.dstore_size, 1))
        else:
            print('Keys are fp32 and vals are int64')
            if not args.no_load_keys:
                self.keys = np.memmap(args.dstore_filename + '_keys.npy', dtype=np.float32, mode='r',
                                      shape=(self.dstore_size, self.dimension))
            self.vals = np.memmap(args.dstore_filename + '_vals.npy', dtype=np.int, mode='r',
                                  shape=(self.dstore_size, 1))

        # If you wish to load all the keys into memory
        # CAUTION: Only do this if your RAM can handle it!
        if args.move_dstore_to_mem:
            print('Loading to memory...')
            start = time.time()

            if not args.no_load_keys:
                del self.keys
                self.keys_from_memmap = np.memmap(args.dstore_filename + '_keys.npy', dtype=np.float32, mode='r',
                                                  shape=(self.dstore_size, self.dimension))
                self.keys = np.zeros((self.dstore_size, self.dimension),
                                     dtype=np.float16 if args.dstore_fp16 else np.float32)
                self.keys = self.keys_from_memmap[:]
                self.keys = self.keys.astype(np.float16 if args.dstore_fp16 else np.float32)

            del self.vals
            self.vals_from_memmap = np.memmap(args.dstore_filename + '_vals.npy', dtype=np.int, mode='r',
                                              shape=(self.dstore_size, 1))
            self.vals = np.zeros((self.dstore_size, 1), dtype=np.int16 if args.dstore_fp16 else np.int)
            self.vals = self.vals_from_memmap[:]
            self.vals = self.vals.astype(np.int16 if args.dstore_fp16 else np.int)
            print('Loading to memory took {} s'.format(time.time() - start))

        return index

    def get_knns(self, queries):
        start = time.time()
        dists, knns = self.index.search(queries.detach().cpu().float().numpy(), self.k)
        return dists, knns

    def get_knn_log_prob(self, queries):
        def dist_func(d, k, q, function=None):
            if not function:
                # Default behavior for L2 metric is to recompute distances.
                # Default behavior for IP metric is to return faiss distances.
                qsize = q.shape
                if self.metric_type == 'l2':
                    start = time.time()
                    knns_vecs = torch.from_numpy(self.keys[k]).cuda().view(qsize[0], self.k, -1)
                    if self.half:
                        knns_vecs = knns_vecs.half()
                    query_vecs = q.view(qsize[0], 1, qsize[1]).repeat(1, self.k, 1)
                    l2 = torch.sum((query_vecs - knns_vecs.detach()) ** 2, dim=2)
                    return -1 * l2
                return d

            if function == 'dot':
                qsize = q.shape
                return (torch.from_numpy(self.keys[k]).cuda() * q.view(qsize[0], 1, qsize[1])).sum(dim=-1)

            if function == 'do_not_recomp_l2':
                return -1 * d

            raise ValueError("Invalid knn similarity function!")

        # queries  are BxLxH
        # reshape: (BxL)xH
        qshape = queries.shape
        queries = queries.view(-1, qshape[-1])
        dists, knns = self.get_knns(queries)
        # (T_reducedxB)xK
        dists = torch.from_numpy(dists).cuda()
        start = time.time()
        dists = dist_func(dists, knns, queries, function=self.sim_func)
        probs = self.log_softmax(dists, dim=-1)

        # (T_reducedxB)
        yhat_knn_prob = torch.logsumexp(probs, dim=-1).clone()
        full_yhat_knn_prob = torch.full([qshape[0] * qshape[1]], -10000).cuda()
        full_yhat_knn_prob = yhat_knn_prob

        # TxBx1
        return full_yhat_knn_prob.view(qshape[0], qshape[1], 1)

    def log_softmax(x, dim, onnx_trace=False):
        if onnx_trace:
            return F.log_softmax(x.float(), dim=dim)
        else:
            return F.log_softmax(x, dim=dim, dtype=torch.float32)
