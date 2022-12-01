'''
Author: Li Wei, Zhu Luyao
Email: wei008@e.ntu.edu.sg
'''
import torch
import torch.nn as nn
import dgl
import copy
from dgl.nn.pytorch import RelGraphConv
from transformers import AlbertModel, AlbertConfig
from transformers import RobertaModel, RobertaConfig
from preRelAtt import RelAtt, Trans_RelAtt
import json
import warnings
warnings.filterwarnings("ignore")

torch.random.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.manual_seed(1234)


class Model(nn.Module):
    def __init__(self, cpt_ids, Configs, cuda_=True):
        super(Model, self).__init__()
        self.input_dim = Configs.input_dim
        self.output_dim = Configs.output_dim
        self.num_class = Configs.num_class
        self.num_relations = Configs.num_relations
        self.cpt_ids = cpt_ids
        if Configs.model_type == 'albert':
            self.bert_encoder = AlbertModel.from_pretrained('albert-base-v2')
        elif Configs.model_type == 'roberta':
            self.bert_encoder = RobertaModel.from_pretrained('roberta-base')
        elif Configs.model_type == 'roberta_large':
            # config_class = RobertaConfig
            # config = config_class.from_pretrained('roberta-large')
            self.bert_encoder = RobertaModel.from_pretrained('roberta-large')
            if Configs.freeze_bert:
                for param in self.bert_encoder.base_model.parameters():
                    param.requires_grad = False
            # the number is the original size of the tokenizer + 9 additional special tokens
            self.bert_encoder.resize_token_embeddings(50274)

        self.window = 2 * Configs.slide_win + 1
        self.slide_win = Configs.slide_win
        self.num_head = Configs.num_head
        self.num_bases = Configs.num_bases
        self.use_future = Configs.use_future_utt
        self.att_type = Configs.att_type
        self.cuda_ = cuda_
        # self.get_cpt_emb()
        self.fw = torch.nn.Linear(self.output_dim, self.input_dim)
        self.fc = torch.nn.Softmax(dim=1)

        self.fw_concept = torch.nn.Linear(self.input_dim, self.input_dim)

        self.conv1 = RelGraphConv(self.input_dim, self.input_dim, self.num_relations, regularizer='basis', num_bases=self.num_bases)
        self.conv2 = RelGraphConv(self.input_dim, self.input_dim, self.num_relations, regularizer='basis', num_bases=self.num_bases)

        if self.use_future:
            self.relAtt = RelAtt(1, 1, (self.window, self.input_dim), heads=self.num_head, dim_head=self.input_dim, dropout=Configs.att_dropout)
            
        else:
            self.relAtt = RelAtt(1, 1, (self.slide_win+1, self.input_dim), heads=self.num_head, dim_head=self.input_dim,
                             dropout=Configs.att_dropout)

        self.r = nn.Parameter(nn.init.uniform_(torch.zeros(3, self.input_dim)), requires_grad=True)
        self.num_feature = Configs.num_features
        self.fusion = nn.Linear(self.num_feature*self.input_dim, self.input_dim)
        self.fusion_2 = nn.Linear(self.input_dim, self.input_dim)
        self.linear = nn.Linear(self.input_dim, self.num_class)
        self.linear_2 = nn.Linear(self.input_dim, self.num_class)
        self.ac = nn.ReLU()
        self.ac_tanh = nn.Tanh()
        self.ac_sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(Configs.dropout)
        self.model_type = Configs.model_type
        self.chunk_size = Configs.chunk_size
        self.att_item = nn.Linear(3*self.input_dim + 1, 1)
        self.att_linear = nn.Linear(2*self.input_dim, 1)

        self.layer_norm = nn.LayerNorm(self.input_dim)
        self.use_layer_norm = Configs.use_layer_norm

        self.use_fixed = Configs.use_fixed
        if self.use_fixed:
            self.lamb = Configs.lamb
        else:
            self.lamb = nn.Linear(1, 1)

        word_embedding = torch.FloatTensor(json.load(open(Configs.glove_path+'glove_{}_{}.json'.format(4, Configs.dst_num_per_rel), 'r')))
        if Configs.freeze_glove:
            self.embedding = torch.nn.Embedding.from_pretrained(word_embedding, freeze=True)
        else:
            self.embedding = torch.nn.Embedding.from_pretrained(word_embedding, freeze=False)
        print('num_feature', self.num_feature)

        self.CoAtt = RelAtt(3, 1, (1, self.input_dim), heads=self.num_head, dim_head=self.input_dim // 2,
                            dropout=Configs.att_dropout)
        if self.num_feature==4:
            self.linear_out = nn.Linear(self.num_feature*self.input_dim, self.input_dim)
        elif self.num_feature ==3:
            self.linear_out = nn.Linear(self.input_dim, self.input_dim)

        self.rel_fun = Configs.rel_fun
        if self.rel_fun == 'vector':

            self.r = nn.Parameter(nn.init.xavier_normal_(torch.zeros(3, self.input_dim)), requires_grad=True)
        elif self.rel_fun == 'ones':
            self.r = nn.Parameter(torch.ones(3, self.input_dim), requires_grad=False)
        elif self.rel_fun == 'linear':

            self.r = nn.Parameter(torch.randn(3, self.input_dim,
                                              self.input_dim))  # nn.ParameterList([nn.Parameter(torch.randn(self.input_dim, self.input_dim)) for _ in range(3)])

    def forward(self, inputs, str_src, str_dst, str_edge_type, chunks, label, loss_func, train=True, eps=1e-8):
        
        if self.model_type == 'albert':
            out = self.bert_encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                    token_type_ids=inputs['token_type_ids'])
        elif self.model_type == 'roberta' or self.model_type == 'roberta_large':
            out = self.bert_encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

        dial_sel = out[0][inputs['input_ids'] >= 50265]
        len_dial = len(dial_sel)
        # out_ = self.fw(out[0][:, 0, :])
        out_ = self.fw(dial_sel)

        # relational graph neural network used to embed dialog structure knowledge
        # need to decide whether there is only one utterance in a dialog
        if out_.size(0) == 1:
            hidden_rgcn = torch.zeros(1, self.input_dim).to(out_.device)
        else:
            g = dgl.graph((str_src, str_dst))
            etype = str_edge_type
            # etype = torch.zeros_like(etype_)
            hidden = self.conv1(g, out_, etype)
            if self.use_layer_norm:
                hidden = torch.relu(self.layer_norm(hidden))
            else:
                hidden = torch.relu(hidden)
            hidden_rgcn = self.conv2(g, hidden, etype)

        # process concept
        output_ = []

        losses = 0

        for idx, chunk in enumerate(chunks):
            srcs_input_ids, srcs_token_type_ids, srcs_sel_mask, dsts_input_ids, dsts_token_type_ids, \
            dsts_sel_mask, weights, sentics, src_masks, masks, rels, utt_idx = chunk

            if masks.sum()==0:
                symbolic_repr = torch.zeros(masks.size(0), self.input_dim).to(out_.device)
            else:

                chunk_size, num_src, num_dst = weights.size()

                src_emb = self.get_cpt_emb([srcs_input_ids, srcs_token_type_ids,
                srcs_sel_mask], chunk_size, num_src)
                dst_emb = self.get_cpt_emb([dsts_input_ids, dsts_token_type_ids,
                dsts_sel_mask], chunk_size, num_src*num_dst)

                cpt_emb = self.symbolic_proc(out_[utt_idx], dst_emb,
                                             weights, sentics, src_masks, masks, chunk_size, num_src, num_dst)

                # integrate relation info into concept embedding
                if self.rel_fun in ['vector', 'ones', 'linear']:
                    r_vector = self.r[rels]
                    if self.rel_fun == 'linear':
                        # re_vector = r_vector(cpt_emb)
                        re_vector = torch.matmul(r_vector, cpt_emb.unsqueeze(-1)).squeeze(-1)
                    else:
                        re_vector = r_vector * cpt_emb # chunk_size, num_src, num_dst, self.input_dim
                    s_score = torch.sum(src_emb.unsqueeze(2) * re_vector, dim=-1)
                    s_score_masked = self.get_att_masked(s_score, masks)
                    alpha = torch.softmax(s_score_masked, dim=2) * src_masks.ne(0).unsqueeze(2)
                    src_emb = src_emb + torch.sum(alpha.unsqueeze(3) * re_vector, dim=2) # /(src_masks.ne(0).unsqueeze(2)+eps)
                else:

                    src_emb = src_emb + torch.sum(cpt_emb, dim=2)/(src_masks.unsqueeze(2)+eps)


                if self.att_type == 'dot_att':
                    dot_sum = torch.sum(src_emb *
                                        out_[utt_idx].unsqueeze(1), dim=-1)

                    src_mask = torch.sum(masks, dim=-1) > 0
                    att_score = torch.softmax(self.get_att_masked(dot_sum, src_mask), dim=-1) * src_masks.ne(0)
                    symbolic_repr = torch.sum(att_score.unsqueeze(2) * src_emb, dim=1)  # /sent_mask_sum


                elif self.att_type == 'linear_att':
                    att_feature = torch.cat((out_[utt_idx].unsqueeze(1).repeat(1, src_emb.size(1), 1), src_emb), dim=-1)
                    att_sum = self.att_linear(att_feature).squeeze(-1)
                    src_mask = torch.sum(masks, dim=-1) > 0
                    att_score = torch.softmax(self.get_att_masked(att_sum, src_mask), dim=-1) * src_masks.ne(0)
                    # sent_mask_sum = torch.sum(src_masks.sum(dim=-1).ne(0)) + eps
                    symbolic_repr = torch.sum(att_score.unsqueeze(2) * src_emb, dim=1) # / sent_mask_sum

                # use item attention to calculate the attention score between src_emb and relatt_out
                elif self.att_type == 'item_att':
                    item_att = self.item_att(out_[utt_idx].unsqueeze(1).repeat(1, src_emb.size(1), 1), src_emb)
                    item_sum = self.att_item(item_att).squeeze(-1)
                    src_mask = torch.sum(masks, dim=-1) > 0
                    att_score = torch.softmax(self.get_att_masked(item_sum, src_mask), dim=-1) * src_masks.ne(0)
                    # sent_mask_sum = torch.sum(src_masks.sum(dim=-1).ne(0)) + eps
                    symbolic_repr = torch.sum(att_score.unsqueeze(2) * src_emb, dim=1) # /sent_mask_sum

                else:
                    print("ValueError!")

            # feature fusion
            if self.num_feature == 3:

                feat_ = torch.stack([out_[utt_idx], hidden_rgcn[utt_idx], symbolic_repr], dim=1).unsqueeze(2)
                feat = self.CoAtt(feat_).squeeze(1).squeeze(1)
                output = torch.log_softmax(self.linear(self.ac_tanh(self.dropout(self.linear_out(feat)))), dim=1)
            
            else:              

                feat = out_[utt_idx] + hidden_rgcn[utt_idx] + symbolic_repr
                if self.use_layer_norm:
                    output = torch.log_softmax(self.linear_2(self.ac_tanh(self.layer_norm(self.dropout(self.fusion_2(feat))))), dim=1)
                else:
                    output = torch.log_softmax(
                        self.linear_2(self.ac_tanh(self.dropout(self.fusion_2(feat)))), dim=1)

            loss = loss_func(output, label[utt_idx]) / len_dial

            if train:
                if len(chunks) == idx + 1:
                    loss.backward()
                else:
                    loss.backward(retain_graph=True)
            output_.append(output.data)
            losses += loss.item()
            del symbolic_repr
            if masks.sum() != 0:
                del src_emb, cpt_emb, dst_emb
            torch.cuda.empty_cache()

        output_ = torch.cat(output_, dim=0)
        return output_, losses

    @staticmethod
    def item_att(x, y):
        try:
            item1 = torch.cat((x, y), dim=-1)
        except RuntimeError:
            print(x.size(), y.size())
        item2 = torch.norm(x - y, p=2, dim=-1, keepdim=True)
        item3 = torch.mul(x, y)
        delta = torch.cat((item1, item2, item3), dim=-1)

        return delta

    def get_cpt_emb(self, nodes_info, chunk_size, num_nd=4, seq_lim=5):

        input_ids = nodes_info[0].contiguous().view(-1, seq_lim)
        sel_mask = nodes_info[2].contiguous().view(-1, seq_lim)
        out = self.embedding(input_ids)
        # embs = out[0]
        # embs = self.fw_concept(out)

        sel_mask_sums = torch.sum(sel_mask.contiguous().view(chunk_size, num_nd, -1), dim=2, keepdim=True)

        sel_embs = torch.sum(sel_mask.contiguous().view(-1, seq_lim).unsqueeze(2) * out, dim=1).contiguous().view(chunk_size, num_nd,
                                                                                    self.input_dim) / sel_mask_sums
        sel_embs = self.fw_concept(sel_embs)
        return sel_embs  # bz, num_nd, self.input_dim

    # deprecated
    def get_cpt_emb_(self, nodes_info, chunk_size, num_nd=4, seq_lim=5):

        input_ids = nodes_info[0].contiguous().view(-1, seq_lim)
        attention_mask = nodes_info[2].contiguous().view(-1, seq_lim)
        sel_mask = nodes_info[3].contiguous().view(-1, seq_lim)
        if self.model_type == 'albert':
            token_type_ids = nodes_info[1].contiguous().view(-1, seq_lim)
        if self.model_type == 'albert':
            out = self.bert_encoder(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids)
        elif self.model_type in ['roberta', 'roberta_large']:
            out = self.bert_encoder(input_ids=input_ids,
                                    attention_mask=attention_mask)
        embs = out[0]
        embs = self.fw_concept(embs)

        sel_mask_sums = torch.sum(sel_mask.contiguous().view(chunk_size, num_nd, -1), dim=2, keepdim=True)

        sel_embs = torch.sum(sel_mask.contiguous().view(-1, seq_lim).unsqueeze(2) * embs, dim=1).contiguous().view(chunk_size, num_nd,
                                                                                    self.input_dim) / sel_mask_sums
        return sel_embs  # bz, num_nd, self.input_dim

    def symbolic_proc(self, relatt_out_chunk, dst_emb, weights, sentics,src_masks, masks, chunk_size, num_src, num_dst):
        cosine_sim = torch.abs(torch.cosine_similarity(relatt_out_chunk.unsqueeze(1).repeat(1, num_src * num_dst, 1),
                                                       dst_emb, dim=-1))
        relatedness = weights * cosine_sim.contiguous().view(chunk_size, num_src, num_dst)
        if self.use_fixed:
            omega = self.lamb * relatedness + (1 - self.lamb) * torch.abs(sentics)
        else:
            omega = self.lamb(relatedness.unsqueeze(-1)).squeeze(-1) +  torch.abs(sentics) - self.lamb(torch.abs(sentics).unsqueeze(-1)).squeeze(-1)
        omega = self.get_att_masked(omega, masks)
        alpha = (src_masks.unsqueeze(2) * torch.softmax(omega, dim=-1)).unsqueeze(2).repeat(1, 1, self.input_dim, 1).transpose(2, 3)
        cpt_emb = alpha * dst_emb.contiguous().view(chunk_size, num_src, num_dst, -1)
        return cpt_emb

    def get_att_masked(self, inp, mask):
        inp[mask == False] = float("-inf")
        return inp
