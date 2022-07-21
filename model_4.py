'''
Author: Li Wei, Zhu Luyao
Email: wei008@e.ntu.edu.sg
'''
import torch
import torch.nn as nn
import dgl
import copy
from dgl.nn.pytorch import RelGraphConv
from transformers import AlbertModel
from transformers import RobertaModel
from preRelAtt import RelAtt, Trans_RelAtt
from dataloader_3 import get_chunk

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
            self.bert_encoder = RobertaModel.from_pretrained('roberta-large')
        self.window = 2 * Configs.slide_win + 1
        self.slide_win = Configs.slide_win
        self.lamb = Configs.lamb
        self.num_head = Configs.num_head
        self.num_bases = Configs.num_bases
        self.use_future = Configs.use_future_utt
        self.use_dot_att = Configs.use_dot_att
        self.cuda_ = cuda_
        # self.get_cpt_emb()
        self.fw = torch.nn.Linear(self.output_dim, self.input_dim)
        self.fc = torch.nn.Softmax(dim=1)

        self.fw_concept = torch.nn.Linear(self.output_dim, self.input_dim)

        self.conv1 = RelGraphConv(self.input_dim, self.input_dim, self.num_relations, regularizer='basis', num_bases=self.num_bases)
        self.conv2 = RelGraphConv(self.input_dim, self.input_dim, self.num_relations, regularizer='basis', num_bases=self.num_bases)

        if self.use_future:
            self.relAtt = RelAtt(1, 1, (self.window, self.input_dim), heads=self.num_head, dim_head=self.input_dim // 2, dropout=Configs.att_dropout)
        else:
            self.relAtt = RelAtt(1, 1, (self.slide_win+1, self.input_dim), heads=self.num_head, dim_head=self.input_dim // 2,
                             dropout=Configs.att_dropout)
        # self.relAtt = Trans_RelAtt(1, 1, (self.window, self.input_dim), heads=self.num_head, dim_head=self.input_dim // 2,
        #                      dropout=Configs.att_dropout)

        self.r = nn.Parameter(nn.init.uniform_(torch.zeros(3, self.input_dim)), requires_grad=True)
        self.num_feature = Configs.num_features
        self.fusion = nn.Linear(self.num_feature*self.input_dim, self.input_dim)
        self.linear = nn.Linear(self.input_dim, self.num_class)
        self.ac = nn.ReLU()
        self.ac_tanh = nn.Tanh()
        self.dropout = nn.Dropout(Configs.dropout)
        self.model_type = Configs.model_type
        self.chunk_size = Configs.chunk_size
        self.linear_att = nn.Linear(3*self.input_dim + 1, 1)
        self.att_linear = nn.Linear(2*self.input_dim, 1)
        print('num_feature', self.num_feature)

    def forward(self, inputs, str_src, str_dst, str_edge_type, chunks, label, loss_func, train=True):
        # torch.autograd.set_detect_anomaly(True)

        len_dial = len(inputs['input_ids'])
        if self.model_type == 'albert':
            out = self.bert_encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                    token_type_ids=inputs['token_type_ids'])
        elif self.model_type == 'roberta' or self.model_type == 'roberta_large':
            out = self.bert_encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        out_ = self.fw(out[0][:, 0, :])

        # relational graph neural network used to embed dialog structure knowledge
        # need to decide whether there is only one utterance in a dialog
        if out_.size(0) == 1:
            hidden_rgcn = torch.zeros(1, self.input_dim).to(out_.device)
        else:
            g = dgl.graph((str_src, str_dst))
            etype = str_edge_type
            hidden = self.conv1(g, out_, etype)
            hidden = torch.relu(hidden)
            hidden_rgcn = self.conv2(g, hidden, etype)

        # utilize CoAtt to generate context representation
        if len_dial < self.window:
            relatt_out = out_
        elif self.use_future:
            pre_pad = torch.LongTensor([0] * self.slide_win)
            post_pad = torch.LongTensor([-1] * self.slide_win)
            utt_ids = torch.cat((pre_pad, torch.arange(len_dial), post_pad))
            relatt_ids = utt_ids.unfold(0, self.window, 1)
            batch_input = out_[relatt_ids].unsqueeze(1)  # batch, channel(1), seq_len, dim
            # relatt_out = self.relAtt(batch_input)[:, :, self.slide_win, :].squeeze(1)
            # use the average vectors instead of the slide_win th vector
            relatt_out = torch.mean(self.relAtt(batch_input), dim=2).squeeze(1)
        else:
            # only use the previous utterances
            pre_pad = torch.LongTensor([0] * self.slide_win)
            utt_ids = torch.cat((pre_pad, torch.arange(len_dial)))
            relatt_ids = utt_ids.unfold(0, self.slide_win + 1, 1)
            batch_input = out_[relatt_ids].unsqueeze(1)  # batch, channel(1), seq_len, dim
            relatt_out = self.relAtt(batch_input)[:, :, self.slide_win, :].squeeze(1)
            # use the average vectors instead of the slide_win th vector
            # relatt_out = torch.mean(self.relAtt(batch_input), dim=2).squeeze(1)

        # process concept
        output_ = []
        losses = 0

        for chunk in chunks:
            srcs_input_ids, srcs_token_type_ids, srcs_attention_mask,srcs_sel_mask, dsts_input_ids, dsts_token_type_ids,\
            dsts_attention_mask, dsts_sel_mask, weights, sentics, src_masks, masks, rels, utt_idx = chunk

            if masks.sum()==0:
                symbolic_repr = torch.zeros(masks.size(0), self.input_dim).to(out_.device)
            else:

                chunk_size, num_src, num_dst = weights.size()

                src_emb = self.get_cpt_emb([srcs_input_ids, srcs_token_type_ids, srcs_attention_mask,
                srcs_sel_mask], chunk_size, num_src)
                dst_emb = self.get_cpt_emb([dsts_input_ids, dsts_token_type_ids, dsts_attention_mask,
                dsts_sel_mask], chunk_size, num_src*num_dst)

                cpt_emb = self.symbolic_proc(relatt_out[utt_idx], dst_emb,
                                             weights, sentics, src_masks, masks, chunk_size, num_src, num_dst)

                # integrate relation info into concept embedding
                r_vector = self.r[rels]
                re_dot = r_vector * cpt_emb # chunk_size, num_src, num_dst, self.input_dim
                s_score = torch.sum(src_emb.unsqueeze(2) * re_dot, dim=-1)
                s_score_masked = self.get_att_masked(s_score, masks)
                alpha = torch.softmax(s_score_masked, dim=2) * src_masks.unsqueeze(2)
                src_emb = src_emb + torch.sum(alpha.unsqueeze(3) * re_dot, dim=2)

                if self.use_dot_att:
                    dot_sum = torch.sum(src_emb *
                                        relatt_out[utt_idx].unsqueeze(1), dim=-1)

                    src_mask = torch.sum(masks, dim=-1) > 0
                    att_score = torch.softmax(self.get_att_masked(dot_sum, src_mask), dim=-1) * src_masks
                    # att_score = torch.softmax(dot_sum, dim=-1) * src_mask
                    symbolic_repr = torch.sum(att_score.unsqueeze(2) * src_emb, dim=1)


                # use item attention to calculate the attention score between src_emb and relatt_out
                else:
                    # item_att = self.item_att(relatt_out[utt_idx].unsqueeze(1).repeat(1, src_emb.size(1), 1), src_emb)
                    # item_sum = self.linear_att(item_att).squeeze(-1)
                    # src_mask = torch.sum(masks, dim=-1) > 0
                    # att_score = torch.softmax(self.get_att_masked(item_sum, src_mask), dim=-1) * src_masks
                    # symbolic_repr = torch.sum(att_score.unsqueeze(2) * src_emb, dim=1)

                    att_feature = torch.cat((relatt_out[utt_idx].unsqueeze(1).repeat(1, src_emb.size(1), 1), src_emb), dim=-1)
                    att_sum = self.att_linear(att_feature).squeeze(-1)
                    src_mask = torch.sum(masks, dim=-1) > 0
                    att_score = torch.softmax(self.get_att_masked(att_sum, src_mask), dim=-1) * src_masks
                    symbolic_repr = torch.sum(att_score.unsqueeze(2) * src_emb, dim=1)


            # feature fusion
            if self.num_feature == 4:
                feat = torch.cat((out_[utt_idx], hidden_rgcn[utt_idx],
                                  relatt_out[utt_idx], symbolic_repr), dim=-1)
                output = torch.log_softmax(self.linear(self.ac_tanh(self.dropout(self.fusion(feat)))), dim=1)
            elif self.num_feature == 3:
                feat = torch.cat((symbolic_repr, hidden_rgcn[utt_idx],
                                  relatt_out[utt_idx]), dim=-1)
                output = torch.log_softmax(self.linear(self.ac_tanh(self.dropout(self.fusion(feat)))), dim=1)
            elif self.num_feature == 2:
                feat = torch.cat((out_[utt_idx], hidden_rgcn[utt_idx]), dim=-1)
                output = torch.log_softmax(self.linear(self.ac_tanh(self.dropout(self.fusion(feat)))), dim=1)
            elif self.num_feature == 1:
                feat = out_[utt_idx]
                output = torch.log_softmax(self.linear(self.ac_tanh(self.dropout(self.fusion(feat)))), dim=1)
            else:
                print("ValueError!")

            loss = loss_func(output, label[utt_idx]) / len_dial

            if train:
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

        sel_mask_sums = torch.sum(sel_mask.contiguous().view(chunk_size,num_nd,-1), dim=2, keepdim=True)

        sel_embs = torch.sum(sel_mask.contiguous().view(-1, seq_lim).unsqueeze(2) * embs, dim=1).contiguous().view(chunk_size, num_nd,
                                                                                    self.input_dim) / sel_mask_sums
        return sel_embs  # bz, num_nd, self.input_dim

    def symbolic_proc(self, relatt_out_chunk, dst_emb, weights, sentics,src_masks, masks, chunk_size, num_src, num_dst):
        cosine_sim = torch.abs(torch.cosine_similarity(relatt_out_chunk.unsqueeze(1).repeat(1, num_src * num_dst, 1),
                                                       dst_emb, dim=-1))
        relatedness = weights * cosine_sim.contiguous().view(chunk_size, num_src, num_dst)
        omega = self.lamb * relatedness + (1 - self.lamb) * torch.abs(sentics)
        omega = self.get_att_masked(omega, masks)
        alpha = (src_masks.unsqueeze(2) * torch.softmax(omega, dim=-1)).unsqueeze(2).repeat(1, 1, self.input_dim, 1).transpose(2, 3)
        cpt_emb = alpha * dst_emb.contiguous().view(chunk_size, num_src, num_dst, -1)
        return cpt_emb

    def get_att_masked(self, inp, mask):
        inp[mask == False] = float("-inf")
        return inp
