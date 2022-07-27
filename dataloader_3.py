'''
Author: Luyao Zhu, Li Wei
Email: wei008@e.ntu.edu.sg
'''

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd
from transformers import AlbertTokenizer
from transformers import RobertaTokenizer
import json
import re
from senticnet.senticnet import SenticNet
from configs import inputconfig_func
Configs = inputconfig_func()

torch.random.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.manual_seed(1234)

def tokenize(data, tokenizer, MAX_L=20, model_type='albert'):
    input_ids = {}
    masks = {}
    token_types = {}
    keys = data.keys()
    for key in keys:
        dial = data[key]
        for idx, _ in enumerate(dial):
            dial[idx] = re.sub(r'\x92', '', dial[idx])
            dial[idx] = re.sub(r'\x91', '', dial[idx])
        # input_ids[key] = []
        # masks[key] = []
        # token_types[key] = []
        #
        # for utt in dial:
        #     # text_tokenized = tokenizer(utt, max_length=MAX_L, padding='max_length', truncation=True)
        #     text_tokenized = tokenizer(utt)
        #     input_ids[key].append(text_tokenized['input_ids'][-MAX_L:])
        #     masks[key].append(text_tokenized['attention_mask'][-MAX_L:])
        #     if model_type == 'albert':
        #         token_types[key].append(text_tokenized['token_type_ids'][-MAX_L:])
        res = tokenizer(dial, padding='longest', return_tensors='pt')
        input_ids[key] = res['input_ids']
        masks[key] = res['attention_mask']
        token_types[key] = []
        if model_type == 'albert':
            token_types[key] = res['token_type_ids']

    return input_ids, masks, token_types


def prepare_graph(structure):
    src = {}
    dst = {}
    edge_type = {}
    keys = structure.keys()

    for key in keys:
        src[key] = []
        dst[key] = []
        edge_type[key] = []
        dial = structure[key]
        for utt in dial:
            src[key].append(utt['x'])
            dst[key].append(utt['y'])
            edge_type[key].append(utt['type_num'])

    return src, dst, edge_type


def gen_cpt_vocab(sn, dst_num_per_rel=Configs.dst_num_per_rel):
    cpt_vocab = ['<pad>']
    rel_list = ['isa', 'causes', 'hascontext']
    dicts = []
    for rel in rel_list:
        with open('./data/dialog_concept/{}_weight_dict_all.json'.format(rel), 'r') as f:
            rel_dict_origin = json.load(f)
        # filter dst node with weight limit
        rel_dict = {}
        for key in rel_dict_origin.keys():
            dst = rel_dict_origin[key]
            weights = [item[1] for item in dst]
            weights_scaled = min_max_scale(weights)
            dst_ = []
            for idx, item in enumerate(dst):
                try:
                    dst_.append([item[0], weights_scaled[idx], abs(float(sn.polarity_value(item[0])))])
                except KeyError:
                    dst_.append([item[0], weights_scaled[idx], 0.])
            dst_.sort(key =lambda i: i[1]+i[2], reverse=True)
            l_idx = min(len(dst_), dst_num_per_rel)
            rel_dict[key] = dst_[0:l_idx]

        keys = rel_dict.keys()
        dicts.append(rel_dict)
        for key in keys:
            if key not in cpt_vocab:
                cpt_vocab.append(key)
            for v in rel_dict[key]:
                if v[0] not in cpt_vocab:
                    cpt_vocab.append(v[0])
    return cpt_vocab, dicts


def tok_cpt_vocab(tokenizer, cpt_vocab, cuda=False):
    cpt_ids = tokenizer(cpt_vocab, max_length=5, padding='max_length', truncation=True)
    # for token in cpt_vocab:
    #     cpt_ids.append(tokenizer(token))
    # cpt_len = [sum(token['attention_mask']) for token in cpt_ids]
    # sel_mask = []
    pad_cpt_ids = {}
    pad_cpt_ids['sel_mask'] = []
    keys = cpt_ids.keys()
    for key in keys:
        items = []
        # sel_mask = []
        for item in cpt_ids[key]:
            if key == 'attention_mask':
                sel_mask = item[:]
                sel_mask[0] = 0
                sel_mask[-1] = 0
                pad_cpt_ids['sel_mask'].append(torch.LongTensor(sel_mask))
            items.append(torch.LongTensor(item))

        pads = pad_sequence(items, batch_first=True)
        pad_cpt_ids[key] = pads#.cuda() if cuda else pads
    sel_masks = pad_sequence(pad_cpt_ids['sel_mask'], batch_first=True)
    pad_cpt_ids['sel_mask'] = sel_masks#.cuda() if cuda else sel_masks
    return pad_cpt_ids


# dst_num = num_rel * dst_num_per_rel
def gen_cpt_graph(text, cpt_vocab, isa_dict, causes_dict, hscnt_dict, sn, src_num=Configs.src_num, dst_num=Configs.dst_num_per_rel*3):
    graph = {}
    keys = text.keys()

    for key in keys:  # key: dialogue index
        dial = text[key]
        srcs = []
        dsts = []
        weights = []
        sentics = []
        rel_types = []
        masks = []
        for idx, utt in enumerate(dial):
            tokens = utt.strip().split()
            src = []
            traversed_src = []
            src_pos = 0
            dst_ = []
            weight_ = []
            sentic_ = []
            rel_type_ = []
            mask_ = []
            for token in tokens:
                dst = [0 for _ in range(dst_num)]
                weight = [0. for _ in range(dst_num)]
                sentic = [0. for _ in range(dst_num)]
                rel_type = [0 for _ in range(dst_num)]
                mask = [0 for _ in range(dst_num)]
                # retrieve cpt dst and rel for each token
                if token in isa_dict:
                    isa_res = isa_dict[token]
                else:
                    isa_res = []
                if token in causes_dict:
                    causes_res = causes_dict[token]
                else:
                    causes_res = []
                if token in hscnt_dict:
                    hscnt_res = hscnt_dict[token]
                else:
                    hscnt_res = []
                res = [isa_res, causes_res, hscnt_res]

                if token not in traversed_src and (token in isa_dict or token in causes_dict or token in hscnt_dict):
                    try:
                        src.append([cpt_vocab.index(token), float(sn.polarity_value(token)), src_pos])
                    except KeyError:
                        src.append([cpt_vocab.index(token), 0., src_pos])
                    traversed_src.append(token)
                    src_pos += 1

                dst_count = 0
                for idx_r, res_ in enumerate(res):

                    for e in res_:
                        # dst.append(cpt_vocab.index(e[0]))
                        # weight.append(e[1])
                        # rel_type.append(idx_r)
                        dst[dst_count]= cpt_vocab.index(e[0])
                        weight[dst_count] = e[1]
                        rel_type[dst_count] = idx_r
                        mask[dst_count] = 1

                        # retrieve polarity_value from senticnet for each dst_node
                        e_ = [tok.lower() for tok in e[0].split(' ')]
                        score = []
                        for tok in e_:
                            try:
                                score.append(float(sn.polarity_value(tok)))
                            except KeyError:
                                pass
                        # sentic.append(sum(score) / len(score) if len(score) > 0 else 0.)
                        sentic[dst_count] = (sum(score) / len(score) if len(score) > 0 else 0.)
                        dst_count += 1

                weight_scaled = min_max_scale(weight)
                if sum(dst)>0:
                    dst_.append(dst)
                    weight_.append(weight_scaled)
                    sentic_.append(sentic)
                    rel_type_.append(rel_type)
                    mask_.append(mask)

            src.sort(key=lambda i: i[1], reverse=True)
            l_idx = min(len(src), src_num)
            srcs.append(torch.LongTensor([item[0] for item in src[:l_idx]]))

            dsts.append([dst_[item[2]] for item in src[:l_idx]])
            weights.append([weight_[item[2]] for item in src[:l_idx]])
            sentics.append([sentic_[item[2]] for item in src[:l_idx]])
            rel_types.append([rel_type_[item[2]] for item in src[:l_idx]])
            masks.append([mask_[item[2]] for item in src[:l_idx]])

        # padding info
        srcs = pad_sequence(srcs, batch_first=True, padding_value=0)
        bz, max_n_srcs = srcs.size()

        for d, w, s, r, m in zip(dsts, weights, sentics, rel_types, masks):
            num_item = len(d)
            if num_item < max_n_srcs:
                for i in range(max_n_srcs-num_item):
                    d.append([0 for _ in range(dst_num)])
                    w.append([0. for _ in range(dst_num)])
                    s.append([0. for _ in range(dst_num)])
                    r.append([0 for _ in range(dst_num)])
                    m.append([0 for _ in range(dst_num)])
        src_masks = torch.sum(torch.LongTensor(masks), dim=-1)
        for m,src_m in zip(masks, src_masks):
            if len(m)>0:
                for i in range(src_m.size(0)):
                    m[i][0] = 1

        dial_graph = [torch.LongTensor(srcs), torch.LongTensor(dsts),
                      torch.FloatTensor(weights), torch.FloatTensor(sentics),
                      src_masks, torch.LongTensor(masks), torch.LongTensor(rel_types)]
        graph[key] = dial_graph

    return graph


def get_chunk(cpt_graph_i, cpt_ids, model_type='albert', chunk_size=10, dst_num=Configs.dst_num_per_rel*3):

    srcs, dsts, weights, sentics, src_masks, masks, rels = cpt_graph_i
    if masks.sum() == 0:
        bz, seq_len = srcs.size()
        utt_idx = torch.arange(bz).to(srcs.device)
        return [[srcs, srcs, srcs, srcs, dsts, dsts,dsts,dsts, weights, sentics, src_masks, masks, rels, utt_idx]]
    bz, seq_len = srcs.size()
    utt_idx = torch.arange(bz).to(srcs.device)
    num_chunck = bz//chunk_size+1 if bz%chunk_size>0 else bz//chunk_size
    srcs = srcs.contiguous().view(bz*seq_len, -1)
    srcs_input_ids = cpt_ids['input_ids'][srcs].contiguous().view(bz, seq_len, -1)

    srcs_attention_mask = cpt_ids['attention_mask'][srcs].contiguous().view(bz, seq_len, -1)
    srcs_sel_mask = cpt_ids['sel_mask'][srcs].contiguous().view(bz, seq_len, -1)
    dsts = dsts.contiguous().view(bz*seq_len*dst_num,-1)
    dsts_input_ids = cpt_ids['input_ids'][dsts].contiguous().view(bz, seq_len, dst_num, -1)

    dsts_attention_mask = cpt_ids['attention_mask'][dsts].contiguous().view(bz, seq_len, dst_num, -1)
    dsts_sel_mask = cpt_ids['sel_mask'][dsts].contiguous().view(bz, seq_len, dst_num, -1)
    srcs_input_ids_chunked = srcs_input_ids.chunk(num_chunck)

    srcs_attention_mask_chunked = srcs_attention_mask.chunk(num_chunck)
    srcs_sel_mask_chunked = srcs_sel_mask.chunk(num_chunck)
    dsts_input_ids_chunked = dsts_input_ids.chunk(num_chunck)

    dsts_attention_mask_chunked = dsts_attention_mask.chunk(num_chunck)
    dsts_sel_mask_chunked = dsts_sel_mask.chunk(num_chunck)
    utt_idx_chunked = utt_idx.chunk(num_chunck)
    weights_chunked = weights.chunk(num_chunck)
    sentics_chunked = sentics.chunk(num_chunck)
    src_masks_chunked = src_masks.chunk(num_chunck)
    masks_chunked = masks.chunk(num_chunck)
    rels_chunked = rels.chunk(num_chunck)
    if model_type == 'albert':
        srcs_token_type_ids = cpt_ids['token_type_ids'][srcs]
        dsts_token_type_ids = cpt_ids['token_type_ids'][dsts]
        srcs_token_type_ids_chunked = srcs_token_type_ids.chunk(num_chunck)
        dsts_token_type_ids_chunked = dsts_token_type_ids.chunk(num_chunck)
    elif model_type in ['roberta', 'roberta_large']:
        srcs_token_type_ids_chunked = [[] for _ in range(num_chunck)]
        dsts_token_type_ids_chunked = [[] for _ in range(num_chunck)]

    dial = []
    for i in range(num_chunck):
        chunk_content = [srcs_input_ids_chunked[i], srcs_token_type_ids_chunked[i], srcs_attention_mask_chunked[i],
                         srcs_sel_mask_chunked[i], dsts_input_ids_chunked[i], dsts_token_type_ids_chunked[i],
                         dsts_attention_mask_chunked[i], dsts_sel_mask_chunked[i], weights_chunked[i], sentics_chunked[i],
                         src_masks_chunked[i], masks_chunked[i], rels_chunked[i], utt_idx_chunked[i]]
        dial.append(chunk_content)

    return dial


# deprecated
def cpt_graph(text, cpt_vocab, rel_dict_ids, sn, MAX_L=20):
    graph = {}
    keys = text.keys()
    for key in keys:
        dial = text[key]
        srcs = []
        dsts = []
        weights = []
        sentics = []

        dst_ = []
        weight_ = []

        for idx, utt in enumerate(dial):
            words = utt.strip().split()
            src = []
            dst_ = []
            weight_ = []
            sentic_ = []


            for word in words:#[:MAX_L]:
                dst = []
                weight = []
                sentic = []
                if word in rel_dict_ids:
                    res = rel_dict_ids[word]
                    # src.append([cpt_vocab.index(word)])
                    try:
                        src.append([cpt_vocab.index(word), float(sn.polarity_value(word))])
                    except KeyError:
                        src.append([cpt_vocab.index(word), 0.])
                    for e in res:
                        # src.append(cpt_vocab.index(word))
                        dst.append(cpt_vocab.index(e[0]))
                        weight.append(e[1])

                        # retrieve polarity_value from senticnet for each dst_node
                        e_ = [tok.lower() for tok in e[0].split(' ')]
                        score = []
                        for tok in e_:
                            try:
                                score.append(float(sn.polarity_value(tok)))
                            except KeyError:
                                pass
                        sentic.append(sum(score) / len(score) if len(score) > 0 else 0.)

                    weight_scaled = min_max_scale(weight)
                    # sentic_scaled = min_max_scale(sentic)
                    dst_.append(dst)
                    weight_.append(weight_scaled)
                    # sentic_.append(sentic_scaled)
                    sentic_.append(sentic)
            src.sort(key = lambda i: i[1], reverse=True)
            l_idx = min(len(src), Configs.src_num)
            srcs.append([[item[0]] for item in src[:l_idx]])

            dsts.append(dst_)
            weights.append(weight_)
            sentics.append(sentic_)

        dial_graph = [srcs, dsts, weights, sentics]
        graph[key] = dial_graph
    return graph


# deprecated
def merge(isa_graph, causes_graph, hcontext_graph):
    keys = isa_graph.keys()
    agg_graph = {}
    # isa_src, causes_src, hcontext_src = isa_graph[], causes_graph, hcontext_graph
    for key in keys:
        # dial_src = [isa_graph[key][0], causes_graph[key][0], hcontext_graph[key][0]]
        # dial_dst = [isa_graph[key][1], causes_graph[key][1], hcontext_graph[key][1]]
        # agg_dial_src = []
        # agg_dial_dst = []
        utt_srcs = []
        utt_dsts = []
        utt_srcs_origin = []
        for idx, utt in enumerate(zip(isa_graph[key][0], causes_graph[key][0], hcontext_graph[key][0])):

            # if len(utt[0]) + len(utt[1]) + len(utt[2]) == 0:
            #     continue
            utt_srcs.append([])
            utt_dsts.append([])
            utt_srcs_origin.append([])
            for idx_i, nodes in enumerate(utt):
                for idx_j, nd in enumerate(nodes):
                    # if len(nd) == 0:
                    #     continue
                    if nd not in utt_srcs_origin[idx]:
                        utt_srcs[idx].append([idx_i, idx_j])
                        utt_srcs_origin[idx].append(nd)
                        # for idx_k in range(len(dial_dst[idx][idx_i][idx_j])):
                        utt_dsts[idx].append(
                            [[idx_i, idx_j]])  # idx_i: relation_type_id, idx_j: node positinon in each nodes
                    else:
                        nd_pos = utt_srcs_origin[idx].index(nd)
                        utt_dsts[idx][nd_pos].append([idx_i, idx_j])
            utt_srcs[idx] = [torch.LongTensor(item) for item in utt_srcs[idx]]
            utt_dsts[idx] = [torch.LongTensor(item) for item in utt_dsts[idx]]
        agg_graph[key] = [utt_srcs, utt_dsts]
    return agg_graph


def min_max_scale(v, new_min=0, new_max=1.0):
    if len(v) == 0:
        return v
    v_min, v_max = min(v), max(v)
    if v_min == v_max:
        v_p = [new_max for e in v]
    else:
        v_p = [(e - v_min) / (v_max - v_min) * (new_max - new_min) + new_min for e in v]

    return v_p


class MELDDataset(Dataset):

    def __init__(self, path, n_classes=7, MAX_L=20, train=True, cuda=False, model_type='albert'):

        if model_type == 'albert':
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        elif model_type == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        elif model_type == 'roberta_large':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        sn = SenticNet()

        if n_classes == 3:
            self.videoIDs, self.videoSpeakers, _, self.videoText, \
            self.videoAudio, self.videoSentence, self.trainVid, \
            self.testVid, self.videoLabels = pickle.load(open(path, 'rb'))
        elif n_classes == 7:
            self.videoIDs_, self.videoSpeakers_, self.videoLabels_, self.videoText_, \
            self.videoAudio_, self.videoSentence_, self.trainVid, \
            self.testVid, _, self.structure_, self.action_ = pickle.load(open(path, 'rb'))
        '''
        label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
        '''
        self.cpt_vocab, [isa_dict, causes_dict, hascnt_dict] = gen_cpt_vocab(sn)
        self.cpt_ids = tok_cpt_vocab(self.tokenizer, self.cpt_vocab, cuda=cuda)
        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.videoAudio, self.videoSentence, self.structure, self.action = [self.partition(data) for data in
                                                                            [self.videoIDs_, self.videoSpeakers_,
                                                                             self.videoLabels_, self.videoText_,
                                                                             self.videoAudio_, self.videoSentence_,
                                                                             self.structure_, self.action_]]
        print('no_max_sen_len')
        self.sent_ids, self.masks, self.token_types = tokenize(self.videoSentence, self.tokenizer, MAX_L=MAX_L,
                                                               model_type=model_type)

        self.node_src, self.node_dst, self.edge_type = prepare_graph(self.structure)

        # isa_dict_ids, isa_src2ids = tokenize_concept(self.tokenizer, rel='isa')
        # causes_dict_ids, causes_src2ids = tokenize_concept(self.tokenizer, rel='causes')
        # hascnt_dict_ids, hascnt_src2ids = tokenize_concept(self.tokenizer, rel='hascontext')
        # self.cpt_graph_isa = concept_graph(self.videoSentence, isa_dict_ids, isa_src2ids)
        # self.cpt_graph_causes = concept_graph(self.videoSentence, causes_dict_ids, causes_src2ids)
        # self.cpt_graph_hascnt = concept_graph(self.videoSentence, hascnt_dict_ids, hascnt_src2ids)

        # self.cpt_graph_isa = locate_concept(self.videoSentence, self.sent_ids, isa_dict_ids, isa_src2ids)
        # self.cpt_graph_causes = locate_concept(self.videoSentence, self.sent_ids, causes_dict_ids, causes_src2ids)
        # self.cpt_graph_hascnt = locate_concept(self.videoSentence, self.sent_ids, hascnt_dict_ids, hascnt_src2ids)
        self.cpt_graph = gen_cpt_graph(self.videoSentence, self.cpt_vocab, isa_dict, causes_dict, hascnt_dict,sn)

        # self.cpt_graph_isa = cpt_graph(self.videoSentence, self.cpt_vocab, isa_dict, sn)
        # self.cpt_graph_causes = cpt_graph(self.videoSentence, self.cpt_vocab, causes_dict, sn)
        # self.cpt_graph_hascnt = cpt_graph(self.videoSentence, self.cpt_vocab, hascnt_dict, sn)
        # self.agg_graph = merge(self.cpt_graph_isa, self.cpt_graph_causes, self.cpt_graph_hascnt)
        # self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        # return torch.FloatTensor(self.videoText[vid]), \
        #        torch.FloatTensor(self.videoAudio[vid]), \
        #        torch.FloatTensor(self.videoSpeakers[vid]), \
        #        torch.FloatTensor([1] * len(self.videoLabels[vid])), \
        #        torch.LongTensor(self.videoLabels[vid]), \
        #        vid
        # return torch.LongTensor(self.sent_ids[vid]), \
        #        torch.LongTensor(self.masks[vid]), \
        #        torch.LongTensor(self.token_types[vid]), \
        return self.sent_ids[vid], \
               self.masks[vid], \
               self.token_types[vid], \
               self.cpt_graph[vid],\
               torch.FloatTensor(self.videoAudio[vid]), \
               torch.FloatTensor(self.videoSpeakers[vid]), \
               torch.FloatTensor([1] * len(self.videoLabels[vid])), \
               torch.LongTensor(self.videoLabels[vid]), \
               torch.LongTensor(self.node_src[vid]), \
               torch.LongTensor(self.node_dst[vid]), \
               torch.LongTensor(self.edge_type[vid]), \
               vid
        # torch.LongTensor(self.action[vid]), \

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        # return [dat[i] for i in dat]
        result = []

        for i in dat:

            if i < 3:
                # result.append(pad_sequence(dat[i][0], batch_first=True, padding_value=0))
                result.append(dat[i][0])
            elif i < 4:
                result.append(dat[i][0])
            elif i < 8:
                result.append(pad_sequence([dat[i][0]], True))
            elif i < 11:
                result.append(dat[i][0])
            else:
                result.append(dat[i].tolist())
        return result

    def partition(self, data):

        return {key: data[key] for key in self.keys}


class DailyDialogueDataset(Dataset):

    def __init__(self, split, path, MAX_L=20, cuda=False):

        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        # self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        sn = SenticNet()
        self.Speakers_, self.InputSequence_, self.InputMaxSequenceLength_, \
        self.ActLabels_, self.EmotionLabels_, self.trainId, self.testId, self.validId, \
        self.structure_, self.action_ = pickle.load(open(path, 'rb'))
        with open('./data/dailydialog/daily_.json', 'r') as f:
            self.text_ = json.load(f)

        if split == 'train':
            self.keys = [x for x in self.trainId]
        elif split == 'test':
            self.keys = [x for x in self.testId]
        elif split == 'valid':
            self.keys = [x for x in self.validId]

        self.cpt_vocab, [isa_dict, causes_dict, hascnt_dict] = gen_cpt_vocab()
        self.cpt_ids = tok_cpt_vocab(self.tokenizer, self.cpt_vocab, cuda=cuda)

        self.Speakers, self.InputSequence, self.InputMaxSequenceLength, \
        self.ActLabels, self.EmotionLabels, self.structure, self.action, self.text \
            = [self.partition(data) for data in [self.Speakers_, self.InputSequence_, self.InputMaxSequenceLength_,
                                                 self.ActLabels_, self.EmotionLabels_, self.structure_, self.action_,
                                                 self.text_]]

        self.sent_ids, self.masks, self.token_types = tokenize(self.text, self.tokenizer, MAX_L=20)
        self.node_src, self.node_dst, self.edge_type = prepare_graph(self.structure)

        # isa_dict_ids, isa_src2ids = tokenize_concept(self.tokenizer, rel='isa')
        # causes_dict_ids, causes_src2ids = tokenize_concept(self.tokenizer, rel='causes')
        # hascnt_dict_ids, hascnt_src2ids = tokenize_concept(self.tokenizer, rel='hascontext')
        # self.cpt_graph_isa = concept_graph(self.text, isa_dict_ids, isa_src2ids)
        # self.cpt_graph_causes = concept_graph(self.text, causes_dict_ids, causes_src2ids)
        # self.cpt_graph_hascnt = concept_graph(self.text, hascnt_dict_ids, hascnt_src2ids)

        # self.cpt_graph_isa = locate_concept(self.text, self.sent_ids, isa_dict_ids, isa_src2ids)
        # self.cpt_graph_causes = locate_concept(self.text, self.sent_ids, causes_dict_ids, causes_src2ids)
        # self.cpt_graph_hascnt = locate_concept(self.text, self.sent_ids, hascnt_dict_ids, hascnt_src2ids)
        self.cpt_graph_isa = cpt_graph(self.text, self.cpt_vocab, isa_dict, sn)
        self.cpt_graph_causes = cpt_graph(self.text, self.cpt_vocab, causes_dict, sn)
        self.cpt_graph_hascnt = cpt_graph(self.text, self.cpt_vocab, hascnt_dict, sn)
        self.agg_graph = merge(self.cpt_graph_isa, self.cpt_graph_causes, self.cpt_graph_hascnt)

        # if split == 'train':
        #     self.keys = [x for x in self.trainId]
        # elif split == 'test':
        #     self.keys = [x for x in self.testId]
        # elif split == 'valid':
        #     self.keys = [x for x in self.validId]

        self.len = len(self.keys)

    def __getitem__(self, index):
        conv = self.keys[index]

        # return torch.LongTensor(self.InputSequence[conv]), \
        #        torch.FloatTensor([[1, 0] if x == '0' else [0, 1] for x in self.Speakers[conv]]), \
        #        torch.FloatTensor([1] * len(self.ActLabels[conv])), \
        #        torch.LongTensor(self.ActLabels[conv]), \
        #        torch.LongTensor(self.EmotionLabels[conv]), \
        #        self.InputMaxSequenceLength[conv], \
        #        conv
        return torch.LongTensor(self.sent_ids[conv]), \
               torch.LongTensor(self.masks[conv]), \
               torch.LongTensor(self.token_types[conv]), \
               self.cpt_graph_isa[conv][0], \
               self.cpt_graph_causes[conv][0], \
               self.cpt_graph_hascnt[conv][0], \
               self.cpt_graph_isa[conv][1], \
               self.cpt_graph_causes[conv][1], \
               self.cpt_graph_hascnt[conv][1], \
               torch.FloatTensor([[1, 0] if x == '0' else [0, 1] for x in self.Speakers[conv]]), \
               torch.FloatTensor([1] * len(self.ActLabels[conv])), \
               torch.LongTensor(self.ActLabels[conv]), \
               torch.LongTensor(self.EmotionLabels[conv]), \
               self.cpt_graph_isa[conv][2], \
               self.cpt_graph_causes[conv][2], \
               self.cpt_graph_hascnt[conv][2], \
               self.cpt_graph_isa[conv][3], \
               self.cpt_graph_causes[conv][3], \
               self.cpt_graph_hascnt[conv][3], \
               self.agg_graph[conv][0], \
               self.agg_graph[conv][1], \
               torch.LongTensor(self.node_src[conv]), \
               torch.LongTensor(self.node_dst[conv]), \
               torch.LongTensor(self.edge_type[conv]), \
               self.InputMaxSequenceLength[conv], \
               conv

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        # return [pad_sequence(dat[i]) if i < 6 else pad_sequence(dat[i], True) if i < 10 else i if i<14 else dat[i].tolist() for i in
        #         dat]
        result = []

        for i in dat:

            if i < 3:
                # result.append(pad_sequence([dat[i][0]]))
                result.append(dat[i][0])
            elif i < 9:
                res = []
                for utt in dat[i][0]:
                    # try:
                    if len(utt) > 0:
                        nodes = [torch.LongTensor(nd) for nd in utt]
                        res.append(pad_sequence(nodes, batch_first=True, padding_value=-1))
                    else:
                        res.append(utt)
                    # if len(utt) > 0:
                    #
                    #     res.append(torch.LongTensor(utt))
                    # else:
                    #     res.append(utt)
                    # except RuntimeError:
                    #     print(utt)
                result.append(res)
                # res = []
            elif i < 13:
                result.append(pad_sequence([dat[i][0]], True))
                # result.append(dat[i])
                # res = []
                # for utt in data[i][0]:
                #     result.append(torch.LongTensor(utt))
            elif i < 19:
                res = []
                for utt in dat[i][0]:
                    # try:
                    if len(utt) > 0:
                        nodes = [torch.FloatTensor(nd) for nd in utt]
                        # res.append(nodes)
                        res.append(pad_sequence(nodes, batch_first=True, padding_value=-1))
                        # res.append(torch.FloatTensor(utt))
                    else:
                        res.append(utt)
                    # except RuntimeError:
                    #     print(utt)
                result.append(res)
                #
                #
                # res = []
                # for utt in dat[i][0]:
                #     # try:
                #     if len(utt) > 0:
                #
                #         res.append(torch.FloatTensor(utt))
                #     else:
                #         res.append(utt)
                #     # except RuntimeError:
                #     #     print(utt)
                # result.append(res)
            elif i < 25:
                result.append(dat[i][0])
            else:
                result.append(dat[i].tolist())
        return result

    def partition(self, data):

        return {key: data[key] for key in self.keys}


# deprecated
def tokenize_concept(tokenizer, rel='isa'):
    isa_dict_ids = {}
    src2ids = {}
    with open('./data/dialog_concept/{}_weight_dict_all.json'.format(rel), 'r') as f:
        isa_dict = json.load(f)

    keys = isa_dict.keys()
    for key in keys:
        tokenized = tokenizer(key)['input_ids']  # [1:-1]
        isa_dict_ids[key] = []
        src2ids[key] = tokenized
        for v, w in isa_dict[key]:
            # print(v)
            # break
            isa_dict_ids[key].append((tokenizer(v)['input_ids'][1:-1], w))
    return isa_dict_ids, src2ids


# deprecated
def locate_concept(text, sent_ids, isa_dict_ids, src2ids):
    graph = {}
    keys = sent_ids.keys()
    for key in keys:
        dial = sent_ids[key]
        dial_text = text[key]
        srcs = []
        dsts = []
        weights = []

        for idx, (utt, utt_text) in enumerate(zip(dial, dial_text)):
            # utt_ = utt[:].tolist()
            words = utt_text.strip().split()
            src = []
            dst = []
            weight = []

            for word in words:
                if word in isa_dict_ids:
                    res = isa_dict_ids[word]

                    for e in res:
                        src_ids = src2ids[word]
                        pos = [utt.index(e) for e in src_ids]
                        src.append(pos)
                        # pos = [utt.index(e) for e in e[0]]
                        dst.append(e[0])
                        weight.append(e[1])
            srcs.append(src)
            dsts.append(dst)
            weights.append(weight)

        dial_graph = [srcs, dsts, weights]
        graph[key] = dial_graph
    return graph