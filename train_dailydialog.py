'''
Author: Li Wei, Zhu Luyao
Email: wei008@e.ntu.edu.sg
'''

import torch
import time
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, \
    classification_report, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import DailyDialogueDataset
import dgl
from dgl.nn.pytorch import RelGraphConv
from configs import inputconfig_func
from tqdm import tqdm
from transformers import AlbertModel
from preRelAtt import RelAtt

np.random.seed(1234)
torch.random.manual_seed(1234)
torch.cuda.manual_seed(1234)


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_loader_daily(path, batch_size=1, num_workers=2, MAX_L=20, pin_memory=False, cuda_=False):
    trainset = DailyDialogueDataset(split='train', path=path, MAX_L=20, cuda=cuda_)
    cpt_ids = trainset.cpt_ids
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    validset = DailyDialogueDataset(split='valid', path=path, MAX_L=20, cuda=cuda_)
    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=validset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = DailyDialogueDataset(split='test', path=path, MAX_L=20, cuda=cuda_)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader, cpt_ids


class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        mask_ = mask.view(-1, 1)  # batch*seq_len, 1
        if type(self.weight) == type(None):
            loss = self.loss(pred * mask_, target) / torch.sum(mask)
        else:
            loss = self.loss(pred * mask_, target) \
                   / torch.sum(self.weight[target] * mask_.squeeze())
        return loss


def generate_pos(input_ids):

    input_ids_l = input_ids[:].tolist()
    return input_ids_l


class Model(nn.Module):
    def __init__(self, input_dim, n_class, n_relations, cpt_ids, slide_win=1, cuda_=True):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.num_class = n_class
        self.num_relations = n_relations
        self.cpt_ids = cpt_ids
        self.bert_encoder = AlbertModel.from_pretrained('albert-base-v2')
        self.window = 2 * slide_win + 1
        self.slide_win = slide_win
        self.lamb = 0.5

        self.fw = torch.nn.Linear(768, self.input_dim)
        self.fc = torch.nn.Softmax(dim=1)

        self.fw_concept = torch.nn.Linear(768, self.input_dim)

        self.conv1 = RelGraphConv(self.input_dim, self.input_dim, n_relations, regularizer='basis', num_bases=2)
        # self.conv2 = RelGraphConv(self.input_dim, self.input_dim, n_relations, regularizer='basis', num_bases=2)

        self.relAtt = RelAtt(1, 1, (self.window, self.input_dim), heads=2, dim_head=self.input_dim // 2)

        self.isa_rl = nn.Parameter(nn.init.uniform_(torch.zeros(1, self.input_dim)))
        self.causes_rl = nn.Parameter(nn.init.uniform_(torch.zeros(1, self.input_dim)))
        self.hcontext_rl = nn.Parameter(nn.init.uniform_(torch.zeros(1, self.input_dim)))
        self.r = [self.isa_rl, self.causes_rl, self.hcontext_rl]

        self.fusion = nn.Linear(3*self.input_dim, self.input_dim)
        self.linear = nn.Linear(self.input_dim, self.num_class)
        self.ac = nn.ReLU()
        self.cuda_ =cuda_

    def forward(self, inp_ids, att_mask, token_type, str_src, str_dst, str_edge_type, isa_src, isa_dst, causes_src,
                causes_dst, hcontext_src, hcontext_dst, isa_weight, causes_weight, hcontext_weight,
                isa_sentic, causes_sentic, hcontext_sentic, agg_graph_src, agg_graph_dst):
        # torch.autograd.set_detect_anomaly(True)

        len_dial = len(inp_ids)
        out = self.bert_encoder(input_ids=inp_ids, attention_mask=att_mask, token_type_ids=token_type.long())
        out_ = self.fw(out[0][:, 0, :])

        # relational graph neural network used to embed dialog structure knowledge
        # need to decide whether there is only one utterance in a dialog
        if out_.size(0) == 1:
            hidden_rgcn = torch.zeros(1, self.input_dim).to(out_.device)
        else:
            g = dgl.graph((str_src, str_dst))
            etype = str_edge_type
            hidden_rgcn = self.conv1(g, out_, etype)

        # utilize RelAtt to generate context representation
        if len_dial < self.window:
            relatt_out = out_
        else:
            pre_pad = torch.LongTensor([0] * self.slide_win)
            post_pad = torch.LongTensor([-1] * self.slide_win)
            utt_ids = torch.cat((pre_pad, torch.arange(len_dial), post_pad))
            relatt_ids = utt_ids.unfold(0, self.window, 1)
            batch_input = out_[relatt_ids].unsqueeze(1)  # batch, channel(1), seq_len, dim
            relatt_out = self.relAtt(batch_input)[:, :, 1, :].squeeze(1)

        # process concept
        symbolic_repr = []
        symbolic_mask = torch.ones(len_dial, 1).to(inp_ids.device)
        for i in range(len_dial):
            if len(isa_src[i]) > 0:
                # src_emb = self.sel_embs[isa_src[i]]
                isa_src_emb = self.get_cpt_emb(isa_src[i], node='src')

                isa_cpt_emb = self.symbolic_proc(isa_dst[i], isa_weight[i], relatt_out[i], isa_sentic[i])
            else:
                isa_src_emb = []
                isa_cpt_emb = []


            if len(causes_src[i]) > 0:
                causes_src_emb = self.get_cpt_emb(causes_src[i], node='causes')

                causes_cpt_emb = self.symbolic_proc(causes_dst[i], causes_weight[i], relatt_out[i], causes_sentic[i])
            else:
                causes_src_emb = []
                causes_cpt_emb = []

            if len(hcontext_src[i]) > 0:
                hcontext_src_emb = self.get_cpt_emb(hcontext_src[i], node='hcontext')

                hcontext_cpt_emb = self.symbolic_proc(hcontext_dst[i], hcontext_weight[i], relatt_out[i], hcontext_sentic[i])

            else:
                hcontext_src_emb = []
                hcontext_cpt_emb = []


            if len(isa_src_emb) + len(causes_src_emb) + len(hcontext_src_emb) == 0:
                symbolic_repr.append(torch.zeros(1, self.input_dim).to(inp_ids.device))
                symbolic_mask[i][0] = 0.
                continue
            src_emb = [isa_src_emb, causes_src_emb, hcontext_src_emb]
            dst_emb = [isa_cpt_emb, causes_cpt_emb, hcontext_cpt_emb]

            src_pos = agg_graph_src[i]
            dst_pos = agg_graph_dst[i]
            emb_src_node = []
            for k in range(len(src_pos)):
                emb_src_k = src_emb[src_pos[k][0]][src_pos[k][1]][:].unsqueeze(0)
                emb_dst_k = torch.cat([dst_emb[dst_pos[k][j][0]][dst_pos[k][j][1]][:] for j in range(len(dst_pos[k]))], dim=0)
                # emb_dst_k = dst_emb[dst_pos[k][0]][dst_pos[k][1]]
                r_vector = torch.cat([self.r[dst_pos[k][j][0]] for j in range(len(dst_pos[k])) for _ in range(len(dst_emb[dst_pos[k][j][0]][dst_pos[k][j][1]]))], dim=0)

                re_dot = r_vector*emb_dst_k
                s_score = torch.sum(emb_src_k*re_dot, dim=-1, keepdim=True)
                alpha = torch.softmax(s_score, dim=0)
                emb_src_k_ = emb_src_k + torch.sum(alpha*re_dot, dim=0, keepdim=True)
                emb_src_node.append(emb_src_k_)

            # generate sen-level knowledge repr from word-level repr via attention
            emb_node = torch.cat(emb_src_node, dim=0)
            # att_score = torch.softmax(torch.sum(relatt_out[i].unsqueeze(0)*emb_node), dim=-1)
            att_score = torch.softmax(torch.matmul(emb_node, relatt_out[i].unsqueeze(1)), dim=0)
            symbolic_repr_i = torch.sum(att_score*emb_node, dim=0).unsqueeze(0)
            symbolic_repr.append(symbolic_repr_i)

        symbolic_repr = torch.cat(symbolic_repr, dim=0)*symbolic_mask

        # feature fusion
        feat = torch.cat((hidden_rgcn, relatt_out, symbolic_repr), dim=-1)
        output = torch.log_softmax(self.linear(self.ac(self.fusion(feat))), dim=1)

        return output



    def symbolic_proc(self, dst_i, weight_i, relatt_out_i, sentic_i):
            # src_emb = self.sel_embs[isa_src[i]]
            # src_emb = self.get_cpt_emb(src_i, node='src')
            dst_emb = self.get_cpt_emb(dst_i)

            dst_emb = torch.stack([torch.stack(de) for de in dst_emb])
            mask_dst_ = dst_i != -1

            cosine_sim = torch.abs(torch.cosine_similarity(relatt_out_i, dst_emb, dim=-1))
            relatedness = weight_i * cosine_sim  # * mask_dst_
            omega = self.lamb * relatedness + (1-self.lamb)*torch.abs(sentic_i)
            omega = self.get_att_masked(omega, mask_dst_)
            alpha = torch.softmax(omega, dim=-1).repeat(1, self.input_dim)\
                .contiguous().view(dst_emb.size(0), self.input_dim, -1).transpose(1, 2)
            cpt_emb = alpha * dst_emb

            return cpt_emb


    def get_cpt_emb(self, ids, node='dst'):
        # for inp in zip(self.cpt_ids, self.):
        #     out = self.bert_encoder(inp)
        #     emb = out[0].masked_select()
        if node == 'dst':
            batch, seq = ids.size()
            ids = ids.contiguous().view(batch * seq)
        if self.cuda_:
            inp_ids = self.cpt_ids['input_ids'][ids].cuda()
            att_mask = self.cpt_ids['attention_mask'][ids].cuda()
            token_type_ids = self.cpt_ids['token_type_ids'][ids].cuda()
            sel_mask = self.cpt_ids['sel_mask'][ids].cuda()
        else:
            inp_ids = self.cpt_ids['input_ids'][ids]
            att_mask = self.cpt_ids['attention_mask'][ids]
            token_type_ids = self.cpt_ids['token_type_ids'][ids]
            sel_mask = self.cpt_ids['sel_mask'][ids]
        if node == 'dst':
            # batch, seq = ids.size()
            # ids = ids.contiguous().view(batch * seq)
            out = self.bert_encoder(input_ids=inp_ids,
                                    attention_mask=att_mask,
                                    token_type_ids=token_type_ids)

            embs = out[0]
            embs = self.fw_concept(embs)
            sel_mask_sums = torch.sum(sel_mask, dim=1)
            # sel_embs = embs*self.cpt_ids['sel_mask'][ids]/sel_mask_sum
            # emb[torch.where(self.cpt_ids['sel_mask'][ids]==1)]
            sel_embs = []
            for emb, sel_m, sel_mask_sum in zip(embs, sel_mask, sel_mask_sums):
                sel = torch.sum(emb[torch.where(sel_m == 1)], dim=0)
                sel_embs.append(sel / sel_mask_sum)
            return [sel_embs[i * seq:(i + 1) * seq] for i in range(batch)]

        out = self.bert_encoder(input_ids=inp_ids.squeeze(1),
                                attention_mask=att_mask.squeeze(1),
                                token_type_ids=token_type_ids.squeeze(1))

        embs = out[0]
        embs = self.fw_concept(embs)
        sel_mask_sums = torch.sum(sel_mask.squeeze(1), dim=1)
        # sel_embs = embs*self.cpt_ids['sel_mask'][ids]/sel_mask_sum
        # emb[torch.where(self.cpt_ids['sel_mask'][ids]==1)]
        sel_embs = []
        for emb, sel_m, sel_mask_sum in zip(embs, sel_mask.squeeze(1), sel_mask_sums):
            sel = torch.sum(emb[torch.where(sel_m == 1)], dim=0)
            sel_embs.append(sel / sel_mask_sum)
        return sel_embs

    def get_att_masked(self, inp, mask):
        # sel_id = torch.where(mask is False)
        inp[mask == False] = float("-inf")
        return inp


def train_or_eval_model(model, loss_Func, dataloader, epoch, optimizer=None, train=True, cuda_=False):
    losses = []
    preds = []
    labels = []
    masks = []

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    for data in tqdm(dataloader):
        if train:
            optimizer.zero_grad()

        sent_ids, mask, token_types, cpt_graph_isa_src, cpt_graph_causes_src, \
        cpt_graph_hascontext_src, cpt_graph_isa_dst, cpt_graph_causes_dst, cpt_graph_hascontext_dst, speakers, umask, _, label, \
        cpt_graph_isa_weight, cpt_graph_causes_weight, cpt_graph_hascontext_weight, cpt_graph_isa_sentic, cpt_graph_causes_sentic, \
        cpt_graph_hascontext_sentic, agg_graph_src, agg_graph_dst, str_src, str_dst, str_edge_type = \
            [d.cuda() if torch.is_tensor(d) else d for d in data[:-2]] if cuda_ else data[:-2]


        len_cpt = len(cpt_graph_isa_src)
        for idx in range(len_cpt):

            for cpt in [cpt_graph_isa_src, cpt_graph_isa_dst, cpt_graph_causes_src, cpt_graph_causes_dst,
                        cpt_graph_hascontext_src, cpt_graph_hascontext_dst, cpt_graph_isa_weight,
                        cpt_graph_causes_weight, cpt_graph_hascontext_weight, cpt_graph_isa_sentic, cpt_graph_causes_sentic,
                        cpt_graph_hascontext_sentic, agg_graph_src, agg_graph_dst]:
                cpt[idx] = cpt[idx].cuda() if torch.is_tensor(cpt[idx]) and cuda_ else cpt[idx]

        log_prob = model.forward(sent_ids, mask, token_types, str_src, str_dst, str_edge_type, cpt_graph_isa_src,
                             cpt_graph_isa_dst, cpt_graph_causes_src, cpt_graph_causes_dst,
                             cpt_graph_hascontext_src, cpt_graph_hascontext_dst, cpt_graph_isa_weight,
                             cpt_graph_causes_weight, cpt_graph_hascontext_weight, cpt_graph_isa_sentic, cpt_graph_causes_sentic,
                        cpt_graph_hascontext_sentic, agg_graph_src, agg_graph_dst)

        labels_ = label.view(-1)
        loss = loss_Func(log_prob, labels_, umask)

        pred_ = torch.argmax(log_prob, 1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())
        losses.append(loss.item() * masks[-1].sum())
        if train:
            # with torch.autograd.detect_anomaly():
            loss.backward()
            optimizer.step()
        torch.cuda.empty_cache()

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'), []

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)
    class_report = classification_report(labels, preds, sample_weight=masks, digits=4)

    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, class_report


if __name__ == '__main__':

    Configs = inputconfig_func()
    print(Configs)

    if Configs.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    batch_size = Configs.batch_size
    n_classes = Configs.num_class
    n_relations = Configs.num_relations
    cuda_ = Configs.cuda
    n_epochs = Configs.epochs
    dropout = Configs.dropout
    max_sen_len = Configs.max_sen_len
    D_m = 100

    loss_weights = torch.FloatTensor([1.2959, 0.7958, 0.8276, 1.4088, 0.9560, 1.0575, 0.6585])


    train_loader, valid_loader, test_loader, cpt_ids = get_loader_daily('./data/dailydialog/Daily.pkl',
                                                                        batch_size=Configs.batch_size,
                                                                        num_workers=Configs.num_workers, MAX_L=max_sen_len, cuda_=cuda_)

    model = Model(D_m, n_classes, n_relations, cpt_ids)
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(num_params / 1e6)

    if cuda_:
        model.cuda()

    if Configs.class_weight:
        loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda_ else loss_weights)
    else:
        loss_function = MaskedNLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=Configs.lr, weight_decay=Configs.l2)

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None

    for e in range(Configs.epochs):
        start_time = time.time()
        train_loss, train_acc, _, _, _, train_fscore, _ = train_or_eval_model(model, loss_Func=loss_function,
                                                                           dataloader=train_loader, epoch=e,
                                                                           optimizer=optimizer, train=True,
                                                                           cuda_=Configs.cuda)

        # valid_loss, valid_acc, _, _, _, valid_fscore, _ = train_or_eval_model(model, loss_Func=loss_function,
        #                                                                    dataloader=valid_loader, epoch=e,
        #                                                                    train=False, cuda_=Configs.cuda)
        with torch.no_grad():
            test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, test_class_report = train_or_eval_model(model, loss_Func=loss_function,
                                                                            dataloader=test_loader, epoch=e,
                                                                            train=False, cuda_=Configs.cuda)

        if best_fscore == None or best_fscore < test_fscore:
            best_fscore, best_loss, best_label, best_pred, best_mask = \
                test_fscore, test_loss, test_label, test_pred, test_mask

        print('epoch {} train_loss {} train_acc {} train_fscore {} test_loss {} test_acc {} test_fscore {} time {}'. \
                format(e + 1, train_loss, train_acc, train_fscore, test_loss,
                       test_acc, test_fscore, round(time.time() - start_time, 2)))
        print(test_class_report)

    print('Test performance..')
    print('Fscore {} accuracy {}'.format(best_fscore,
                                         round(accuracy_score(best_label, best_pred, sample_weight=best_mask) * 100,
                                               2)))
    print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
    print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))