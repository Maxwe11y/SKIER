'''
Author: Li Wei, Zhu Luyao
Email: wei008@e.ntu.edu.sg
'''

import torch
import time
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, \
    classification_report, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader_10 import MELDDataset, DailyDialogueDataset, get_chunk, EmoryNLPDataset

from configs import inputconfig_func
from tqdm import tqdm
from model_dd import Model
from transformers import get_linear_schedule_with_warmup
import numpy
import random
from prepare_glove import config_logger
import warnings
warnings.filterwarnings("ignore")

np.random.seed(1234)
torch.random.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.manual_seed(1234)
random.seed(1234)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(1234)

def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_loader_meld(path, batch_size=1, valid=0.1, num_workers=2, MAX_L=20, num_class = 7, model_type='albert', pin_memory=False, cuda_=False):
    trainset = MELDDataset(path=path, n_classes=num_class, MAX_L=MAX_L, train=True, cuda=cuda_, model_type=model_type)
    cpt_ids = trainset.cpt_ids
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              worker_init_fn=seed_worker,
                              generator=g)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              worker_init_fn=seed_worker,
                              generator=g)

    testset = MELDDataset(path=path, n_classes=num_class,  MAX_L=MAX_L, train=False, cuda=cuda_, model_type=model_type)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             worker_init_fn=seed_worker,
                             generator=g)

    return train_loader, valid_loader, test_loader, cpt_ids


def get_loader_daily(path, batch_size=1, num_workers=2, MAX_L=20, model_type='albert', pin_memory=False, cuda_=False):
    trainset = DailyDialogueDataset(split='train', path=path, cuda=cuda_, model_type=model_type)
    cpt_ids = trainset.cpt_ids
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    validset = DailyDialogueDataset(split='valid', path=path, cuda=cuda_, model_type=model_type)
    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=validset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = DailyDialogueDataset(split='test', path=path, cuda=cuda_, model_type=model_type)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader, cpt_ids


def get_loader_emory(path, batch_size=1, num_workers=2, MAX_L=20, model_type='albert', n_classes=7, pin_memory=False, cuda_=False):
    trainset = EmoryNLPDataset(split='train', path=path, n_classes=n_classes, cuda=cuda_, model_type=model_type)
    cpt_ids = trainset.cpt_ids
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    validset = EmoryNLPDataset(split='valid', path=path, n_classes=n_classes, cuda=cuda_, model_type=model_type)
    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=validset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = EmoryNLPDataset(split='test', path=path, n_classes=n_classes, cuda=cuda_, model_type=model_type)
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

    def forward(self, pred, target, mask=None):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        if mask is None:
            loss = self.loss(pred, target)
            return loss

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


def train_or_eval_model(model, loss_Func, dataloader, epoch, optimizer=None, scheduler = None, model_type='albert',
                        chunk_size=10, data_type='meld', train=True, cuda_=False):
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

        if data_type == 'meld':
            sent_ids, mask, token_types, cpt_graph_i, _, speakers, umask, label, \
             str_src, str_dst, str_edge_type = \
                [d.cuda() if torch.is_tensor(d) else d for d in data[:-1]] if cuda_ else data[:-1]

        elif data_type == 'daily':
            sent_ids, mask, token_types, cpt_graph_i, speakers, umask, _, label, \
            str_src, str_dst, str_edge_type, _ = \
                [d.cuda() if torch.is_tensor(d) else d for d in data[:-1]] if cuda_ else data[:-1]

        elif data_type == 'emory':
            sent_ids, mask, token_types, cpt_graph_i, speakers, umask, label, \
            str_src, str_dst, str_edge_type = \
                [d.cuda() if torch.is_tensor(d) else d for d in data[:-1]] if cuda_ else data[:-1]

        else:
            print('data_type error!')

        cpt_graph_i = get_chunk(cpt_graph_i, model.cpt_ids, model_type=model.model_type, chunk_size=chunk_size)
        cpt_graph_i = [[item.cuda() if torch.is_tensor(item) else item for item in chunk]
                       for chunk in cpt_graph_i]if cuda_ else cpt_graph_i

        inputs = {'input_ids': sent_ids, 'attention_mask': mask, 'token_type_ids': token_types}
        labels_ = label.view(-1)

        log_prob, loss = model.forward(inputs, str_src, str_dst, str_edge_type, cpt_graph_i, labels_,
                                       loss_func=loss_Func, train=train)


        pred_ = torch.argmax(log_prob, 1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        # losses.append(loss)
        losses.append(loss * masks[-1].sum())
        if train:
            # with torch.autograd.detect_anomaly():
            # loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            scheduler.step()
            torch.cuda.empty_cache()

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)

    else:
        return float('nan'), float('nan'), [], [], [], float('nan'), []

    if data_type in ['meld', 'emory']:
        avg_loss = round(np.sum(losses) / np.sum(masks), 4)
        avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
        avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)
        class_report = classification_report(labels, preds, sample_weight=masks, digits=4)

        return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, class_report

    elif data_type == 'daily':
        avg_loss = round(np.sum(losses) / np.sum(masks), 4)
        avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
        pre_macro, rec_macro, fbeta_macro, _ = precision_recall_fscore_support(labels, preds, average='macro')
        pre_micro, rec_micro, fbeta_micro, _ = precision_recall_fscore_support(labels, preds, labels=[0, 2, 3, 4, 5, 6], average='micro')

        return avg_loss, avg_accuracy, labels, preds, masks, round(fbeta_macro, 4), round(fbeta_micro, 4)


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
    att_dropout = Configs.att_dropout
    max_sen_len = Configs.max_sen_len
    slide_win = Configs.slide_win
    D_m = 100

    if n_classes == 7:
        loss_weights = torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    if Configs.data_type == 'meld' and Configs.num_relations == 16:
        train_loader, valid_loader, test_loader, cpt_ids = \
            get_loader_meld('./data/meld/MELD.pkl', batch_size=Configs.batch_size, valid=Configs.valid,
                            num_workers=Configs.num_workers, MAX_L=max_sen_len, num_class=n_classes, model_type=Configs.model_type, cuda_=cuda_)

    elif Configs.data_type == 'meld' and Configs.num_relations == 10:
        train_loader, valid_loader, test_loader, cpt_ids = \
            get_loader_meld('./data/meld/MELD_revised.pkl', batch_size=Configs.batch_size, valid=Configs.valid,
                            num_workers=Configs.num_workers, MAX_L=max_sen_len, num_class=n_classes, model_type=Configs.model_type, cuda_=cuda_)

    elif Configs.data_type == 'daily' and Configs.num_relations == 16:
        train_loader, valid_loader, test_loader, cpt_ids = get_loader_daily('./data/dailydialog/Daily.pkl',
                                                                            batch_size=Configs.batch_size,
                                                                            num_workers=Configs.num_workers, model_type=Configs.model_type,
                                                                            cuda_=cuda_)
    elif Configs.data_type == 'daily' and Configs.num_relations == 11:
        train_loader, valid_loader, test_loader, cpt_ids = get_loader_daily('./data/dailydialog/Daily_revised.pkl',
                                                                            batch_size=Configs.batch_size,
                                                                            num_workers=Configs.num_workers,
                                                                            model_type=Configs.model_type,
                                                                            cuda_=cuda_)
    elif Configs.data_type == 'emory' and Configs.num_relations == 14:
        train_loader, valid_loader, test_loader, cpt_ids = get_loader_emory('./data/EMORY/EMORY.pkl',
                                                                            batch_size=Configs.batch_size,
                                                                            num_workers=Configs.num_workers,
                                                                            model_type=Configs.model_type,
                                                                            n_classes=n_classes,
                                                                            cuda_=cuda_)
    elif Configs.data_type == 'emory' and Configs.num_relations == 9:
        train_loader, valid_loader, test_loader, cpt_ids = get_loader_emory('./data/EMORY/EMORY_revised.pkl',
                                                                            batch_size=Configs.batch_size,
                                                                            num_workers=Configs.num_workers,
                                                                            model_type=Configs.model_type,
                                                                            n_classes=n_classes,
                                                                            cuda_=cuda_)

    else:
        raise ValueError("Please input a valid data type!")


    model = Model(cpt_ids, Configs=Configs, cuda_=cuda_)
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

    # optimizer = optim.Adam(model.parameters(), lr=Configs.lr, weight_decay=Configs.l2)

    def configure_optimizers(Configs):
        params = list(model.named_parameters())

        def is_backbone(n): return 'bert' in n

        grouped_parameters = [
            {"params": [p for n, p in params if is_backbone(n)], 'lr': Configs.base_lr},
            {"params": [p for n, p in params if not is_backbone(n)], 'lr': Configs.lr},
        ]

        optimizer = torch.optim.AdamW(
            grouped_parameters, lr=Configs.lr, weight_decay=Configs.l2
        )

        return optimizer

    optimizer = configure_optimizers(Configs=Configs)
    num_training_steps = len(train_loader) * (n_epochs+Configs.delta_epoch)
    num_warmup_steps = len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)


    if Configs.data_type == 'meld':
        best_va_fscore, best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None, None
        for e in range(Configs.epochs):
            start_time = time.time()
            print('base_lr {} lr {}'.format(optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))
            train_loss, train_acc, _, _, _, train_fscore, _, = train_or_eval_model(model, loss_Func=loss_function,
                                                                               dataloader=train_loader, epoch=e,
                                                                               optimizer=optimizer, scheduler=scheduler,
                                                                                model_type=Configs.model_type,
                                                                                chunk_size=Configs.chunk_size, data_type=Configs.data_type,
                                                                                train=True, cuda_=Configs.cuda)
            with torch.no_grad():
                valid_loss, valid_acc, _, _, _, valid_fscore, _, = train_or_eval_model(model, loss_Func=loss_function,
                                                                                       dataloader=valid_loader, epoch=e,
                                                                                       optimizer=optimizer,
                                                                                       scheduler=scheduler,
                                                                                       model_type=Configs.model_type,
                                                                                       chunk_size=Configs.chunk_size,
                                                                                       data_type=Configs.data_type,
                                                                                       train=False, cuda_=Configs.cuda)
            with torch.no_grad():
                test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, test_class_report = \
                                                                                train_or_eval_model(model, loss_Func=loss_function,
                                                                                dataloader=test_loader, epoch=e,
                                                                                model_type=Configs.model_type,
                                                                                chunk_size=Configs.chunk_size,
                                                                                data_type=Configs.data_type,
                                                                                train=False, cuda_=Configs.cuda)
            if best_va_fscore == None or best_va_fscore < valid_fscore:
                if train_fscore>=valid_fscore:
                    best_va_fscore, best_fscore, best_loss, best_label, best_pred, best_mask = \
                        valid_fscore, test_fscore, test_loss, test_label, test_pred, test_mask
                # state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': e}
                # torch.save(state, Configs.model_path + 'model_5_2.pth')

            print('epoch {} train_loss {} train_acc {} train_fscore {} va_loss {} va_acc {} va_fscore {} test_loss {} test_acc {} test_fscore {} time {}'. \
                    format(e + 1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss,
                           test_acc, test_fscore, round(time.time() - start_time, 2)))
        # print(test_class_report)

        print('Test performance..')
        print('Fscore {} accuracy {}'.format(best_fscore,
                                             round(accuracy_score(best_label, best_pred, sample_weight=best_mask) * 100,
                                                   2)))
        print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
        print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))


    elif Configs.data_type == 'daily':
        localtime = time.localtime(time.time())
        newtime = time.asctime(localtime)
        log_path = './log'+ '/train_dd' + newtime + '.log'
        logger = config_logger(log_path)
        best_va_fscore, best_fscore, best_fmacro, best_fmicro, best_loss, best_label, best_pred, best_mask = None, None, 0, 0, None, None, None, None
        for e in range(Configs.epochs):
            start_time = time.time()
            train_loss, train_acc, _, _, _, train_fmacro, train_fmicro = train_or_eval_model(model, loss_Func=loss_function,
                                                                                   dataloader=train_loader, epoch=e,
                                                                                   optimizer=optimizer,
                                                                                   scheduler=scheduler,
                                                                                   model_type=Configs.model_type,
                                                                                   chunk_size=Configs.chunk_size,
                                                                                   data_type=Configs.data_type,
                                                                                   train=True, cuda_=Configs.cuda)
            with torch.no_grad():
                valid_loss, valid_acc, _, _, _, valid_fmacro, valid_fmicro = train_or_eval_model(model,
                                                                                                 loss_Func=loss_function,
                                                                                                 dataloader=valid_loader,
                                                                                                 epoch=e,
                                                                                                 optimizer=optimizer,
                                                                                                 scheduler=scheduler,
                                                                                                 model_type=Configs.model_type,
                                                                                                 chunk_size=Configs.chunk_size,
                                                                                                 data_type=Configs.data_type,
                                                                                                 train=False,
                                                                                                 cuda_=Configs.cuda)
            with torch.no_grad():
                test_loss, test_acc, test_label, test_pred, test_mask, test_fmacro, test_fmicro = \
                    train_or_eval_model(model, loss_Func=loss_function,
                                        dataloader=test_loader, epoch=e,
                                        model_type=Configs.model_type,
                                        chunk_size=Configs.chunk_size,
                                        data_type=Configs.data_type,
                                        train=False, cuda_=Configs.cuda)
            train_fscore = train_fmacro + train_fmicro
            valid_fscore = valid_fmacro + valid_fmicro
            test_fscore = test_fmacro + test_fmicro
            if best_va_fscore == None or best_va_fscore < valid_fscore:
                if train_fscore >= valid_fscore:
                    best_va_fscore, best_fscore, best_fmacro, best_fmicro, best_loss, best_label, best_pred, best_mask = \
                        valid_fscore,test_fscore, test_fmacro, test_fmicro, test_loss, test_label, test_pred, test_mask
                # state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': e}
                # torch.save(state, Configs.model_path + 'model_5_2.pth')

            logger.info('epoch {} train_loss {} train_acc {} train_fmacro {} train_fmicro {} valid_loss {} valid_acc {} valid_fmacro {} valid_fmicro {} test_loss {} test_acc {} test_fmacro {} test_fmicro {} time {}'. \
                    format(e + 1, train_loss, train_acc, train_fmacro, train_fmicro, valid_loss, valid_acc, valid_fmacro, valid_fmicro, test_loss,
                           test_acc, test_fmacro, test_fmicro, round(time.time() - start_time, 2)))


        logger.info('Test performance..')
        logger.info('Fmacro {} Fmicro {} accuracy {}'.format(best_fmacro, best_fmicro, round(accuracy_score(best_label, best_pred, sample_weight=best_mask) * 100,
                                                   2)))


    elif Configs.data_type == 'emory':
        best_va_fscore, best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None, None
        for e in range(Configs.epochs):
            start_time = time.time()
            train_loss, train_acc, _, _, _, train_fscore, _, = train_or_eval_model(model, loss_Func=loss_function,
                                                                                   dataloader=train_loader, epoch=e,
                                                                                   optimizer=optimizer,
                                                                                   scheduler=scheduler,
                                                                                   model_type=Configs.model_type,
                                                                                   chunk_size=Configs.chunk_size,
                                                                                   data_type=Configs.data_type,
                                                                                   train=True, cuda_=Configs.cuda)

            with torch.no_grad():
                valid_loss, valid_acc, _, _, _, valid_fscore, _ = train_or_eval_model(model, loss_Func=loss_function,
                                                                                      dataloader=valid_loader, epoch=e,
                                                                                      optimizer=optimizer,
                                                                                      scheduler=scheduler,
                                                                                      model_type=Configs.model_type,
                                                                                      chunk_size=Configs.chunk_size,
                                                                                      data_type=Configs.data_type,
                                                                                      train=False, cuda_=Configs.cuda)

            with torch.no_grad():
                test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, test_class_report = \
                    train_or_eval_model(model, loss_Func=loss_function,
                                        dataloader=test_loader, epoch=e,
                                        model_type=Configs.model_type,
                                        chunk_size=Configs.chunk_size,
                                        data_type=Configs.data_type,
                                        train=False, cuda_=Configs.cuda)

            if best_va_fscore == None or best_va_fscore < valid_fscore:
                if train_fscore>=valid_fscore:
                    best_va_fscore, best_fscore, best_loss, best_label, best_pred, best_mask = \
                        valid_fscore, test_fscore, test_loss, test_label, test_pred, test_mask

                # state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': e}
                # torch.save(state, Configs.model_path + 'model_5_2.pth')

            print(
                'epoch {} train_loss {} train_acc {} train_fscore {} valid_loss {} valid_acc {} valid_fscore {} test_loss {} test_acc {} test_fscore {} time {}'. \
                format(e + 1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss,
                       test_acc, test_fscore, round(time.time() - start_time, 2)))
        # print(test_class_report)

        print('Test performance..')
        print('Fscore {} accuracy {}'.format(best_fscore,
                                             round(accuracy_score(best_label, best_pred, sample_weight=best_mask) * 100,
                                                   2)))
        print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
        print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))