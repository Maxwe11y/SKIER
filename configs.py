'''
Author: Li Wei
Email: wei008@e.ntu.edu.sg
'''

import argparse


def inputconfig_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True, help='use GPU acceleration or not')
    parser.add_argument('--lr', type=float, default=0.00005, metavar='LR', help='learning rate')
    parser.add_argument('--base_lr', type=float, default=0.000005, metavar='BLR', help='learning rate for base model')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--valid', type=float, default=0.1, metavar='va', help='valid for MELD dataset')
    parser.add_argument('--dropout', type=float, default=0.20, metavar='dropout', help='dropout rate')
    parser.add_argument('--att_dropout', type=float, default=0.10, metavar='attdropout', help='dropout rate for '
                                                                                              'relative attention')
    parser.add_argument('--batch-size', type=int, default=1, metavar='BS', help='batch size')
    parser.add_argument('--chunk_size', type=int, default=4, metavar='CS', help='chunk size')
    parser.add_argument('--epochs', type=int, default=10, metavar='E', help='number of epochs')
    parser.add_argument('--input_dim', type=int, default=100, metavar='D', help='input dimension')
    parser.add_argument('--output_dim', type=int, default=768, metavar='O', help='output dimension of pretrained model')
    parser.add_argument('--num_workers', type=int, default=2, metavar='NW', help='number of workers in '
                                                                                 'Dataloader function')
    parser.add_argument('--num_class', type=int, default=7, metavar='N', help='number of sentiment classes')
    parser.add_argument('--num_relations', type=int, default=16, metavar='NR', help='number of dialog parsing relations')
    parser.add_argument('--class-weight', action='store_true', default=False, help='class weight')
    parser.add_argument('--activation', type=str, default='sigmoid', help='activation function')
    parser.add_argument('--data_type', type=str, default='meld', help='whether use meld or dailydialog')
    parser.add_argument('--model_type', type=str, default='albert', help='pretrained_model_type')
    parser.add_argument('--max_sen_len', type=int, default=30, help='max sentence length')
    parser.add_argument('--slide_win', type=int, default=2, help='size of the sliding window')
    parser.add_argument('--num_head', type=int, default=8, help='number of head in CoAtt')
    parser.add_argument('--num_bases', type=int, default=2, help='number of bases of RGCN')
    parser.add_argument('--lamb', type=float, default=0.5, help='a trade-off hyperparameter')
    parser.add_argument('--num_features', type=int, default=4, help='number of features used in the model')
    parser.add_argument('--use_future_utt', action='store_true', default=False, help='use future utterances or not')
    parser.add_argument('--use_dot_att', action='store_true', default=False, help='use dot attention or item attention')
    parser.add_argument('--src_num', type=int, default=4, help='number of words for source sentence')
    parser.add_argument('--dst_num_per_rel', type=int, default=2, help='number of destination words per '
                                                                       'relation per src word')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')

    return parser.parse_args()