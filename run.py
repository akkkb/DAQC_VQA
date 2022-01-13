import os
import argparse
import torch
import torch.nn as nn
import _pickle as cPickle
from torch.utils.data import DataLoader
import numpy as np
import model as model
from torch.autograd import Variable
import torch.nn.functional as F
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--output', type=str, default='result')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    batch_size = 256
    dictionary = DictionaryAll.load_from_file('../../tools/data/dictionary.pkl')
    Model = {}

    constructor = 'build_%s' % args.model
    question_type= ['absurd','activity_recognition','attribute','color','counting',
    'object_presence','object_recognition','positional_reasoning',
    'scene_recognition','sentiment_understanding','sport_recognition','utility_affordance']

    num_task_train = []
    num_task_val = []

    for q_type in question_type:
        num_ans = len(cPickle.load(open('data/trainval_ans2label_' + q_type + '.pkl','rb')))
        constructor = 'build_%s' % args.model
        model = getattr(base_model, constructor)(len(dictionary), num_ans, args.num_hid).cuda()
        Model[question_type.index(q_type)] = model

    constructor = 'build_baseline0_ques'
    model_ques = getattr(base_model, constructor)(len(dictionary), len(question_type), args.num_hid).cuda()
  
    constructor = 'build_baseline0_jr'
    model = getattr(base_model, constructor)(len(dictionary), args.num_hid).cuda()
    model.w_emb.init_embedding('data/glove6b_init_300d.npy')

    args.model = 'rev_newatt'
    constructor = 'build_%s' % args.model
    model_rev = getattr(base_model, constructor)(9318, args.num_hid).cuda()
    model_rev.w_emb.init_embedding('data/glove6b_init_300d.npy')

    train_dset = VQAFeatureDatasetAll('train', dictionary)
    eval_dset = VQAFeatureDatasetAll('val', dictionary)

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=4)
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=4)

    train(model_ques, model_rev, model, Model, train_loader, eval_loader, args.epochs, args.output, num_task_train, num_task_val)

