import os
import argparse
import torch
import torch.nn as nn
import _pickle as cPickle
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import time
from data_prep import DictionaryAll,VQAFeatureDatasetAll
import model as model
from train import train

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='ans_pred')
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
    dictionary = DictionaryAll.load_from_file('data/dictionary.pkl')
    Model = {}
    
    question_type= ['absurd','activity_recognition','attribute','color','counting',
    'object_presence','object_recognition','positional_reasoning',
    'scene_recognition','sentiment_understanding','sport_recognition','utility_affordance']

    num_task_train = []
    num_task_val = []

    for q_type in question_type:
        num_ans = len(cPickle.load(open('data/trainval_ans2label_' + q_type + '.pkl','rb')))
        constructor = '%s' % args.model
        model = getattr(base_model, constructor)(len(dictionary), num_ans, args.num_hid).cuda()
        Model[question_type.index(q_type)] = model

    constructor = 'ques_cat'
    ques_cat = getattr(base_model, constructor)(len(dictionary), len(question_type), args.num_hid).cuda()
  
    constructor = 'img_att'
    img_att = getattr(base_model, constructor)(len(dictionary), args.num_hid).cuda()
    img_att.w_emb.init_embedding('data/glove6b_init_300d.npy')

    constructor = 'ques_att'
    ques_att = getattr(base_model, constructor)(9318, args.num_hid).cuda()
    ques_att.w_emb.init_embedding('data/glove6b_init_300d.npy')

    train_dset = VQAFeatureDatasetAll('train', dictionary)
    eval_dset = VQAFeatureDatasetAll('val', dictionary)

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=4)
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=4)

    train(ques_cat, ques_att, img_att, Model, train_loader, eval_loader, args.epochs, args.output)

