import os
import argparse
import torch
import torch.nn as nn
import _pickle as cPickle
from torch.utils.data import DataLoader
import numpy as np

import base_model_I2Q2Q as base_model
#from dataset_attmap import DictionaryAll, VQAFeatureDatasetAll
#from train_attmap import train
from dataset_attmap import DictionaryAll, VQAFeatureDatasetAll
from rev_attmap import train

##from train_ques_acc import train

#import base_model_combine as base_model
#import base_model_selfatt as base_model
#import base_model as base_model
#from dataset import Dictionary,VQAFeatureDataset
# import base_I2Q_tfidf as base_model

##from dataset_qtype import DictionaryAll, VQAFeatureDatasetAll

# from train_ques_acc import train
#from dataset_100_bb import DictionaryAll,VQAFeatureDatasetAll
#from dataset_qtype_bert import DictionaryAll,VQAFeatureDatasetAll
#from train_lsp import train
#from train_I2Q_OTM import train
#from train_I2Q import train,evaluate
#from train_cca import train
#from train_ques_acc import train
#from train_final_je import train,evaluate
#from je_train_binary import train,evaluate
#from train_ques_selfatt import train,evaluate
#from dataset_ques import Dictionary,VQAFeatureDatasetQues
#from train_bert import train,evaluate

from torch.autograd import Variable
import torch.nn.functional as F
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=2048)
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

    #question_type= ['absurd','activity_recognition']
    num_task_train = []
    num_task_val = []

    for q_type in question_type:
        num_ans = len(cPickle.load(open('../../tools/data/cache/trainval_ans2label_' + q_type + '.pkl','rb')))
#        num_ans = len(cPickle.load(open('../tools/data/cache/trainval_ans2label_' + q_type + '.pkl','rb')))
        print("%s -------> %d" %(q_type,num_ans))
        constructor = 'build_%s' % args.model
        model = getattr(base_model, constructor)(9318, num_ans, args.num_hid).cuda()
#        Model[question_type.index(q_type)] = model.classifier
        Model[question_type.index(q_type)] = torch.load('result/lsp_emb_emb_mltmodel.pth')[question_type.index(q_type)]

    total_param = 0
    for q_type in question_type:
        for p in Model[question_type.index(q_type)].parameters():
                pytorch_1 = p.numel()
                print(pytorch_1)
                total_param += pytorch_1

    print("Total each ques classifier:",total_param)


    constructor = 'build_baseline0_ques'
    model_ques = getattr(base_model, constructor)(9318, 12, args.num_hid).cuda()
    pytorch_1 = sum(p.numel() for p in model_ques.parameters())
    print("Total for final classifier:",pytorch_1)

    #model_ques.w_emb.init_embedding('../tools/data/glove6b_init_300d.npy'
    #print("---------------------------------- Joint Representation Model ----------------------------\n")
#    for name, param in model.named_parameters():
#        if param.requires_grad:
#                print(name," ----> ",param.data.shape)

    #print("---------------------------------- Question Representation Model ----------------------------\n")

    #for name, param in model_ques.named_parameters():
    #    if param.requires_grad:
    #            print(name,"---->",param.data.shape)


    constructor = 'build_baseline0_jr'
#    constructor = 'build_san'
    model = getattr(base_model, constructor)(9318, 1616, args.num_hid).cuda()
    model.w_emb.init_embedding('../../tools/data/glove6b_init_300d.npy')

    args.model = 'rev_newatt'
    constructor = 'build_%s' % args.model
    model_rev = getattr(base_model, constructor)(9318, 1616, args.num_hid).cuda()
    model_rev.w_emb.init_embedding('../../tools/data/glove6b_init_300d.npy')

#    model = nn.DataParallel(model).cuda()
 #   model_rev = nn.DataParallel(model_rev).cuda()

    for name, param in model_ques.named_parameters():
        if param.requires_grad:
                print(name," ----> ",param.data.shape)

    train_dset = VQAFeatureDatasetAll('train', dictionary)
    eval_dset = VQAFeatureDatasetAll('val', dictionary)

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=4)
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=4)

    train(model_ques, model_rev, model, Model, train_loader, eval_loader, args.epochs, args.output, num_task_train, num_task_val)

