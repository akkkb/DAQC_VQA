import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from torch.autograd import Variable
import _pickle as cPickle
import numpy as np
import statistics
from torch.optim.lr_scheduler import MultiStepLR

torch.set_printoptions(threshold=10000)

def _to_one_hot(y, n_dims, dtype=torch.cuda.FloatTensor):
    scatter_dim = len(y.size())
    y_tensor = y.type(torch.cuda.LongTensor).view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), n_dims).type(dtype)
    return zeros.scatter(scatter_dim, y_tensor, 1)

def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss

def compute_score_with_logits(logits, labels):
    #print(logits.shape)
    logits = torch.max(logits, 1)[1].data # argmax
    #print(logits.shape,*labels.size())
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    #print(scores)
    return scores

def compute_score_with_logits_inclass(logits, labels, type_, m):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    temp_tensor = (type_ == m)
    temp_tensor = temp_tensor.cuda()
    scores = scores[temp_tensor,:]
    labels = labels[temp_tensor,:]
#    print("m is -----> ",m,"and sum is",scores.sum())
    return scores.sum(), labels.sum()

def init_dict(names_list):
    out_dict = {}
    for nm in names_list:
        out_dict[nm] = [0,0]
    return out_dict

def train(model_ques, model_rev, model, Model, train_loader, eval_loader, num_epochs, output, num_task_train, num_task_val):

    utils.create_dir(output)
    ls_params = []

    for i in range(12):
        ls_params += list(Model[i].parameters())

    ls_params += list(model.parameters())
    ls_params += list(model_ques.parameters())
    ls_params += list(model_rev.parameters())

    filename = 'emb_embatt'
    optim = torch.optim.Adamax(ls_params)
    scheduler1 = MultiStepLR(optim, milestones=[5,7,9,11,15], gamma=0.5)

    count_array = torch.cuda.FloatTensor([166882,  57606, 219269])
    max, _ = torch.max(count_array,0)
    for i in range(count_array.shape[0]):
        count_array[i] = max / count_array[i]


    logger = utils.Logger(os.path.join(output, '%s.txt' %(filename)))

    best_eval_score = 0
    accuracy = []
    train_loss = []

    with open('../../tools/data/cache_qs/class_index_map.pkl',"rb") as map_fl:
        class_index_map = cPickle.load(map_fl)

    question_type= ['absurd','activity_recognition','attribute','color','counting','object_presence','object_recognition','positional_reasoning',
    'scene_recognition','sentiment_understanding','sport_recognition','utility_affordance']


#    count_array = torch.cuda.FloatTensor([246243, 5848, 19476, 133074, 111857, 441810, 62862, 26042, 44674, 1461, 21602, 350])
    #print(count_array.shape[0])
    max, _ = torch.max(count_array,0)
    for i in range(count_array.shape[0]):
        count_array[i] = max / count_array[i]

    max_mean = 0
    for epoch in range(num_epochs):
        t = time.time()
        print("\n-----------------------------> Epoch is %d <------------------------------" %(epoch))
        train_score = init_dict(question_type)
        train_score_ques = 0
        total_loss = 0
        count_0 = 0
        count_1 = 0
        count_2 = 0

        for i, (v, q, a, type_) in enumerate(train_loader):
            v = Variable(v).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()

            type_onehot = _to_one_hot(type_, 12 , dtype=torch.cuda.FloatTensor)
            v_feat, emb = model(v,q,a)
            q_feat, att = model_rev(v_feat, q, a)
            emb_att = v_feat * q_feat
            ques_pred = model_ques(emb, type_onehot)
#            ques_pred = model_ques(q_feat, type_onehot)
            ss = F.softmax(ques_pred, dim=1)

            loss_ques = instance_bce_with_logits(ques_pred, type_onehot)
            ss = Variable(ss).cuda()
            max_indx = torch.argmax(ss,dim=1)
            one_hot = _to_one_hot(max_indx, 12 , dtype=torch.cuda.FloatTensor)

            for i_type in range(type_.shape[0]):
                if type_[i_type] == 0:
                        count_0 += 1
                if type_[i_type] == 1:
                        count_1 += 1
                if type_[i_type] == 2:
                        count_2 += 1

            loss_d = {}
            a_gt_tensor = {}
            pred = {}

#            emb = v_feat * q_feat
            for m in Model:
                pred[m] = Model[m](one_hot[:,m].unsqueeze(1) * emb_att)
                tmp_indx = torch.tensor(class_index_map[question_type[m]])
                tmp_indx = Variable(tmp_indx).cuda()
                tmp_tnst = torch.index_select(a,1,tmp_indx)

                a_gt_tensor[m] = Variable(tmp_tnst).cuda()

                temp_tensor = (type_ == m)
                temp_tensor = temp_tensor.cpu().data.numpy()
                count_tem_tns = 0
                loss_d[m] = instance_bce_with_logits(pred[m], a_gt_tensor[m])

#            for m in range(3):
#                loss_d[m] = count_array[m] * loss_d[m]


            loss = sum(loss_d.values())/12 +  loss_ques
            loss.backward()
            optim.step()
            optim.zero_grad()

            print("\r[epoch %2d][step %4d/%4d] loss: %.4f,loss_ques: %.4f" % ( epoch + 1, i, int( len(train_loader.dataset) / a.shape[0]), loss.cpu().data.numpy(), loss_ques ), end='          ')

            score_ques = compute_score_with_logits(ques_pred, type_onehot.data).sum()
            train_score_ques += score_ques

            total_loss += loss.data.item() * v.size(0)
            count_lab = 0
            for m in Model:
                bs_p, bs_a = compute_score_with_logits_inclass(pred[m], a_gt_tensor[m].data, type_, m)
                train_score[question_type[m]][0] += bs_p
                train_score[question_type[m]][1] += bs_a

        avg_accuracy = 0
        count = 0
        num = 0
        den = 0

#        scheduler1.step()
        ques_train_score = 100 * train_score_ques / len(train_loader.dataset)

        logger.write('\nepoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\tQuestion train score: %.2f' %(ques_train_score))

        for qtp in question_type:
                num += train_score[qtp][0]
                den += train_score[qtp][1]

        print("\nPredicted samples are: %d" %(num))
        print("Total sample are: %d" %(den))

        avg_accuracy = num/den
        logger.write('\ttrain score: %.2f' % (avg_accuracy * 100))
        for qtp in question_type:
                num += train_score[qtp][0]
       #         den += train_score[qtp][1]

        den = count_0 + count_1 + count_2
        avg_accuracy = num/den

        #print("\n************** Classwise Accuracy *************************\n ")

        train_ls = []
        for i in question_type:
                class_accuracy = train_score[i][0]/train_score[i][1]
                train_ls.append(class_accuracy)
                logger.write('\t%s is\t\t %d / %d  -----> %.2f' %(i,train_score[i][0],train_score[i][1],100 * class_accuracy))

        train_loss.append(total_loss)
        accuracy.append(train_score)

        for m in Model:
            Model[m].train(False)

        model.train(False)
        model_ques.train(False)
        model_rev.train(False)

        ques_score, eval_score, eval_ls = evaluate(model_ques, model_rev, model, Model, eval_loader)

        logger.write('\n\tQuestion eval score: %.2f' %(ques_score))
        print("\n")

        for qtp in question_type:
                count = eval_ls[qtp][0]/eval_ls[qtp][1]
                logger.write('\t%s is: \t\t %d / %d  -----> %.2f'%(qtp, eval_ls[qtp][0], eval_ls[qtp][1], count*100))

        for m in Model:
                Model[m].train(True)

        model.train(True)
        model_ques.train(True)
        model_rev.train(True)

        mean_array = 0
        mean_list = []
        #count_mean = 0
        for ma in question_type:
                count_mean = eval_ls[ma][0]/eval_ls[ma][1]
                count_mean = count_mean * 100
                mean_array = mean_array + count_mean
                mean_list.append(count_mean.data)

        logger.write('\n\teval score: \t\t ----->  %.2f' % (eval_score * 100))
        logger.write('\tArithmetic Mean is ----> %.2f' %(mean_array/12))


        logger.write('\n\teval score: \t\t ----->  %.2f' % (eval_score * 100))

        logger.write('\n\tLoss of Question classifier is: %.5f' %(loss_ques.data))
        logger.write('\t Main classifier loss  is: %.5f' %(sum(loss_d.values())/12))

        if eval_score > best_eval_score:
                epc = epoch
                model_path = os.path.join(output, '%s_jr.pth' %(filename))
                torch.save(model, model_path)
                model_path = os.path.join(output, '%s_mltmodel.pth' %(filename))
                torch.save(Model, model_path)
                model_path = os.path.join(output, '%s_quesmodel.pth' %(filename))
                torch.save(model_ques, model_path)
                best_eval_score = eval_score

        if mean_array/12 > max_mean: 
                max_mean = mean_array/12
                epc_mean = epoch

    logger.write('\n\t Best Evaluation score is obtained at epoch %d : %.2f and best AM is obtained at %d is %.2f' %(epc,best_eval_score * 100, epc_mean, max_mean))


def evaluate(model_ques, model_rev, model, Model,dataloader):
    score = 0
    upper_bound = 0
    num_data = 0

    with open('../../tools/data/cache_qs/class_index_map.pkl',"rb") as map_fl:
        class_index_map = cPickle.load(map_fl)

    question_type= ['absurd','activity_recognition','attribute','color','counting', 'object_presence','object_recognition','positional_reasoning',
    'scene_recognition','sentiment_understanding','sport_recognition','utility_affordance']


    val_score = init_dict(question_type)
    ques_score = 0

    count_0 = 0
    count_1 = 0
    count_2 = 0

    for v, q, a, type_ in iter(dataloader):
        with torch.no_grad():
            v = v.cuda()
            a = a.cuda()
            q = q.cuda()
            type_onehot = _to_one_hot(type_, 12 , dtype=torch.cuda.FloatTensor)

        v_feat, emb = model(v,q,None)
        q_feat, att = model_rev(v_feat, q, None)
        emb_att = v_feat * q_feat
        ques_pred = model_ques(emb, None)

#        ques_pred = model_ques(q_feat, None)
        ss = F.softmax(ques_pred, dim=1)
        max_indx = torch.argmax(ss,dim=1)
        one_hot = _to_one_hot(max_indx, 12 , dtype=torch.cuda.FloatTensor)

        batch_score = compute_score_with_logits(ques_pred, type_onehot).sum()
        ques_score += batch_score

        a_gt_tensor = {}
        pred = {}

        ind = torch.argmax(ss,dim=1)

#        emb = v_feat * q_feat
        for m in Model:
            pred[m] = Model[m](one_hot[:,m].unsqueeze(1) * emb_att)
            tmp_indx = torch.tensor(class_index_map[question_type[m]])
            tmp_indx = Variable(tmp_indx).cuda()
            tmp_tnst = torch.index_select(a,1,tmp_indx)
            a_gt_tensor[m] = Variable(tmp_tnst).cuda()
            bs_p, bs_a = compute_score_with_logits_inclass(pred[m], a_gt_tensor[m].data, type_, m)
            val_score[question_type[m]][0] += bs_p
            val_score[question_type[m]][1] += bs_a

    num = 0
    den = 0

    for itm in val_score:
        num += val_score[itm][0]
        den += val_score[itm][1]

    print("\n\tEval Predicted is: %d" %(num))
    print("\n\tEval Actual is: %d" %(den))

    ques_score = ques_score / len(dataloader.dataset)
    ques_score = ques_score * 100
    score = num / den

    return ques_score,score,val_score

