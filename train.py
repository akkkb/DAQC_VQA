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
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def compute_score_with_logits_inclass(logits, labels, type_, m):
    logits = torch.max(logits, 1)[1].data
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    match = (type_ == m)
    match = match.cuda()
    scores = scores[match,:]
    labels = labels[match,:]
    return scores.sum(), labels.sum()

def init_dict(names_list):
    out_dict = {}
    for nm in names_list:
        out_dict[nm] = [0,0]
    return out_dict

def train(ques_cat, ques_att, img_att, Model, train_loader, eval_loader, num_epochs, output):

    question_type= ['absurd','activity_recognition','attribute','color','counting','object_presence','object_recognition','positional_reasoning',
    'scene_recognition','sentiment_understanding','sport_recognition','utility_affordance']

    utils.create_dir(output)
    ls_params = []

    for i in range(len(question_type)):
        ls_params += list(Model[i].parameters())

    ls_params += listques_cat.parameters())
    ls_params += list(ques_att.parameters())
    ls_params += list(img_att.parameters())

    filename = 'result'
    optim = torch.optim.Adamax(ls_params)
    logger = utils.Logger(os.path.join(output, '%s.txt' %(filename)))
    best_eval_score = 0
    accuracy = []
    train_loss = []

    with open('data/index_map.pkl',"rb") as map_fl:
        class_index_map = cPickle.load(map_fl)

    for epoch in range(num_epochs):
        t = time.time()
        print("\n-----------------------------> Epoch is %d <------------------------------" %(epoch))
        train_score = init_dict(question_type)
        train_score_ques = 0
        total_loss = 0

        for i, (v, q, a, type_) in enumerate(train_loader):
            v = Variable(v).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()

            type_onehot = _to_one_hot(type_, len(question_type) , dtype=torch.cuda.FloatTensor)
            v_feat, att_v = model(v, q, a)
            q_feat, att_q = model_rev(v_feat, q, a)
            inp_emb = v_feat * q_feat
            ques_pred = model_ques(inp_emb, type_onehot)
            loss_ques = instance_bce_with_logits(ques_pred, type_onehot)
  
            ques_cat_ind = F.softmax(ques_pred, dim=1)
            ques_cat_ind = Variable(ques_cat_ind).cuda()
            max_indx = torch.argmax(ques_cat_ind, dim=1)
            one_hot = _to_one_hot(max_indx, len(question_type), dtype=torch.cuda.FloatTensor)

            loss_mod = {}
            ans_gt = {}
            pred = {}

            for m in Model:
                pred[m] = Model[m](one_hot[:,m].unsqueeze(1) * inp_emb)
                tmp_indx = torch.tensor(class_index_map[question_type[m]])
                tmp_indx = Variable(tmp_indx).cuda()
                tmp_tnst = torch.index_select(a, 1, tmp_indx)
                ans_gt[m] = Variable(tmp_tnst).cuda()
                temp_tensor = (type_ == m)
                temp_tensor = temp_tensor.cpu().data.numpy()
                loss_mod[m] = instance_bce_with_logits(pred[m], ans_gt[m])

            loss = sum(loss_mod.values())/12 +  loss_ques
            loss.backward()
            optim.step()
            optim.zero_grad()
            print("\r[epoch %2d][step %4d/%4d] loss: %.4f" % ( epoch + 1, i, int( len(train_loader.dataset) / a.shape[0]), loss.cpu().data.numpy()), end='          ')

            score_ques = compute_score_with_logits(ques_pred, type_onehot.data).sum()
            train_score_ques += score_ques

            total_loss += loss.data.item() * v.size(0)
            for m in Model:
                bs_p, bs_a = compute_score_with_logits_inclass(pred[m], ans_gt[m].data, type_, m)
                train_score[question_type[m]][0] += bs_p
                train_score[question_type[m]][1] += bs_a

        avg_accuracy = 0
        count = 0
        num = 0
        den = 0

        ques_train_score = 100 * train_score_ques / len(train_loader.dataset)

        logger.write('\nepoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\tQuestion train score: %.2f' %(ques_train_score))

        for qtp in question_type:
                num += train_score[qtp][0]
                den += train_score[qtp][1]

        avg_accuracy = num/den
        logger.write('\ttrain score: %.2f' % (avg_accuracy * 100))
        for qtp in question_type:
                num += train_score[qtp][0]

        avg_accuracy = num/den

        train_ls = []
        for i in question_type:
                class_accuracy = train_score[i][0]/train_score[i][1]
                train_ls.append(class_accuracy)
                logger.write('\t%s is\t\t %d / %d  -----> %.2f' %(i, train_score[i][0],train_score[i][1],100 * class_accuracy))

        train_loss.append(total_loss)
        accuracy.append(train_score)

        for m in Model:
            Model[m].train(False)

        model.train(False)
        model_ques.train(False)
        model_rev.train(False)

        ques_score, eval_score, eval_ls = evaluate(ques_cat, ques_att, img_att, Model, eval_loader)

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


        logger.write('\n\teval score: \t\t ----->  %.2f' % (eval_score * 100))

        if eval_score > best_eval_score:
                epc = epoch
                model_path = os.path.join(output, '%s_img_att.pth' %(filename))
                torch.save(img_att, model_path)
                model_path = os.path.join(output, '%s_ans_pred.pth' %(filename))
                torch.save(Model, model_path)
                model_path = os.path.join(output, '%s_ques_att.pth' %(filename))
                torch.save(ques_att, model_path)
                model_path = os.path.join(output, '%s_ques_cat.pth' %(filename))
                torch.save(ques_cat, model_path)
                best_eval_score = eval_score

    logger.write('\n\t Best Evaluation score is obtained at epoch %d : %.2f' %(epc,best_eval_score * 100))


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

