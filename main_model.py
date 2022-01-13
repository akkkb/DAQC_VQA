import torch
import torch.nn as nn
from attention import Attention, ImgAttention, QuesAttention
from language_model import WordEmbedding, QuestionEmbedding, WordEmbeddingNew
from classifier import SimpleClassifier,distLinear
from fc import FCNet

class ImgAtt(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net):
        super(ImgAtt, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.q_net = q_net
        self.v_net = v_net
        self.v_att = v_att
        
    def forward(self, v, q, labels):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]
        att_v = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]
        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        return v_repr, att_v

class QuesAtt(nn.Module):
    def __init__(self, w_emb, q_emb, q_att, q_net):
        super(QuesAtt, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.q_att = q_att
        self.q_net = q_net

    def forward(self, v, q, labels):
        w_emb = self.w_emb(q)
        att_q = self.q_att(w_emb, v)
        w = (att * w_emb) # [batch, v_dim]
        q_emb = self.q_emb(w) 
        q_repr = self.q_net(q_emb)
        return q_repr, att_q

class QuesCat(nn.Module):
    def __init__(self, classifier):
        super(QuesCat, self).__init__()
        self.classifier = classifier

    def forward(self, q, labels):
        logits = self.classifier(q)
        return logits

class AnsPred(nn.Module):
    def __init__(self, classifier):
        super(AnsPred, self).__init__()
        self.classifier = classifier

    def forward(self, emb):
        ans = self.classifier(emb)

def ques_cat(dataset, num_hid):
    classifier = SimpleClassifier(num_hid, int(num_hid / 2), 12, 0.5)
    return QuesCat(w_emb, q_emb, net, classifier)

def image_att(dataset, num_hid):
    w_emb = WordEmbedding(dataset, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([num_hid, num_hid])
    return ImgAtt(w_emb, q_emb, v_att, q_net, v_net)

def ques_att(dataset, num_hid):
    w_emb = WordEmbedding(dataset, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention_rev(300, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([num_hid, num_hid])
    return QuesAtt(w_emb, q_emb, v_att, q_net)

def build_qus(dataset, num_ans_candidates, num_hid):
    classifier = SimpleClassifier(num_hid, num_hid, num_ans_candidates, 0.5)
    return AnsPred(classifier)
