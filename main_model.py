import torch
import torch.nn as nn
from attention import Attention, NewAttention, NewAttention_rev, StackedAttention
from language_model import WordEmbedding, QuestionEmbedding, WordEmbeddingNew
from classifier import SimpleClassifier,distLinear
from fc import FCNet

class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb,  q_net, v_net, classifier):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

    def forward(self, v, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]
        v_emb =  v # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        #print(joint_repr.shape)
        #logits = self.classifier(joint_repr)
        return joint_repr

class BaseModel_jr(nn.Module):
    #def __init__(self, w_emb, q_emb,  q_net, v_net):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net):

        super(BaseModel_jr, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net

    def forward(self, v, q, labels):
        """Forward
        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]
        #v_emb =  v # [batch, v_dim]
        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1)
        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
##        return v_repr, joint_repr
        return v_repr, att, joint_repr

class BaseModel_ques(nn.Module):
    def __init__(self, w_emb, q_emb, net, classifier):
        super(BaseModel_ques, self).__init__()
        self.classifier = classifier
        self.net = net
        self.w_emb = w_emb
        self.q_emb = q_emb

    def forward(self, q, labels):
        """Forward
        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]
        return: logits, not probs
        """
#        w_emb = self.w_emb(q)
#        q_emb = self.q_emb(w_emb) # [batch, q_dim]
#        q_repr = self.net(q_emb)
#        q_proj = self.net(q)
        #print(q_proj.shape)
        logits = self.classifier(q)
        #print("Representation of question is:",q_repr.shape) 
        return logits

class BaseModel_rev(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, classifier):
        super(BaseModel_rev, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        # self.v_net = v_net
        self.classifier = classifier

    def forward(self, v, q, labels):
        """Forward
        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]
        return: logits, not probs
        """
        w_emb = self.w_emb(q)
#         q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(w_emb,v)
        w = (att * w_emb) # [batch, v_dim]
        q_emb = self.q_emb(w) 
        q_repr = self.q_net(q_emb)
#         v_repr = self.v_net(v_emb)
#         joint_repr = q_repr * v_repr
#         logits = self.classifier(joint_repr)
        return q_repr,att

class BaseModel_transform(nn.Module):
    def __init__(self, emb):
        super(BaseModel_transform, self).__init__()
        self.emb = emb

    def forward(self, q):
        """Forward
        v: [batch, num_objs, obj_dim]
        """
        emb = self.emb(q)
        return emb

class StackedAttentionModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, classifier):
        super(StackedAttentionModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.classifier = classifier

    def forward(self, v, q):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)  # [batch, q_dim]

        att = self.v_att(v, q_emb)

#        logits = self.classifier(att)
        return att


def build_baseline0_ques(dataset, num_ans_candidates, num_hid):
    w_emb = WordEmbedding(dataset, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    q_net = FCNet([q_emb.num_hid, num_hid])
    net = FCNet([2048, 1024])
    classifier = SimpleClassifier(num_hid, int(num_hid / 2), 12, 0.5)
    return BaseModel_ques(w_emb, q_emb, net, classifier)

def build_baseline0_jr(dataset, num_ans_candidates, num_hid):
    w_emb = WordEmbedding(dataset, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(2048, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([2048, num_hid])
    return BaseModel_jr(w_emb, q_emb, v_att, q_net, v_net)

def build_baseline0_newatt(dataset, num_ans_candidates, num_hid):
    w_emb = WordEmbedding(dataset, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    #v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([2048, num_hid])
    classifier = SimpleClassifier(num_hid, num_hid * 2, num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, q_net, v_net, classifier)

def build_rev_newatt(dataset, num_ans_candidates, num_hid):
    w_emb = WordEmbedding(dataset, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention_rev(300, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([2048, num_hid])
    classifier = SimpleClassifier(num_hid, num_hid * 2, num_ans_candidates, 0.5)
    return BaseModel_rev(w_emb, q_emb, v_att, q_net, classifier)

def build_transform(dataset, num_hid):
    emb = FCNet([num_hid, num_hid])
    return BaseModel_transform(emb)

def build_san(dataset, num_ans_candidates, num_hid):
    w_emb = WordEmbeddingNew(dataset, 300, 0.0, 'c')
    q_emb = QuestionEmbedding(600 , num_hid, 1, False, 0.0)
    v_att = StackedAttention(2, 2048, num_hid, num_hid, num_ans_candidates,0.5)

    # Loading tfidf weighted embedding
#    if hasattr(args, 'tfidf'):
#        w_emb = tfidf_loading(args.tfidf, w_emb, args, 'data_vqa')
    classifier = SimpleClassifier(num_hid, 2 * num_hid, num_ans_candidates, 0.5)
    return StackedAttentionModel(w_emb, q_emb, v_att, classifier)


