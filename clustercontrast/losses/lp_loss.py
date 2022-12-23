from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class CriterionLP(nn.Module):
    """Label Preserving Loss in ISE"""
    def __init__(self, args):
        super(CriterionLP, self).__init__()
        self.args = args

    def forward(self, feats, feats_s, labels, labels_s):
        B = feats.size(0)
        C = feats.size(-1)
        topk = self.args.topk
        feats_s = feats_s.reshape(B * topk, C)
        sim_instance_support = feats.mm(feats_s.t()) / self.args.temp_lp_loss
        exp_instance_support = torch.exp(sim_instance_support)
        loss = 0
        for idx, (exp_i_s, lb) in enumerate(zip(exp_instance_support, labels)):
            pos_sim, pos_ind, neg_sim, neg_ind = self.find_hardest_support(exp_i_s, idx, lb, labels_s)
            loss += (-torch.log((pos_sim.sum() / (pos_sim.sum() + neg_sim.sum() + 1e-6)) + 1e-6))
        loss = loss / B
        return loss

    def find_hardest_support(self, sim, idx, label, labels_s):
        is_pos = labels_s == label
        is_neg = ~is_pos
        pos_sim, relative_p_inds = torch.min(sim[is_pos].contiguous().view(1, -1), 1, keepdim=False)

        sim_negs = sim[is_neg]
        try:
            is_neg = is_neg.reshape(-1, self.args.topk * self.args.num_instances)
            sim_negs = sim_negs.reshape(-1, self.args.topk * self.args.num_instances)
            neg_sim, relative_n_inds = torch.max(sim_negs.contiguous(), 1, keepdim=False)
        except:
            print("is_neg shape is: {}".format(is_neg.size()))
            print("sim_negs shape is: {}".format(sim_negs.size()))
            print("use all negative samples")
            relative_n_inds = None
            neg_sim = sim_negs

        return pos_sim, relative_p_inds, neg_sim, relative_n_inds
