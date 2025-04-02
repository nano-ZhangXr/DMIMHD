import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pickle
from .MIUnits import RectifiedIdentityUnit, SemanticRelationUnit, CrossmodalEnhancementUnit, ContextualGuidanceUnit


def unsqueeze2d(x):
    return x.unsqueeze(-1).unsqueeze(-1)

def unsqueeze3d(x):
    return x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class DynamicInteraction_Layer(nn.Module):
    def __init__(self, cfg, num_cell, num_out_path):
        super(DynamicInteraction_Layer, self).__init__()
        self.cfg = cfg
        self.threshold = cfg.threshold
        self.eps = 1e-8
        self.num_cell = num_cell
        self.num_out_path = num_out_path
        self.riu = RectifiedIdentityUnit(cfg, num_out_path)
        self.sru = SemanticRelationUnit(cfg, num_out_path)
        self.cgu = ContextualGuidanceUnit(cfg, num_out_path)
        self.cmeu = CrossmodalEnhancementUnit(cfg, num_out_path)


    def forward(self, rgn, img, wrd, stc, stc_lens):
        path_prob = [None] * self.num_cell
        emb_lst = [None] * self.num_cell

        emb_lst[0], path_prob[0] = self.riu(rgn)
        emb_lst[1], path_prob[1] = self.cgu(rgn, img, wrd, stc, stc_lens)
        emb_lst[2], path_prob[2] = self.sru(rgn)
        emb_lst[3], path_prob[3] = self.cmeu(rgn, img, wrd, stc, stc_lens)


        gate_mask = (sum(path_prob) < self.threshold).float()
        all_path_prob = torch.stack(path_prob, dim=2)
        all_path_prob = all_path_prob / (all_path_prob.sum(dim=-1, keepdim=True) + self.eps)
        path_prob = [all_path_prob[:, :, i] for i in range(all_path_prob.size(2))]

        aggr_res_lst = []
        for i in range(self.num_out_path):
            skip_emb = unsqueeze2d(gate_mask[:, i]) * emb_lst[0]
            res = 0
            for j in range(self.num_cell):
                cur_path = unsqueeze3d(path_prob[j][:, i])
                if emb_lst[j].dim() == 3:
                    cur_emb = emb_lst[j].unsqueeze(1)
                else:  # 4
                    cur_emb = emb_lst[j]
                res = res + cur_path * cur_emb
            res = res + skip_emb.unsqueeze(1)
            aggr_res_lst.append(res)

        return aggr_res_lst, all_path_prob
