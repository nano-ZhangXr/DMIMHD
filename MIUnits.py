import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .SelfAttention import SelfAttention
from .Selector import Selector
from .Refinement import Refinement


class RectifiedIdentityUnit(nn.Module):
    def __init__(self, cfg, num_out_path):
        super(RectifiedIdentityUnit, self).__init__()
        self.keep_mapping = nn.ReLU()
        self.selector = Selector(num_out_path, cfg.embed_size, cfg.hid_selector)

    def forward(self, x):
        path_prob = self.selector(x)
        emb = self.keep_mapping(x)

        return emb, path_prob

class SemanticRelationUnit(nn.Module):
    def __init__(self, cfg, num_out_path):
        super(SemanticRelationUnit, self).__init__()
        self.cfg = cfg
        self.selector = Selector(num_out_path, cfg.embed_size, cfg.hid_selector)
        self.sa = SelfAttention(cfg.embed_size, cfg.hid_sru, cfg.num_head_sru, cfg.dropout)

    def forward(self, inp, stc_lens=None):
        path_prob = self.selector(inp)
        if inp.dim() == 4:
            n_img, n_stc, n_local, dim = inp.size()
            x = inp.view(-1, n_local, dim)
        else:
            x = inp

        sa_emb = self.sa(x)
        if inp.dim() == 4:
            sa_emb = sa_emb.view(n_img, n_stc, n_local, -1)
        return sa_emb, path_prob

class CrossmodalEnhancementUnit(nn.Module):
    def __init__(self, cfg, num_out_path):
        super(CrossmodalEnhancementUnit, self).__init__()
        self.refine = Refinement(cfg.embed_size, cfg.lambda_softmax_cmeu)
        self.selector = Selector(num_out_path, cfg.embed_size, cfg.hid_selector)

    def forward(self, rgn, img, wrd, stc, stc_lens):
        l_emb = rgn
        path_prob = self.selector(l_emb)
        rf_pairs_emb = self.refine(rgn, img, wrd, stc, stc_lens)
        return rf_pairs_emb, path_prob

class ContextualGuidanceUnit(nn.Module):
    def __init__(self, cfg, num_out_path):
        super(ContextualGuidanceUnit, self).__init__()
        self.cfg = cfg
        self.selector = Selector(num_out_path, cfg.embed_size, cfg.hid_selector)
        self.fc = nn.Linear(cfg.embed_size, cfg.embed_size)


    def forward(self, rgn, img, wrd, stc, stc_lens):  
        path_prob = self.selector(rgn)  
          
        n_img = rgn.size(0)  
        n_rgn = rgn.size(-2)  
        n_stc = stc.size(0)  
        ref_rgns = []  
          
        for i in range(n_stc):  
            if rgn.dim() == 4:  
                query = rgn[:, i, :, :]  
            else:  
                query = rgn  
              
            stc_i = stc[i].unsqueeze(0).unsqueeze(1).contiguous()  
            stc_i_expand = stc_i.expand(n_img, n_rgn, -1)  
              
            l_emb_mid = self.fc(query)  
            x = l_emb_mid * stc_i_expand  
            x = F.normalize(x, dim=-2)  
            ref_rgn = (1 + x) * query  
              
            ref_rgn = ref_rgn.unsqueeze(1)  
            ref_rgns.append(ref_rgn)  
          
        ref_rgns = torch.cat(ref_rgns, dim=1)  
          
        return ref_rgns, path_prob



