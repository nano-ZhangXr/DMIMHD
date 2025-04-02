import torch
import torch.nn as nn
import torch.nn.functional as F
from .DynamicInteractionLayer import DynamicInteraction_Layer


class InteractionModule(nn.Module):
    def __init__(self, cfg, num_units=4):
        super(InteractionModule, self).__init__()
        self.cfg = cfg
        self.num_cells = num_units
        self.dynamic_IL = DynamicInteraction_Layer(cfg, num_units, num_out_path=1)


    def forward(self, rgn, img, wrd, stc, stc_lens):
        pairs_emb_lst, paths_l = self.dynamic_IL(rgn, img, wrd, stc, stc_lens)
        Aggr = pairs_emb_lst[-1]
        Aggr = Aggr.sum(1)

        return Aggr.mean(1)
