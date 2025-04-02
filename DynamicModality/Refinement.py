import torch
import torch.nn as nn
import torch.nn.functional as F


def l2norm(X, dim, eps=1e-8):
    # L2-normalize
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def func_attention(query, context, smooth, eps=1e-8):
    # query: (n_context, queryL, d), context: (n_context, sourceL, d)
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)   #(n, d, qL)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)   #(n, cL, qL)

    # clipped_l2norm
    attn = nn.LeakyReLU(0.1)(attn)
    # attn = l2norm(attn, 2)
    attn = F.normalize(attn, dim=2)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()   #(n, qL, cL)
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)      #(n*qL, cL)
    attn = F.softmax(attn*smooth, dim=-1)             #(n*qL, cL)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)     #(n, qL, cL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()  #(n, cL, qL)
    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)   #(n, d, cL)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)    #(n, d, qL)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)    #(n, qL, d)

    return weightedContext, attnT


class Refinement(nn.Module):
    def __init__(self, embed_size, lambda_softmax):
        super(Refinement, self).__init__()
        self.lambda_softmax = lambda_softmax
        self.fc_scale = nn.Linear(embed_size, embed_size)
        self.fc_shift = nn.Linear(embed_size, embed_size)
        self.fc_1 = nn.Linear(embed_size, embed_size)
        self.fc_2 = nn.Linear(embed_size, embed_size)

    def refine(self, query, weiContext):
        scaling = torch.tanh(self.fc_scale(weiContext))
        shifting = self.fc_shift(weiContext)  
        modu_res = self.fc_2(F.relu(self.fc_1(query * scaling + shifting))) 
        ref_q = modu_res + query

        return ref_q
    
    def forward_ref(self, rgn, wrd, cap_lens):
        ref_imgs = []
        n_image = rgn.size(0)
        n_caption = wrd.size(0) 

        for i in range(n_caption):
            if rgn.dim() == 4:
                query = rgn[:, i, :, :]  # (n_img, r_rgn, d)
            else:
                query = rgn
            # Get the i-th text description
            n_word = cap_lens[i]

            cap_i = wrd[i, :n_word, :].unsqueeze(0).contiguous()
            # (n_image, n_word, d)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            weiContext, attn = func_attention(query, cap_i_expand, smooth=self.lambda_softmax)
            ref_img = self.refine(query, weiContext)
            ref_img = ref_img.unsqueeze(1)
            ref_imgs.append(ref_img)

        ref_imgs = torch.cat(ref_imgs, dim=1)   #(n_img, n_stc, n_rgn, d)
        return ref_imgs


    def forward(self, rgn, img, wrd, stc, stc_lens):
        ref_emb = self.forward_ref(rgn, wrd, stc_lens) #(n_img, n_stc, n_rgn, d)

        return ref_emb

    

