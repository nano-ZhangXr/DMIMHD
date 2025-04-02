import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class TxtEncoder(nn.Module):
    def __init__(self, cfg):
        super(TxtEncoder, self).__init__()
        self.cfg = cfg

        # Pretrained BERT
        self.bert = BertModel.from_pretrained(cfg.model.model_name_or_path)
        
        # 1D-CNN
        Ks = [1, 2, 3]
        in_channel = 1
        out_channel = cfg.selecting.embed_size
        bert_hid = self.bert.config.hidden_size
        self.fc = nn.Linear(bert_hid, cfg.selecting.embed_size)
        self.convs1 = nn.ModuleList([nn.Conv2d(in_channel, out_channel, (K, bert_hid)) for K in Ks])
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.mapping = nn.Linear(len(Ks) * out_channel, cfg.selecting.embed_size)

    def forward(self, input_ids, attention_mask):
        # BERT output
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        all_encoder_layers = outputs.last_hidden_state

        # 1D-CNN Process
        x = all_encoder_layers.unsqueeze(1)
        x_emb = self.fc(all_encoder_layers)
        x1 = F.relu(self.convs1[0](x)).squeeze(3)
        x2 = F.relu(self.convs1[1](F.pad(x, (0, 0, 0, 1)))).squeeze(3)
        x3 = F.relu(self.convs1[2](F.pad(x, (0, 0, 1, 1)))).squeeze(3)
        x = torch.cat([x1, x2, x3], dim=1)
        x = x.transpose(1, 2)
        word_emb = self.mapping(x)
        word_emb = word_emb + x_emb
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in [x1, x2, x3]]
        x = torch.cat(x, 1)

        # Global feature->sentence_emb | Regional feature->word_emb
        sentence_emb = self.mapping(x)
        sentence_emb = sentence_emb + x_emb.mean(1)
        sentence_emb = F.normalize(sentence_emb, p=2, dim=-1)
        word_emb = F.normalize(word_emb, p=2, dim=-1)

        return sentence_emb, word_emb

