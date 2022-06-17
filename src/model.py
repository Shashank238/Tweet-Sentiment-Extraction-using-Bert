import config
import torch
import transformers
import torch.nn as nn
from torch.nn import functional as F


class TweetModel(nn.Module):
    def __init__(self):
        super(TweetModel, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.drop_out = nn.Dropout(0.3) 
        self.l0 = nn.Linear(768 , 2)
       
    
    def forward(self, ids, mask, token_type_ids):
        sequence_output,_ = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )

        logits = self.l0(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
