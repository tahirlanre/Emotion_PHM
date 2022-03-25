import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnnutils

from transformers import AutoModel

from configs import const

### Credits https://github.com/eturcan/emotion-infused/
class BertMultitask(nn.Module):
    def __init__(
        self, num_labels, is_multilabel, bert_type="bert-base-uncased"
    ) -> None:
        super().__init__()

        self.bert = AutoModel.from_pretrained(bert_type)

        self.num_labels = num_labels
        self.is_multilabel = is_multilabel

        self.dropout = nn.Dropout = nn.Linear()
