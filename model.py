import torch
import torch.nn as nn
from transformers import AutoModel


class BertMLP(nn.Module):
    def __init__(self, base_model, emotion_model, num_labels) -> None:
        super().__init__()

        self.bert = base_model
        self.emobert = emotion_model
        self.fc = nn.Linear(768 * 2, num_labels)

    def forward(self, input_ids):
        bert_output = self.bert(input_ids)
        emobert_output = self.emobert(input_ids)

        combined_features = torch.cat([bert_output[1], emobert_output[1]], dim=-1)

        logits = self.fc(combined_features)

        return logits
