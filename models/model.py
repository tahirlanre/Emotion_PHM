import torch
import torch.nn as nn
from transformers import AutoModel


class AttnGating(nn.Module):
    def __init__(self, hidden_size, dropout_prob):
        super(AttnGating, self).__init__()

        # self.linear = nn.Linear(7, embedding_size)
        self.relu = nn.ReLU(inplace=True)

        self.weight_emotion_W1 = nn.Parameter(
            torch.Tensor(hidden_size * 2, hidden_size)
        )
        self.weight_emotion_W2 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        nn.init.uniform_(self.weight_emotion_W1, -0.1, 0.1)
        nn.init.uniform_(self.weight_emotion_W2, -0.1, 0.1)

        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, bert_embeddings, emotion_embeddings):
        # Concatnate word and emotion representations
        features_combine = torch.cat(
            (bert_embeddings, emotion_embeddings), axis=-1
        )  # (bs, 50, 968)

        g_feature = self.relu(torch.matmul(features_combine, self.weight_emotion_W1))

        # Attention gating
        H = torch.mul(
            g_feature, torch.matmul(emotion_embeddings, self.weight_emotion_W2)
        )
        alfa = min(0.001 * (torch.norm(bert_embeddings) / torch.norm(H)), 1)
        E = torch.add(torch.mul(alfa, H), bert_embeddings)

        # Layer normalization and dropout
        embedding_output = self.dropout(self.LayerNorm(E))

        return E


class BertClassificationModel(nn.Module):
    def __init__(self, model_name_or_path, num_labels):
        super(BertClassificationModel, self).__init__()

        self.bert = AutoModel.from_pretrained(
            model_name_or_path, add_pooling_layer=False, return_dict=True
        )

        self.dropout = nn.Dropout(0.5)
        self.num_labels = num_labels

        self.classifier = nn.Linear(768, num_labels)

        self.beta = 0.1

    def forward(self, embedding_output, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=None,
            inputs_embeds=embedding_output,
            attention_mask=attention_mask,
        )
        sequence_output = outputs.last_hidden_state

        x = sequence_output[:, 0, :]
        x = self.dropout(x)
        # x = torch.tanh(x)
        # x = self.dropout(x)

        logits = self.classifier(x)
        loss = None

        # Training on binary complaint
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,) + outputs[2:]

        return ((loss,) + output) if loss is not None else output
