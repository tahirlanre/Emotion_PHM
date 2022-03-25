import torch
import torch.nn as nn
from transformers import AutoModel


class BertMLP(nn.Module):
    def __init__(self, base_model, emotion_model, num_labels) -> None:
        super().__init__()

        self.bert = base_model
        self.emobert = emotion_model
        self.fc = nn.Linear(768, num_labels)
        self.fc_emo = nn.Linear(768, 7)
        self.attn_gate = AttnGating(200, 768, 0.5)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids):
        bert_output = self.bert(input_ids, output_hidden_states=True)
        emobert_output = self.emobert(input_ids)

        pooled_output = emobert_output[1]
        pooled_output = self.dropout(pooled_output)
        emo_logits = self.fc_emo(pooled_output)

        emotion_feature = 1 / (1 + torch.exp(-emo_logits))  # Sigmoid

        combined_features = self.attn_gate(bert_output[2][0], emotion_feature)

        outputs = self.bert(input_ids=None, inputs_embeds=combined_features)
        sequence_output = outputs.last_hidden_state

        x = sequence_output[:, 0, :]
        x = self.dropout(x)
        x = torch.tanh(x)
        x = self.dropout(x)

        logits = self.fc(x)

        return logits


class AttnGating(nn.Module):
    def __init__(self, embedding_size, hidden_size, dropout_prob):
        super(AttnGating, self).__init__()

        self.linear = nn.Linear(7, embedding_size)
        self.relu = nn.ReLU(inplace=True)

        self.weight_emotion_W1 = nn.Parameter(
            torch.Tensor(hidden_size + embedding_size, hidden_size)
        )
        self.weight_emotion_W2 = nn.Parameter(torch.Tensor(embedding_size, hidden_size))

        nn.init.uniform_(self.weight_emotion_W1, -0.1, 0.1)
        nn.init.uniform_(self.weight_emotion_W2, -0.1, 0.1)

        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, embeddings_bert, linguistic_feature):
        # Project linguistic representations into vectors with comparable size
        linguistic_feature = self.linear(linguistic_feature)
        emotion_feature = linguistic_feature.repeat(
            embeddings_bert.size(1), 1, 1
        )  # (50, bs, 200)
        emotion_feature = emotion_feature.permute(1, 0, 2)  # (bs, 50, 200)

        # Concatnate word and linguistic representations
        features_combine = torch.cat(
            (emotion_feature, embeddings_bert), axis=2
        )  # (bs, 50, 968)

        g_feature = self.relu(torch.matmul(features_combine, self.weight_emotion_W1))

        # Attention gating
        H = torch.mul(g_feature, torch.matmul(emotion_feature, self.weight_emotion_W2))
        alfa = min(0.001 * (torch.norm(embeddings_bert) / torch.norm(H)), 1)
        E = torch.add(torch.mul(alfa, H), embeddings_bert)

        # Layer normalization and dropout
        embedding_output = self.dropout(self.LayerNorm(E))

        return E
