import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class AttnGating(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
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
        self.dropout = nn.Dropout(dropout)

    def forward(self, bert_embeddings, emotion_embeddings):
        # Concatnate word and emotion representations
        features_combine = torch.cat((bert_embeddings, emotion_embeddings), axis=-1)

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
    def __init__(self, model_name_or_path, num_labels, dropout=0.1):
        super(BertClassificationModel, self).__init__()

        self.bert = AutoModel.from_pretrained(
            model_name_or_path, add_pooling_layer=False, return_dict=True
        )

        self.dropout = nn.Dropout(dropout)
        self.num_labels = num_labels

        self.classifier = nn.Linear(768, num_labels)

    def forward(self, embedding_output, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=None,
            inputs_embeds=embedding_output,
            attention_mask=attention_mask,
        )
        sequence_output = outputs.last_hidden_state

        x = sequence_output[:, 0, :]
        x = self.dropout(x)
        x = torch.tanh(x)
        x = self.dropout(x)

        logits = self.classifier(x)
        loss = None

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,) + outputs[2:]

        return ((loss,) + output) if loss is not None else output


class BiLSTMAttn(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, num_labels, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            dropout=dropout if num_layers > 1 else 0,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def attnetwork(self, encoder_out, final_hidden):
        hidden = final_hidden.squeeze(0)
        attn_weights = torch.bmm(encoder_out, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden = torch.bmm(
            encoder_out.transpose(1, 2), soft_attn_weights.unsqueeze(2)
        ).squeeze(2)

        return new_hidden

    def forward(self, features, labels=None):
        outputs, (hn, cn) = self.encoder(features)
        fbout = outputs[:, :, : self.hidden_dim] + outputs[:, :, self.hidden_dim :]
        fbhn = (hn[-2, :, :] + hn[-1, :, :]).unsqueeze(0)
        fbhn = self.dropout(fbhn)
        attn_out = self.attnetwork(fbout, fbhn)

        logits = self.classifier(attn_out)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,)

        return ((loss,) + output) if loss is not None else output


class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, num_labels, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            dropout=dropout if num_layers > 1 else 0,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, features, labels=None):
        outputs, (hn, cn) = self.encoder(features)
        fbhn = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        fbhn = self.dropout(fbhn)

        logits = self.classifier(fbhn)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,)

        return ((loss,) + output) if loss is not None else output
