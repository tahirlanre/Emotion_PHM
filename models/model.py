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
        input_ids_combine = torch.cat((bert_embeddings, emotion_embeddings), axis=-1)

        g_feature = self.relu(torch.matmul(input_ids_combine, self.weight_emotion_W1))

        # Attention gating
        H = torch.mul(
            g_feature, torch.matmul(emotion_embeddings, self.weight_emotion_W2)
        )
        alfa = min(0.001 * (torch.norm(bert_embeddings) / torch.norm(H)), 1)
        E = torch.add(torch.mul(alfa, H), bert_embeddings)

        # Layer normalization and dropout
        embedding_output = self.dropout(self.LayerNorm(E))

        return E


class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights=None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = (
            layer_weights
            if layer_weights is not None
            else nn.Parameter(
                torch.tensor(
                    [1] * (num_hidden_layers + 1 - layer_start), dtype=torch.float
                )
            )
        )

    def forward(self, all_hidden_states):
        all_layer_embedding = all_hidden_states[self.layer_start :, :, :, :]
        weight_factor = (
            self.layer_weights.unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(all_layer_embedding.size())
        )
        weighted_average = (weight_factor * all_layer_embedding).sum(
            dim=0
        ) / self.layer_weights.sum()
        return weighted_average


class BertClassificationModel(nn.Module):
    def __init__(
        self, model_name_or_path, emo_model_name_or_path, num_labels, dropout=0.1
    ):
        super(BertClassificationModel, self).__init__()

        self.bert = AutoModel.from_pretrained(model_name_or_path)
        self.enc_model = AutoModel.from_pretrained(
            model_name_or_path, output_hidden_states=True
        )
        self.emo_model = AutoModel.from_pretrained(
            emo_model_name_or_path, output_hidden_states=True
        )
        self.attn_gate = AttnGating(self.enc_model.config.hidden_size, dropout)
        self.num_labels = num_labels

        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        enc_outputs = self.enc_model(input_ids, attention_mask=attention_mask)
        embeds = enc_outputs.hidden_states[0]

        emo_outputs = self.emo_model(input_ids, attention_mask=attention_mask)
        emo_embeds = emo_outputs.hidden_states[0]

        combined_embeds = self.attn_gate(embeds, emo_embeds)

        outputs = self.bert(
            input_ids=None,
            inputs_embeds=combined_embeds,
        )
        pooled_output = outputs[1]

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,) + outputs[2:]

        return ((loss,) + output) if loss is not None else output


class BiLSTMAttn(nn.Module):
    def __init__(
        self,
        enc_model_name_or_path,
        emo_model_name_or_path,
        hidden_dim,
        num_layers,
        num_labels,
        dropout=0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout)

        self.enc_model = AutoModel.from_pretrained(
            enc_model_name_or_path, output_hidden_states=True
        )
        self.emo_model = AutoModel.from_pretrained(
            emo_model_name_or_path, output_hidden_states=True
        )
        self.attn_gate = AttnGating(self.enc_model.config.hidden_size, dropout)

        self.bilstm = nn.LSTM(
            self.enc_model.config.hidden_size,
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

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        enc_outputs = self.enc_model(input_ids, attention_mask=attention_mask)
        embeds = enc_outputs.hidden_states[0]

        emo_outputs = self.emo_model(input_ids, attention_mask=attention_mask)
        emo_embeds = emo_outputs.hidden_states[0]

        combined_embeds = self.attn_gate(embeds, emo_embeds)

        outputs, (hn, cn) = self.bilstm(combined_embeds)
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
    def __init__(
        self,
        enc_model_name_or_path,
        emo_model_name_or_path,
        hidden_dim,
        num_layers,
        num_labels,
        dropout=0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout)

        self.enc_model = AutoModel.from_pretrained(
            enc_model_name_or_path, output_hidden_states=True
        )
        self.emo_model = AutoModel.from_pretrained(
            emo_model_name_or_path, output_hidden_states=True
        )
        self.attn_gate = AttnGating(self.enc_model.config.hidden_size, dropout)

        self.bilstm = nn.LSTM(
            self.enc_model.config.hidden_size,
            hidden_dim,
            dropout=dropout if num_layers > 1 else 0,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        enc_outputs = self.enc_model(input_ids, attention_mask=attention_mask)
        embeds = enc_outputs.hidden_states[0]

        emo_outputs = self.emo_model(input_ids, attention_mask=attention_mask)
        emo_embeds = emo_outputs.hidden_states[0]

        combined_embeds = self.attn_gate(embeds, emo_embeds)

        outputs, (hn, cn) = self.bilstm(combined_embeds)
        fbhn = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        fbhn = self.dropout(fbhn)

        logits = self.classifier(fbhn)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,)

        return ((loss,) + output) if loss is not None else output


class MLP(nn.Module):
    def __init__(
        self, enc_model_path_or_name, emo_model_path_or_name, num_labels, dropout=0.1
    ):
        super().__init__()
        self.fc = nn.Linear(768, num_labels)
        self.dropout = nn.Dropout(dropout)
        self.num_labels = num_labels
        self.enc_model = AutoModel.from_pretrained(
            enc_model_path_or_name, output_hidden_states=True
        )
        self.emo_model = AutoModel.from_pretrained(
            emo_model_path_or_name, output_hidden_states=True
        )
        self.attn_gate = AttnGating(self.enc_model.config.hidden_size, dropout)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        enc_outputs = self.enc_model(input_ids, attention_mask=attention_mask)
        embeds = enc_outputs[1]
        embeds = self.dropout(embeds)

        emo_outputs = self.emo_model(input_ids, attention_mask=attention_mask)
        emo_embeds = emo_outputs[1]
        emo_embeds = self.dropout(emo_embeds)

        combined_embeds = self.attn_gate(embeds, emo_embeds)

        logits = self.fc(combined_embeds)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        output = (logits,)

        return ((loss,) + output) if loss is not None else output


class EmoBERTMLP(nn.Module):
    def __init__(self, emo_model_name_or_path, num_labels, dropout=0.1) -> None:
        super().__init__()

        self.enc_model = AutoModel.from_pretrained(emo_model_name_or_path)
        self.num_labels = num_labels

        self.classifier = nn.Linear(768, num_labels)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.enc_model(input_ids, attention_mask=attention_mask)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        output = (logits,)

        return ((loss,) + output) if loss is not None else output
