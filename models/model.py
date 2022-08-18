import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class MultiheadAttention_test(nn.Module):
    # n_heads：number of heads
    # hid_dim：the hidden dimension
    def __init__(self, hid_dim, n_heads, dropout):
        super(MultiheadAttention_test, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)  # dimension reduction
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to("cuda")

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(
            0, 2, 1, 3
        )
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(
            0, 2, 1, 3
        )
        V = V.view(bsz, -1, self.n_heads, (self.hid_dim) // self.n_heads).permute(
            0, 2, 1, 3
        )

        # [64,6,12,50] * [64,6,50,10] = [64,6,12,10]
        # attention：[64,6,12,10]
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        # attention: [64,6,12,10]
        attention = self.do(torch.softmax(attention, dim=-1))

        # [64,6,12,10] * [64,6,10,50] = [64,6,12,50]
        # x: [64,6,12,50]
        x = torch.matmul(attention, V)

        # x: [64,6,12,50] -> [64,12,6,50]
        x = x.permute(0, 2, 1, 3).contiguous()
        # x: [64,12,6,50] -> [64,12,300]
        x = x.view(bsz, -1, self.n_heads * ((self.hid_dim) // self.n_heads))
        x = self.fc(x)
        return x


class BERT_MTL_ATTN(nn.Module):
    def __init__(
        self, enc_model_path_or_name, emo_model_path_or_name, num_labels, dropout=0.2
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
        self.attention = MultiheadAttention_test(768, 6, dropout=0.1)
        # self.attn_gate = AttnGating(self.enc_model.config.hidden_size, dropout)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        enc_outputs = self.enc_model(input_ids, attention_mask=attention_mask)
        embeds = enc_outputs[1]
        embeds = self.dropout(embeds)

        emo_outputs = self.emo_model(input_ids, attention_mask=attention_mask)
        emo_embeds = emo_outputs[1]
        emo_embeds = self.dropout(emo_embeds)

        combined_embeds = torch.cat(
            (embeds.view(-1, 1, 768), emo_embeds.view(-1, 1, 768)), 1
        )

        output_afterAttention = self.attention(
            combined_embeds, combined_embeds, combined_embeds
        )[:, 0, :]
        # The size of outputs :[BatchSize*1*768]
        # outputs_final = self.dropout(output_afterAttention.view(-1,768))
        outputs_final = output_afterAttention.view(-1, 768)

        logits = self.fc(outputs_final)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        output = (logits,)

        return ((loss,) + output) if loss is not None else output


class BERT_MTL(nn.Module):
    def __init__(
        self, enc_model_path_or_name, emo_model_path_or_name, num_labels, dropout=0.2
    ):
        super().__init__()
        self.fc = nn.Linear(768 * 2, num_labels)
        self.dropout = nn.Dropout(dropout)
        self.num_labels = num_labels
        self.enc_model = AutoModel.from_pretrained(
            enc_model_path_or_name, output_hidden_states=True
        )
        self.emo_model = AutoModel.from_pretrained(
            emo_model_path_or_name, output_hidden_states=True
        )
        # self.attn_gate = AttnGating(self.enc_model.config.hidden_size, dropout)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        enc_outputs = self.enc_model(input_ids, attention_mask=attention_mask)
        embeds = enc_outputs[1]
        embeds = self.dropout(embeds)

        emo_outputs = self.emo_model(input_ids, attention_mask=attention_mask)
        emo_embeds = emo_outputs[1]
        emo_embeds = self.dropout(emo_embeds)

        combined_embeds = torch.cat((embeds, emo_embeds), 1)

        logits = self.fc(combined_embeds)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        output = (logits,)

        return ((loss,) + output) if loss is not None else output


class BERT_STL(nn.Module):
    def __init__(self, enc_model_name_or_path, num_labels, dropout=0.2) -> None:
        super().__init__()

        self.enc_model = AutoModel.from_pretrained(enc_model_name_or_path)
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
