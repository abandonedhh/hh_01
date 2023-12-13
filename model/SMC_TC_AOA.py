import torch
from transformers import BertModel
import torch.nn.functional as F


class SMC_TC(torch.nn.Module):
    def __init__(self, config):
        super(SMC_TC, self).__init__()
        self.fc = torch.nn.Linear(config.hidden_size * 6, config.num_labels)
        self.pretrained = BertModel.from_pretrained(config.pretrained)
        for param in self.pretrained.parameters():
            param.requires_grad = False
        self.gru1 = torch.nn.GRU(config.bert_h, config.hidden_size, batch_first=True, bidirectional=True,
                                 dropout=config.dropout)
        self.gru2 = torch.nn.GRU(config.bert_h, config.hidden_size, batch_first=True, bidirectional=True,
                                 dropout=config.dropout)
        self.lstm = torch.nn.LSTM(config.bert_h, config.hidden_size, batch_first=True, bidirectional=True,
                                  dropout=config.dropout)
        self.self_attention = torch.nn.MultiheadAttention(embed_dim=config.hidden_size * 2, num_heads=1,
                                                          dropout=config.dropout, batch_first=True)

    def forward(self, data):
        text_input_ids, text_attention_mask, text_token_type_ids = data[1]
        topic_token_type_ids, topic_attention_mask, topic_input_ids = data[2]
        word_token_type_ids, word_attention_mask, word_input_ids = data[3]
        tc, _1 = self.pretrained(input_ids=topic_input_ids,
                                 attention_mask=topic_attention_mask,
                                 token_type_ids=topic_token_type_ids,
                                 return_dict=False)
        tt, _2 = self.pretrained(input_ids=text_input_ids,
                                 attention_mask=text_attention_mask,
                                 token_type_ids=text_token_type_ids,
                                 return_dict=False)
        tw, _3 = self.pretrained(input_ids=word_input_ids,
                                 attention_mask=word_attention_mask,
                                 token_type_ids=word_token_type_ids,
                                 return_dict=False)

        hc, ht, hw = self.gru1(F.relu(tc))[0], self.gru2(F.relu(tt))[0], self.lstm(
            F.relu(tw))[0]  # shape[batch_size,max_length,2*hidden_size]
        hc_ = F.avg_pool1d(hc.transpose(1, 2), hc.size(1))[:, :, 0]  # shape[batch_size,*2hidden_size]
        ht_ = F.avg_pool1d(ht.transpose(1, 2), ht.size(1))[:, :, 0]  # shape[batch_size,2*hidden_size]

        hw = self.self_attention(hw, hw, hw)[0]  # shape[batch_size,max_length,2*hidden_size]

        hw_ = F.avg_pool1d(hw.transpose(1, 2), hw.size(1))[:, :, 0]  # shape[batch_size,2*hidden_size]
        x = torch.cat((hw_, hc_, ht_), dim=1)  # shape[batch_size,8*hidden_size]

        x = self.fc(F.relu(x))
        return F.sigmoid(x)
