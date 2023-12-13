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
        self.gru1 = torch.nn.GRU(config.bert_h, config.hidden_size, batch_first=True, bidirectional=True, dropout=config.dropout)
        self.gru2 = torch.nn.GRU(config.bert_h, config.hidden_size, batch_first=True, bidirectional=True, dropout=config.dropout)


    def forward(self, data):
        text_input_ids, text_attention_mask, text_token_type_ids = data[1]
        topic_token_type_ids, topic_attention_mask, topic_input_ids = data[2]

        tc, _1 = self.pretrained(input_ids=topic_input_ids,
                                 attention_mask=topic_attention_mask,
                                 token_type_ids=topic_token_type_ids,
                                 return_dict=False)
        tt, _2 = self.pretrained(input_ids=text_input_ids,
                                 attention_mask=text_attention_mask,
                                 token_type_ids=text_token_type_ids,
                                 return_dict=False)


        hc, ht = self.gru1(F.relu(tc))[0], self.gru2(F.relu(tt))[0]# shape[batch_size,max_length,2*hidden_size]
        S = self.aoa_attention(hc, ht)[:, :, 0]  # shape[batch_size,2*hidden_size]
        hc_ = F.avg_pool1d(hc.transpose(1, 2), hc.size(1))[:, :, 0]  # shape[batch_size,*2hidden_size]
        ht_ = F.avg_pool1d(ht.transpose(1, 2), ht.size(1))[:, :, 0]  # shape[batch_size,2*hidden_size]


        x = torch.cat((S, hc_, ht_), dim=1)  # shape[batch_size,8*hidden_size]

        x = self.fc(F.relu(x))
        return F.sigmoid(x)

    def aoa_attention(self, hc, ht):
        I = torch.matmul(hc, ht.transpose(1, 2))  # shape[batch_size,topic_len,text_len]
        A = torch.softmax(I, dim=2)  # shape[batch_size,topic_len,text_len]
        B = torch.softmax(I, dim=1)  # shape[batch_size,topic_len,text_len]
        B_ = F.avg_pool1d(B, B.size(2))  # shape[batch_size,topic_len,1]
        C = torch.matmul(A.transpose(1, 2), B_)  # shape[batch_size,1,text_len]
        out = torch.matmul(C.transpose(1, 2), ht)  # shape[batch_size,1,2*hidden_size]
        return out.transpose(1, 2)  # shape[batch_size,2*hidden_size,1]
