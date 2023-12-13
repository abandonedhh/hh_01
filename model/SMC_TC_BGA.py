import torch
from transformers import BertModel
import torch.nn.functional as F


class SMC_TC(torch.nn.Module):
    def __init__(self, config):
        super(SMC_TC, self).__init__()
        self.fc = torch.nn.Linear(config.hidden_size * 2, config.num_labels)
        self.pretrained = BertModel.from_pretrained(config.pretrained)
        for param in self.pretrained.parameters():
            param.requires_grad = False
        self.lstm = torch.nn.LSTM(config.bert_h, config.hidden_size, batch_first=True, bidirectional=True, dropout=config.dropout)
        self.self_attention = torch.nn.MultiheadAttention(embed_dim=config.hidden_size * 2, num_heads=1,
                                                          dropout=config.dropout, batch_first=True)

    def forward(self, data):
        word_token_type_ids, word_attention_mask, word_input_ids = data[3]

        tw, _3 = self.pretrained(input_ids=word_input_ids,
                                 attention_mask=word_attention_mask,
                                 token_type_ids=word_token_type_ids,
                                 return_dict=False)
        hw=self.lstm(F.relu(tw))[0]  # shape[batch_size,max_length,2*hidden_size]
        hw = self.self_attention(hw, hw, hw)[0]  # shape[batch_size,max_length,2*hidden_size]
        hw_ = F.avg_pool1d(hw.transpose(1, 2), hw.size(1))[:, :, 0]  # shape[batch_size,2*hidden_size]
        x = self.fc(F.relu(hw_))
        return F.sigmoid(x)


