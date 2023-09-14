from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from .FCN import ANN_Classifier
class LSTM_regressor(torch.nn.Module):
    def __init__(self, seq_len, feature_size ,hidden_dim,inter_dim, output_dim, num_layers=1,dropout=0.2, bidirectional=False):
        super(LSTM_regressor, self).__init__()
        self.seq_len = seq_len
        self.feature_size=feature_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        
        self.rnn = nn.LSTM(feature_size,hidden_dim, num_layers)
        self.fc1 = torch.nn.Linear(hidden_dim, inter_dim)
        self.fc2= torch.nn.Linear(inter_dim,output_dim)
        
#         self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        

        output, (hn, cn) = self.rnn(x)
        # o=torch.cat((hn[-1],cn[-1]),dim=1)
        o=self.fc1(hn[-1])
        o=F.relu(o)
        y_pred=self.fc2(o)

        # y_pred = self.fc(hn[-1])
        
        return y_pred


class RNN_ANNclassifier(nn.Module):
    def __init__(self, seq_len, feature_size, param_dim ,RNN_hidden_dim,RNN_num_layers,fc_hidden_dims:list,apply_dropout=True,dropout=0.2):
        super().__init__()
        self.rnn = nn.LSTM(feature_size,RNN_hidden_dim, RNN_num_layers,batch_first=True)
        self.ann_classifier=ANN_Classifier(RNN_hidden_dim,param_dim,fc_hidden_dims,apply_dropout,dropout)


    def forward(self, x,params):
        

        output, (hn, cn) = self.rnn(x)
        ann_input_data=hn[-1]
        ann_output=self.ann_classifier(ann_input_data,params)
        

        return ann_output#,ann_input_data#for summary

