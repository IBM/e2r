import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F

class BiLSTM(nn.Module):
    def __init__(self,lstm_layer=1,vocab_size=1000, hidden_dim=100, embedding_dim=300, quant_embedding_dim=75, dropout=0.2,device='cuda'):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        self.vocab_size=vocab_size
        self.embedding_dim=embedding_dim
        self.device=device
        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.num_layers=lstm_layer

        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=lstm_layer,
                            bidirectional=True)

        self.context2quantam = nn.Linear(self.hidden_dim*4, quant_embedding_dim)

        self.fc_last_1 = nn.Linear(quant_embedding_dim, quant_embedding_dim)
        self.fc_last_2 = nn.Linear(quant_embedding_dim, quant_embedding_dim)
        self.fc_last_3 = nn.Linear(quant_embedding_dim, quant_embedding_dim)

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.2)


    def forward(self, sents,lengths,ids,left_right):


        bs = sents.size(1)  # batch size
        #print('batch size', bs)

        h0 = torch.zeros(self.num_layers * 2, bs, self.hidden_dim).to(self.device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, bs, self.hidden_dim).to(self.device)

        embs =self.emb(sents)
        embs = pack_padded_sequence(embs, lengths)
        lstm_out, (self.hidden,c_n) = self.lstm(embs, (h0,c0))
        #lstm_out, lengths = pad_packed_sequence(lstm_out)


        #return lstm_out[-1,:,:]

        #return torch.cat((self.hidden[0,:,:], self.hidden[1,:,:]), 1) //imp

        prediction= torch.cat((self.hidden[0,:,:], self.hidden[1,:,:]), 1)

        context_embd_dim=2*self.hidden_dim

        final_prediction = torch.zeros(int(bs/2), 2*context_embd_dim).to(self.device)


        ids_list=list(set(ids))

        i = 0
        for id in ids_list:
            first = ids.index(id)
            second = first + 1 + ids[first + 1:].index(id)

            if left_right[first] == 0:
                left = first
                right = second
            else:
                left = second
                right = first
            final_prediction[i, :context_embd_dim] = prediction[left, :]
            final_prediction[i, context_embd_dim:] = prediction[right, :]
            i = i + 1

        predicted_quant_embd_1=self.context2quantam(final_prediction)
        predicted_quant_embd_2=self.fc_last_1(self.tanh(predicted_quant_embd_1))
        predicted_quant_embd_3=self.fc_last_2(self.tanh(predicted_quant_embd_2))
        predicted_quant_embd_4=self.fc_last_3(self.tanh(predicted_quant_embd_3))
        predicted_quant_embd_4=F.normalize(predicted_quant_embd_4,p=2,dim=1)



        return predicted_quant_embd_4,ids_list



def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim),
                torch.randn(2, 1, self.hidden_dim))