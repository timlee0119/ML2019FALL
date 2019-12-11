##################### Define Dataset and model classes ##################
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import spacy

class hw5_dataset(Dataset):
    def __init__(self, is_training, dataX, dataY, w2v):
        self.is_training = is_training
        self.dataX = dataX
        self.dataY = dataY
        self.w2v = w2v # word2vec model
        
    def __len__(self):
        return len(self.dataX)
    
    def __getitem__(self, idx):
        x = [self.w2v.wv[x] for x in self.dataX[idx]]
        if not self.is_training:
            return x
        y = self.dataY[idx]
        return x, y

def pad_collate(batch):
    (xx, yy) = zip(*batch)
    xx = [torch.tensor(x) for x in xx]
    yy = torch.tensor(yy)
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    return xx_pad, yy

def pad_collate_test(batch):
    batch = [torch.tensor(x) for x in batch]
    xx_pad = pad_sequence(batch, batch_first=True, padding_value=0)
    return xx_pad

def word_segmentation(comments):
    ret = comments.copy()
    nlp = spacy.load("en_core_web_sm")
    for i in range(len(ret)):
        doc = nlp(ret[i])
        ret[i] = [x.text for x in doc]
    return ret.tolist()

class RNN_LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_size=128, num_layers=1, dropout=0, bidirectional=False):
        super(RNN_LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, 
                            num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size*(2 if bidirectional else 1), 32),
            nn.LeakyReLU(),
            nn.BatchNorm1d(32)
        )
        self.fc2 = nn.Linear(32, 2)
        
        # orthogonal init lstm parameters
        for param in self.lstm.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.fc2(out)
        return out
        