import os
from argparse import ArgumentParser
import numpy as np 
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from hw4_common import Autoencoder

def dump_model(model, model_path):
    dirname = os.path.dirname(model_path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
        
    from datetime import datetime
    now = datetime.now().strftime('%Y%m%d_%H%M')
    p = f'{model_path}_{now}'
    torch.save(model.state_dict(), p)
    print('Model saved to %s' % p)
    
def build_argparser():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', help="Required. Path to trainX.npy.", required=True, type=str)
    parser.add_argument('-o', '--output', help="Required. Path to the output autoencoder model file.", required=True, type=str)
    parser.add_argument('-e', '--epoch', help="Required. Number of training epoch.", required=True, type=int)
    return parser

if __name__ == '__main__':
    args = build_argparser().parse_args()
    
    use_gpu = torch.cuda.is_available()
    print(("Using GPU" if use_gpu else "Using CPU"))
    
    autoencoder = Autoencoder()
    
    # load data and normalize to [-1, 1]
    trainX = np.load(args.input)
    trainX = np.transpose(trainX, (0, 3, 1, 2)) / 255. * 2 - 1
    trainX = torch.Tensor(trainX)
    
    if use_gpu:
        autoencoder.cuda()
        trainX = trainX.cuda()
    
    train_dataloader = DataLoader(trainX, batch_size=60, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.005)
    epochs = args.epoch
    
    # stop training if (1) finish epochs or (2) loss greater than last 10
    last_loss = 1000
    for epoch in range(epochs):
        cumulate_loss = 0
        for x in train_dataloader:
            latent, reconstruct = autoencoder(x)
            loss = criterion(reconstruct, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cumulate_loss += loss.item() * x.shape[0]

        if epoch % 10 == 9:
            cur_loss = (cumulate_loss / trainX.shape[0])
            print(f'Epoch { "%03d" % (epoch+1) }: Loss: { "%f" % cur_loss }')
            if cur_loss > last_loss:
                break
            else:
                last_loss = cur_loss
                  
    dump_model(autoencoder, args.output)
