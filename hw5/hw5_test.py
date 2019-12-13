import os
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from hw5_common import RNN_LSTM, hw5_dataset, pad_collate_test, word_segmentation

def write_ans(result, path):
    print(f'Writing results to {path}')
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
        
    df = pd.DataFrame({'id': np.arange(0, len(result)), 'label': result})
    df.to_csv(path, index=False)

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument('--testx', help="Required. Path to test_x.csv.", required=True, type=str)
    parser.add_argument('--w2v', help="Required. Path to word2Vec model.", required=True, type=str)
    parser.add_argument('-m', '--model', help="Required. Path to a trained lstm model.", required=True, type=str)
    parser.add_argument('-o', '--output', help="Required. Path to submission.csv.", required=True, type=str)
    
    return parser

if __name__ == '__main__':
    ####### hyper parameters #######
    embedding_dim = 200
    hidden_size = 96
    batch_size = 64
    bd = False
    n_l = 3
    dp = 0.8
    ################################
    
    args = build_argparser().parse_args()
    
    # Load models
    print(f'Loading model parameters from {args.model}')
    model = RNN_LSTM(embedding_dim, hidden_size=hidden_size, bidirectional=bd, num_layers=n_l, dropout=dp) # must set to same value as trained model
    model.load_state_dict(torch.load(args.model))
    print(f'Loading Word2Vec model from {args.w2v}')
    myw2v = Word2Vec.load(args.w2v)
    
    # Construct DataLoaders
    test_x = pd.read_csv(args.testx)
    test_comments = word_segmentation(test_x.comment)
    
    # if test comment is empty, set it equal to it's previous
    for i in range(len(test_comments)):
        if len(test_comments[i]) == 0:
            test_comments[i] = test_comments[i-1]

    test_data = hw5_dataset(False, test_comments, None, myw2v)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_test)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.cuda()

    model.eval()
    results = []
    with torch.no_grad():
        for comments in test_loader:
            if use_gpu:
                comments = comments.cuda()
            outputs = model(comments)
            predict = torch.max(outputs, 1)[1]
            results += predict.cpu().tolist()
            
    # write results
    write_ans(results, args.output)
    
