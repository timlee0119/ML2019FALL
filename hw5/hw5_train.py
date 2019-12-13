import os
from datetime import datetime
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from gensim.models import Word2Vec
from hw5_common import RNN_LSTM, hw5_dataset, pad_collate, word_segmentation

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument('--trainx', help="Required. Path to train_x.csv.", required=True, type=str)
    parser.add_argument('--trainy', help="Required. Path to train_y.csv.", required=True, type=str)
    parser.add_argument('--testx', help="Required. Path to test_x.csv.", required=True, type=str)

    return parser

def data_preprocess(trainx, trainy, testx):
    print('Start data preprocessing...')
    train_x = pd.read_csv(trainx)
    train_y = pd.read_csv(trainy)
    test_x = pd.read_csv(testx)
    
    train_comments_tmp = word_segmentation(train_x.comment)
    train_labels_tmp = train_y.label.tolist()
    
    # remove all empty comments in train data
    train_comments = []
    train_labels = []
    for i in range(len(train_comments_tmp)):
        if len(train_comments_tmp[i]) != 0:
            train_comments.append(train_comments_tmp[i])
            train_labels.append(train_labels_tmp[i])
    
    test_comments = word_segmentation(test_x.comment)
    return train_comments, train_labels, test_comments

def train_word2vec_model(all_comments, embedding_dim, ep):
    print('Start training word2vec model...')
    model_path = f'models/myword2vec_{embedding_dim}.model'
    if os.path.exists(model_path):
        myw2v = Word2Vec.load(model_path)
    else:
        myw2v = Word2Vec(all_comments, min_count = 1, size = embedding_dim)
        myw2v.train(all_comments, total_examples = myw2v.corpus_count, epochs = 300)
        myw2v.save(model_path)

    return myw2v

def train_test_split(trainX, trainY, valid_size, random_state):
    assert len(trainX) == len(trainY)
    np.random.seed(random_state)
    valid_len = int(len(trainX) * valid_size)
    valid_set = set(np.random.choice(len(trainX), valid_len))
    train_set = set(range(len(trainX))) - valid_set
    
    return np.array(trainX)[list(train_set)].tolist(), np.array(trainX)[list(valid_set)].tolist(), \
           np.array(trainY)[list(train_set)].tolist(), np.array(trainY)[list(valid_set)].tolist()

def plot_history(history):
    import matplotlib.pyplot as plt

    epoch = len(history['train_acc'])
    x = list(range(1, epoch+1))
    # plot loss
    plt.plot(x, history['val_loss'], color='red', label='validation loss')
    plt.plot(x, history['train_loss'], label='training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()
    plt.savefig('loss_vs_epoch.png')
    plt.cla()
    # plot acc
    plt.plot(x, history['val_acc'], color='red', label='validation accuracy')
    plt.plot(x, history['train_acc'], label='training accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.legend(loc='lower right')
    plt.savefig('acc_vs_epoch.png')

if __name__ == '__main__':
    ####### hyper parameters #######
    embedding_dim = 200
    hidden_size = 96
    batch_size = 64
    lr = 5e-4
    epochs = 50
    bd = False
    n_l = 3
    dp = 0.8
    ################################
    
    args = build_argparser().parse_args()
    
    train_comments, train_labels, test_comments = data_preprocess(args.trainx, args.trainy, args.testx)
    all_comments = train_comments + test_comments
    
    # Train word2vec model
    myw2v = train_word2vec_model(all_comments, embedding_dim, 300)
    
    # Create Datasets and DataLoaders
    trainX, validX, trainY, validY = train_test_split(train_comments, train_labels, 0.2, 42)
    train_data = hw5_dataset(True, trainX, trainY, myw2v)
    valid_data = hw5_dataset(True, validX, validY, myw2v)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)
    
    # Start training & validating
    model = RNN_LSTM(embedding_dim, hidden_size=hidden_size, bidirectional=bd, num_layers=n_l, dropout=dp)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.cuda()
        
    history = {'val_loss': [], 'val_acc': [], 'train_loss': [], 'train_acc': []}
    for epoch in range(epochs):
        model.train()
        train_loss = []
        train_acc = []
        for comments, labels in train_loader:
            if use_gpu:
                comments = comments.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(comments)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            predict = torch.max(outputs, 1)[1]
            acc = np.mean((labels == predict).cpu().numpy())
            train_acc.append(acc)
            train_loss.append(loss.item())

    #     if epoch % 10 == 0:
        train_loss = np.mean(train_loss)
        train_acc = np.mean(train_acc)
        print(f'Epoch: {"%03d" % (epoch)}, train loss: {"%.4f" % train_loss}, train acc: {"%.4f" % train_acc}')
        torch.save(model.state_dict(), f'models/rnn_epoch{epoch}_{datetime.now().strftime("%m%d%H%M")}')

        # validation
        model.eval()
        with torch.no_grad():
            valid_loss = []
            valid_acc = []
            for comments, labels in valid_loader:
                if use_gpu:
                    comments = comments.cuda()
                    labels = labels.cuda()
                outputs = model(comments)
                loss = loss_fn(outputs, labels)
                predict = torch.max(outputs, 1)[1]
    #             print(predict)
                acc = np.mean((labels == predict).cpu().numpy())
                valid_acc.append(acc)
                valid_loss.append(loss.item())
            valid_loss = np.mean(valid_loss)
            valid_acc = np.mean(valid_acc)
            print(f'Epoch: {"%03d" % (epoch)}, valid loss: {"%.4f" % valid_loss}, valid acc: {"%.4f" % valid_acc}')

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)        
            history['val_loss'].append(valid_loss)
            history['val_acc'].append(valid_acc) 
        
    plot_history(history)
