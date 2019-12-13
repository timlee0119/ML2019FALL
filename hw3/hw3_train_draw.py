import os
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from hw3_common import load_data_valid, hw3_dataset, Resnet18
import matplotlib.pyplot as plt

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', help="Required. Path to a folder with all train images.",
                        required=True, type=str)
    parser.add_argument('-l', '--label', help="Required. Path to a .csv file with the train labels.",
                        required=True, type=str)
    parser.add_argument('-e', '--epoch', help="Required. Number of training epoch",
                        required=True, type=int)
    parser.add_argument('-r', '--lr', help="Required. Learning rate",
                        required=True, type=float)

    return parser

def plot_history(history):
    import matplotlib.pyplot as plt
    %matplotlib inline

    epoch = len(history['train_acc'])
    x = list(range(1, epoch+1))
    # plot loss
    plt.plot(x, history['val_loss'], color='red', label='validation loss')
    plt.plot(x, history['train_loss'], label='training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()
    plt.show()
    plt.cla()
    # plot acc
    plt.plot(x, history['val_acc'], color='red', label='validation accuracy')
    plt.plot(x, history['train_acc'], label='training accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.legend(loc='lower right')
    plt.show()
    plt.cla()

if __name__ == '__main__':
    args = build_argparser().parse_args()

    use_gpu = torch.cuda.is_available()
    print(("Using GPU" if use_gpu else "Using CPU"))

    train_set, valid_set = load_data_valid(args.input, args.label)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5], inplace=False)
    ])

    train_dataset = hw3_dataset(train_set, transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    valid_dataset = hw3_dataset(valid_set, transform)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)

    print("Loading pretrained Resnet18 model...", end=' ', flush=True)
    model = Resnet18()
    print("Finished!")

    if use_gpu:
        model.cuda()
    optimizer = Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    print("Start training...")
    history = {'val_loss': [], 'val_acc': [], 'train_loss': [], 'train_acc': []}
    for epoch in range(args.epoch):
        model.train()
        train_loss = []
        train_acc = []
        for idx, (imgs, labels) in enumerate(train_loader):
            if use_gpu:
                imgs = imgs.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            output = model(imgs)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            predict = torch.max(output, 1)[1]
            acc = np.mean((labels == predict).cpu().numpy())
            train_acc.append(acc)
            train_loss.append(loss.item())
        train_loss = np.mean(train_loss)
        train_acc = np.mean(train_acc)
        print("Epoch: {}, train Loss: {:.4f}, train Acc: {:.4f}".format(epoch + 1, train_loss, train_acc))

        # validation
        model.eval()
        with torch.no_grad():
            valid_loss = []
            valid_acc = []
            for idx, (imgs, labels) in enumerate(valid_loader):
                if use_gpu:
                    imgs = imgs.cuda()
                    labels = labels.cuda()
                output = model(imgs)
                loss = loss_fn(output, labels)
                predict = torch.max(output, 1)[1]
                acc = np.mean((labels == predict).cpu().numpy())
                valid_loss.append(loss.item())
                valid_acc.append(acc)
            valid_loss = np.mean(valid_loss)
            valid_acc = np.mean(valid_acc)
            print("Epoch: {}, valid Loss: {:.4f}, valid Acc: {:.4f}".format(epoch + 1, valid_loss, valid_acc))

        # update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)        
        history['val_loss'].append(valid_loss)
        history['val_acc'].append(valid_acc)

    print("Finished!")

    assert args.epoch == len(history['train_acc']), "history length equals args.epoch"
    print("Drawing train/valid history plot")
    plot_history(history)
