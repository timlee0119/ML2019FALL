import os
from time import time
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from hw3_common import load_data, hw3_dataset, Resnet18

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', help="Required. Path to a folder with all train images.",
                        required=True, type=str)
    parser.add_argument('-l', '--label', help="Required. Path to a .csv file with the train labels.",
                        required=True, type=str)
    parser.add_argument('-o', '--output', help="Required. Path to the output model file.",
                        required=True, type=str)
    parser.add_argument('-e', '--epoch', help="Required. Number of training epoch",
                        required=True, type=int)
    return parser

def dump_model(model, model_path, epoch=None):
    dirname = os.path.dirname(model_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    p = model_path if not epoch else (model_path + "_epoch" + str(epoch))
    torch.save(model.state_dict(), p)
    print('Model saved to %s' % p)

if __name__ == '__main__':
    args = build_argparser().parse_args()

    use_gpu = torch.cuda.is_available()
    print(("Using GPU" if use_gpu else "Using CPU"))

    train_set = load_data(args.input, args.label)
    transform = transforms.Compose([
        # transforms.RandomAffine(15, translate=(0.1,0.1), scale=(0.9,1.1), shear=10, fillcolor=0),
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5], inplace=False)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = hw3_dataset(train_set, transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    print("Loading pretrained Resnet18 model...", end=' ', flush=True)
    model = Resnet18()
    print("Finished!")

    if use_gpu:
        model.cuda()
    optimizer = Adam(model.parameters(), lr=0.0005)
    loss_fn = nn.CrossEntropyLoss()

    print("Start training...")
    train_start_time = time()
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
        print("Epoch: {}, train Loss: {:.4f}, train Acc: {:.4f}".format(epoch + 1, np.mean(train_loss), np.mean(train_acc)))

        if epoch % 10 == 9:
            dump_model(model, args.output, epoch)
    dump_model(model, args.output)

    print("Finished!")
    training_time = (time() - train_start_time) / 60
    print("Total training time: {:.2f} minutes".format(train_start_time))
