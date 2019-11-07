import os
import random
import glob
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.models as models
from PIL import Image

def load_data(img_path, label_path = None):
    images = sorted(glob.glob(os.path.join(img_path, '*.jpg')))
    if label_path:
        labels = pd.read_csv(label_path)
        labels = labels.iloc[:,1].values.tolist()
    else:
        labels = [0 for x in range(len(images))]
    
    data = list(zip(images, labels))
    if label_path:
        random.seed(42)
        random.shuffle(data)
    
    return data

class hw3_dataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img = Image.open(self.data[idx][0]).convert('RGB')
        img = self.transform(img)
        label = self.data[idx][1]
        return img, label

class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.resnet = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-1])
        # self.fc = nn.Linear(512,7)
        self.fc = nn.Linear(512,128)
        self.fc2 = nn.Linear(128,7)
    def forward(self, x):
        x = self.resnet(x)
        x = x.view(-1, 1*1*512)
        x = self.fc(x)
        x = self.fc2(x)
        
        return x
