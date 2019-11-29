import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from hw4_common import Autoencoder

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', help="Required. Path to trainX.npy.", required=True, type=str)
    parser.add_argument('-o', '--output', help="Required. Path to the result .csv file.", required=True, type=str)
    parser.add_argument('-m', '--model', help="Reuqired. Path to the trained model.", required=True, type=str)
    return parser

def write_ans(result, path):
    print(f'Writing results to {path}')
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
        
    df = pd.DataFrame({'id': np.arange(0, len(result)), 'label': result})
    df.to_csv(path, index=False)

if __name__ == '__main__':
    args = build_argparser().parse_args()
    
    use_gpu = torch.cuda.is_available()
    print(("Using GPU" if use_gpu else "Using CPU"))

    # Load trained model
    autoencoder = Autoencoder()
    autoencoder.load_state_dict(torch.load(args.model))
    # Load images and normalize to [-1, 1]
    trainX = np.load(args.input)
    trainX = np.transpose(trainX, (0, 3, 1, 2)) / 255. * 2 - 1
    trainX = torch.Tensor(trainX)
    
    if use_gpu:
        autoencoder.cuda()
        trainX = trainX.cuda()

    test_dataloader = DataLoader(trainX, batch_size=60, shuffle=False)
        
    # Collect the latents and standardize it
    autoencoder.eval()
    latents = []
    reconstructs = []
    for x in test_dataloader:
        latent, reconstruct = autoencoder(x)
        latents.append(latent.cpu().detach().numpy())
        reconstructs.append(reconstruct.cpu().detach().numpy())
        
    latents = np.concatenate(latents, axis=0).reshape([9000, -1])
    latents = (latents - np.mean(latents, axis=0)) / np.std(latents, axis=0)
    
    # Dimension reduction #2
    #print('Starting PCA...')
    #latents = PCA(n_components=32).fit_transform(latents)
    #print('Starting TSNE...')
    #latents = TSNE(n_components=8, method='exact').fit_transform(latents) 
    latents = TSNE(n_components=2).fit_transform(latents)

    # Clustering!
    result = KMeans(n_clusters=2).fit(latents).labels_
    #result = SpectralClustering(n_clusters=2, eigen_solver='arpack', affinity="nearest_neighbors").fit(latents).labels_
    
    # We know first 5 labels are zeros, it's a mechanism to check are your answers
    # need to be flipped or not.
    if np.sum(result[:5]) >= 3:
        result = 1 - result
    
    # Generate submission
    write_ans(result, args.output)   
