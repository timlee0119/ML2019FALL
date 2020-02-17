import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # define: encoder
        self.encoder = nn.Sequential(
            # in_channel, out_channel, kernel_size, stride, padding
#             nn.Conv2d(3, 8, 3, 2, 1),
#             nn.Conv2d(8, 16, 3, 2, 1),
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # 32, 32, 32
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2), # 32, 16, 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 64, 16, 16
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2), # 64, 8, 8       
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2), # 128, 4, 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2) # 256, 2, 2
        )

        # define: decoder
        self.decoder = nn.Sequential(
#           nn.ConvTranspose2d(16, 8, 2, 2),
#           nn.ConvTranspose2d(8, 3, 2, 2),
            nn.ConvTranspose2d(256, 128, 2, 2),  #128, 4, 4
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 2, 2),  #64, 8, 8
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 2, 2),  #32, 16, 16
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 3, 2, 2),  #3,32,32
            nn.BatchNorm2d(3),
            nn.Tanh(),
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(2*2*256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),        
            nn.Linear(128, 50),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(50, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),            
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 2*2*256),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = encoded.view(-1, 2*2*256)
        encoded = self.fc1(encoded)
        decoded = self.fc2(encoded)
        decoded = decoded.view(60, 256, 2, 2)
        decoded = self.decoder(decoded)

        # Total AE: return latent & reconstruct
        return encoded, decoded
    