import torch.nn.functional as F
from torch.autograd import Variable
import torch 
import torch.nn as nn
class Vanilla_Autoencoder(nn.Module):
    def __init__(self, input_size, hidden=10):
        super(Vanilla_Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size,512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, hidden)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, input_size)
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    def encode(self,x):
        return self.encoder(x)
