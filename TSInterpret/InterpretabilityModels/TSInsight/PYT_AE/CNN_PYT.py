import torch.nn as nn
import torch.nn.functional as F

# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self,feature_number,hidden_size_1=32, hidden_size_2=64, kernal_size=3):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        self.conv1 = nn.Conv1d(feature_number, hidden_size_1, kernal_size, padding=1)  
        self.conv2 = nn.Conv1d(hidden_size_1, hidden_size_2, kernal_size, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        
        ## decoder layers ##
        self.t_conv1 = nn.ConvTranspose1d(hidden_size_2, hidden_size_1, kernal_size-1, stride=2)
        self.t_conv2 = nn.ConvTranspose1d(hidden_size_1, feature_number, kernal_size-1, stride=2)

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation
        
        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        #x =F.tanh(self.t_conv2(x))
        #x=self.linear(self.t_conv2(x))
        x=self.t_conv2(x)   
        #y = y.view(-1, x.size(1), y.size(-1))
        #print(x.shape)     
        return x
