import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_first =True




class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size ,num_classes , rnndropout):
        super().__init__()
        self.hidden_size = hidden_size

        self.drop = nn.Dropout(rnndropout)
        self.fc = nn.Linear(hidden_size, num_classes) 
        self.rnn = nn.LSTM(input_size,hidden_size,batch_first=True)
        self.device='cpu'



    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(self.device) 
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(self.device)
        h0 = h0.float()
        c0 = c0.float()
        x=x.float()
        x = self.drop(x)
        output, _ = self.rnn(x, (h0, c0))
        output = self.drop(output)
        output=output[:,-1,:]
        out = self.fc(output)
        out =F.softmax(out, dim=1)
        return out

#def fit(model, train_loader, val_loader, num_epochs: int = 1500,
#            val_size: float = 0.2, learning_rate: float = 0.001,
#           patience: int = 100) -> None: # patience war 10 

#	optimizerTimeAtten = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
#
#
#	saveModelName="../Models/"+models[m]+"/"+modelName
#	saveModelBestName =saveModelName +"_BEST.pkl"
#	saveModelLastName=saveModelName+"_LAST.pkl"
#

#
#
#
#
#				total_step = len(train_loaderRNN)
#				Train_acc_flag=False
#				Train_Acc=0
#				Test_Acc=0
#				BestAcc=0
#				BestEpochs = 0
#				patience=200
#
#				for epoch in range(args.num_epochs):
#					noImprovementflag=True
#					for i, (samples, labels) in enumerate(train_loaderRNN):
##
#						net.train()
#						samples = samples.reshape(-1, args.NumTimeSteps, args.NumFeatures).to(device)
#						samples = Variable(samples)
#						labels = labels.to(device)
#						labels = Variable(labels).long()
#
#						outputs = net(samples)
#						loss = criterion(outputs, labels)
#
#						optimizerTimeAtten.zero_grad()
#						loss.backward()
#						optimizerTimeAtten.step()
#
#						if (i+1) % 3 == 0:
#							Test_Acc = checkAccuracy(test_loaderRNN, net,args)
#							Train_Acc = checkAccuracy(train_loaderRNN, net,args)
#							if(Test_Acc>BestAcc):
#								BestAcc=Test_Acc
#								BestEpochs = epoch+1
#								torch.save(net, saveModelBestName)
#								noImprovementflag=False
#
#							print ('{} {}-->Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Train Accuracy {:.2f}, Test Accuracy {:.2f},BestEpochs {},BestAcc {:.2f} patience {}' 
#							   .format(args.DataName, models[m] ,epoch+1, args.num_epochs, i+1, total_step, loss.item(),Train_Acc, Test_Acc,BestEpochs , BestAcc , patience))
#						if(Train_Acc>=99 or BestAcc>=99 ):
#							torch.save(net,saveModelLastName)
##							Train_acc_flag=True
#							break
#
#					if(noImprovementflag):
#						patience-=1
#					else:
#						patience=200
#
					#if(epoch+1)%10==0:
				#		torch.save(net, saveModelLastName)

					#if(Train_acc_flag or patience==0):
				#		break

					#Train_Acc =checkAccuracy(train_loaderRNN , net, args)
					#print('{} {} BestEpochs {},BestAcc {:.4f}, LastTrainAcc {:.4f},'.format(args.DataName, models[m] ,BestEpochs , BestAcc , Train_Acc))
    