import torchfile
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from models import contrastiveVAE
from utils import mmd

cuda = torch.cuda.is_available()
if cuda:
    torch.cuda.manual_seed(10)
else:
    torch.manual_seed(10)

#load the data <Private data>
# qrsData = torchfile.load('')
# qrsPairs = torchfile.load('')
pairNum = 343434 #len(qrsPairs)
dataNum = 10292 #len(qrsData.train_x)


#params
mb_size = 200
Z_dim = 20
X_dim = 1200
S_dim = 20
h1_dim = 800
h2_dim = 600
lr = 1e-4

#hyper-parameters
marginPA = 15
alphaHinge = 0.2
alphaKL = 3

model = contrastiveVAE()
if cuda:
    model = model.cuda()

recon_loss = nn.MSELoss()
recon_loss.size_average = False
hinge_loss = nn.HingeEmbeddingLoss(marginPA)

def loss_function(recon_x1, recon_x2, x1, x2, mu1, mu2, logvar1, logvar2, VTlabel, z1, z2):
    x1 = x1.float()
    x2 = x2.float()
    reconLoss = (recon_loss(recon_x1, x1) + recon_loss(recon_x2, x2)) / mb_size
    KLLoss = torch.mean(0.5 * torch.sum(torch.exp(logvar1) + mu1**2 - 1. - logvar1, 1)) + torch.mean(0.5 * torch.sum(torch.exp(logvar2) + mu2**2 - 1. - logvar2, 1))

    #mmd
    x1x1, x1x2, x2x2 = mmd(z1, z2, 1.0)
    labels = (VTlabel -1 ) / (-2)
    mmdLoss = (torch.mean(x1x1 * labels)  - 2 * torch.mean(x1x2 * labels)  + torch.mean(x2x2 * labels))

    return reconLoss, KLLoss, mmdLoss

solver = optim.Adam(model.parameters(), lr=lr)
model.train()

#Define variables
X1 = np.ndarray(shape=(mb_size,1200))
X2 = np.ndarray(shape=(mb_size,1200))
VTlabel = np.ndarray(shape=(mb_size))

XX1 = np.ndarray(shape=(mb_size,1200))
XX2 = np.ndarray(shape=(mb_size,1200))
VTTlabel = np.ndarray(shape=(mb_size))

#Throw away lag data
sampleEachEpoch =  dataNum - dataNum % mb_size

totalLoss = []
for it in range(10):
    #data
    for i in range(0, sampleEachEpoch, mb_size):
        shuffle = torch.randperm(pairNum)[0:sampleEachEpoch]

        k = 0
        #training one epoch
        # for j in range(i, i+mb_size):
        #     XX1[k] = qrsData.train_x[int(qrsPairs[shuffle[j]][0])]
        #     XX2[k] = qrsData.train_x[int(qrsPairs[shuffle[j]][1])]
        #     VTTlabel[k] = qrsPairs[shuffle[j]][2]
        #     k = k + 1
        #
        # #converting numpy to format acceptable by PyTorch
        # X1 = Variable(torch.from_numpy(XX1))
        # X2 = Variable(torch.from_numpy(XX2))
        # VTlabel = Variable(torch.from_numpy(VTTlabel.astype('float32')))

        #To demonstrate the working setup without the real-data
        X1 = Variable(torch.randn(mb_size, X_dim))
        X2 = Variable(torch.randn(mb_size, X_dim))
        VTlabel = Variable(torch.ones(mb_size)) #Actually VTlabel in our real dataset is in the form of 1 and -1.
        if cuda:
            X1 = X1.cuda()
            X2 = X2.cuda()
            VTlabel = VTlabel.cuda()

        #zero grad
        solver.zero_grad()

        # Forward data
        _, z1, zm1, zlogvar1, px1, _, z2, zm2, zlogvar2, px2, _ = model(X1, X2)

        # Calculate loss
        reconLoss, KLLoss, mmdLoss = loss_function(px1, px2, X1, X2, zm1, zm2, zlogvar1, zlogvar2,
                                                   VTlabel, z1, z2)

        # Backward
        loss = (reconLoss + 1 * KLLoss + mmdLoss) / 2
        loss.backward()

        #Update
        solver.step()

    if it % 5 == 0:
        print('Iter-{}; Loss: {:.4}'.format(it, loss.data[0]))
        print('Iter-{}; Recon: {:.4}'.format(it, reconLoss.data[0]))
        print('Iter-{}; KLLoss: {:.4}'.format(it, KLLoss.data[0]))
        print('Iter-{}; mmdLoss: {:.4}'.format(it, mmdLoss.data[0]))

    if it % 50 == 0:
        torch.save(model.state_dict(), 'model_MMDVAE.pt')
        totalLoss.append(loss.data[0])
        np.save('MMDVAELoss.npy', totalLoss)
