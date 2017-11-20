import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

cuda = torch.cuda.is_available()

# params
Z_dim = 20
X_dim = 1200
S_dim = 20
h1_dim = 800
h2_dim = 600

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # qz
        self.fc4 = nn.Linear(X_dim, h1_dim)
        self.fc5 = nn.Linear(h1_dim, h2_dim)
        self.fc6m = nn.Linear(h2_dim, Z_dim)
        self.fc6v = nn.Linear(h2_dim, Z_dim)

        # pz
        self.fc7 = nn.Linear(Z_dim, h2_dim)
        self.fc8 = nn.Linear(h2_dim, h1_dim)
        self.fc9 = nn.Linear(h1_dim, X_dim)

        # for getting Contrastive loss
        self.distance = nn.PairwiseDistance(2)

        # utils
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.batchNorm1 = nn.BatchNorm1d(h1_dim)
        self.batchNorm2 = nn.BatchNorm1d(h2_dim)

        # params
        self.Z_dim = Z_dim

    def Q_z(self, Xb):
        h3 = self.relu(self.batchNorm1(self.fc4(Xb)))
        h4 = self.relu(self.batchNorm2(self.fc5(h3)))
        z_mean = self.fc6m(h4)
        z_logvar = self.fc6v(h4)
        z = self.sample_z(z_mean, z_logvar)
        return z, z_mean, z_logvar

    def P_x(self, z):
        h5 = self.relu(self.batchNorm2(self.fc7(z)))
        h6 = self.relu(self.batchNorm1(self.fc8(h5)))
        px_logit = self.fc9(h6)
        return px_logit

    def sample_z(self, mu, log_var):
        eps = Variable(torch.randn(mu.size(0), self.Z_dim))
        if cuda:
            eps = eps.cuda()
        return mu + torch.exp(log_var / 2) * eps

    def forward(self, X1, X2):
        z1, zm1, zlogvar1 = self.Q_z(X1.float())
        z2, zm2, zlogvar2 = self.Q_z(X2.float())
        px1 = self.P_x(z1)
        px2 = self.P_x(z2)
        return px1, px2, zm1, zm2, zlogvar1, zlogvar2

    def getEncoderOp(self, X):
        z, zm, zlogvar = self.Q_z1(X.float())
        return z

class contrastiveVAE(nn.Module):
    def __init__(self):
        super(contrastiveVAE, self).__init__()

        # qs
        self.fc1 = nn.Linear(X_dim, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.fc3 = nn.Linear(h2_dim, S_dim[0])

        # qz
        self.fc4 = nn.Linear(X_dim, h1_dim)
        self.fc5 = nn.Linear(h1_dim, h2_dim)
        self.fc6m = nn.Linear(h2_dim, Z_dim)
        self.fc6v = nn.Linear(h2_dim, Z_dim)

        # pz
        self.fc7 = nn.Linear(Z_dim, h2_dim)
        self.fc8 = nn.Linear(h2_dim, h1_dim)
        self.fc9 = nn.Linear(h1_dim, X_dim)

        # for getting Contrastive loss
        self.distance = nn.PairwiseDistance(2)

        # utils
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.batchNorm1 = nn.BatchNorm1d(h1_dim)
        self.batchNorm2 = nn.BatchNorm1d(h2_dim)

        # params
        self.Z_dim = Z_dim

    def Q_s(self, Xb):
        h1 = self.relu(self.batchNorm1(self.fc1(Xb)))
        h2 = self.relu(self.batchNorm2(self.fc2(h1)))
        qs = self.fc3(h2)
        return qs

    def Q_z(self, Xb):
        h3 = self.relu(self.batchNorm1(self.fc4(Xb)))
        h4 = self.relu(self.batchNorm2(self.fc5(h3)))
        z_mean = self.fc6m(h4)
        z_logvar = self.fc6v(h4)
        z = self.sample_z(z_mean, z_logvar)
        return z, z_mean, z_logvar

    def P_x(self, z):
        h5 = self.relu(self.batchNorm2(self.fc7(z)))
        h6 = self.relu(self.batchNorm1(self.fc8(h5)))
        px_logit = self.fc9(h6)
        return px_logit

    def sample_z(self, mu, log_var):
        eps = Variable(torch.randn(mu.size(0), self.Z_dim))
        if cuda:
            eps = eps.cuda()
        return mu + torch.exp(log_var / 2) * eps

    def getDist(self, s1, s2):
        return self.distance(s1, s2)

    def forward(self, X1, X2):
        qs1 = self.Q_s(X1.float())
        qs2 = self.Q_s(X2.float())
        z1, zm1, zlogvar1 = self.Q_z(X1, qs1)
        z2, zm2, zlogvar2 = self.Q_z(X2, qs2)
        px1 = self.P_x(z1, qs1)
        px2 = self.P_x(z2, qs2)
        dist = self.getDist(qs1, qs2)
        return qs1, z1, zm1, zlogvar1, px1, qs2, z2, zm2, zlogvar2, px2, dist

    def getEncoderOp(self, X):
        s = self.Q_s(X.float())
        z, z_mean, z_logvar = self.Q_z(X, s)
        return z, s

