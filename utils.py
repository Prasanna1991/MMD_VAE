import torch
import numpy as np


def binarize(X):
	Xb = np.greater(X, np.random.uniform(0.0, 1.0, X.shape))
	return Xb.astype(float)

def convertToOneHot(y):
	c = np.ndarray(shape=(10))
	c.fill(0)
	c[int(y) - 1] = int(1.0)
	return c

def mmd(x1, x2, beta):
	x1x1 = gaussKernel(x1, x1, beta)
	x1x2 = gaussKernel(x1, x2, beta)
	x2x2 = gaussKernel(x2, x2, beta)
	return x1x1, x1x2, x2x2

def gaussKernel(x1, x2, beta = 1.0):
	return torch.sum(torch.exp(-beta * ((x1 - x2) * (x1 - x2))), 1)

#This is from: https://discuss.pytorch.org/t/maximum-mean-discrepancy-mmd-and-radial-basis-function-rbf/1875
def MMD1(x1, x2, alpha, label):
    n = x1.size(0) #batchsize
    x1x1, x1x2, x2x2 = torch.mm(x1, x1.t()), torch.mm(x1, x2.t()), torch.mm(x2, x2.t()) # x1x1 = n x n

    r1 = (x1x1.diag().unsqueeze(0).expand_as(x1x1))
    r2 = (x2x2.diag().unsqueeze(0).expand_as(x2x2))

    #Gaussian kernel
    K = torch.exp(- alpha * (r1.t() + r1 - 2*x1x1))
    L = torch.exp(- alpha * (r2.t() + r2 - 2*x2x2))
    P = torch.exp(- alpha * (r1.t() + r2 - 2*x1x2))

    beta = (1./(n *(n-1)))
    gamma = (2./(n*n))
    return beta * (torch.sum(K * label) + torch.sum(L * label)) - gamma * (torch.sum(P * label))
