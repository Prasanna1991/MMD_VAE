import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from models import VAE, contrastiveVAE

cuda = torch.cuda.is_available()

#load the data <Private data>
# qrsData = torchfile.load('')
dataNum = 10292 #len(qrsData.train_x)
dataNum_val = 3017 #len(qrsData.val_x)
dataNum_test = 3538 #len(qrsData.test_x)

#load the trained model
# 1. VAE
model = VAE()
model.load_state_dict(torch.load('model_VAE.pt'))

#2. MMDVAE and contrastiveMMDVAE
# model = contrastiveVAE()
# model.load_state_dict(torch.load('model_contrastiveMMDVAE.pt'))
# model.load_state_dict(torch.load('model_MMDVAE.pt'))

# params
evallr = 0.1
X_dim = 1200
Z_dim = 20
y_dim = 10
max_iterE = 25

model.eval()
if cuda:
    model = model.cuda()

# get learned Z
# 1. VAE
train_z = model.getEncoderOp(Variable(torch.randn(dataNum, X_dim)).cuda())
val_z = model.getEncoderOp(Variable(torch.randn(dataNum_val, X_dim)).cuda())
test_z = model.getEncoderOp(Variable(torch.randn(dataNum_test, X_dim)).cuda())

#2. MMDVAE and contrastiveMMDVAE
# train_z, train_s = model.getEncoderOp(Variable(torch.randn(dataNum, X_dim)).cuda())
# val_z, train_s = model.getEncoderOp(Variable(torch.randn(dataNum_val, X_dim)).cuda())
# test_z, train_s = model.getEncoderOp(Variable(torch.randn(dataNum_test, X_dim)).cuda())


# getLabels
trainLabel = np.random.randint(10, size=(dataNum)) #qrsData.train_y[:, 0]
valLabel = np.random.randint(10, size=(dataNum_val)) #qrsData.val_y[:, 0]
testLabel = np.random.randint(10, size=(dataNum_test)) #qrsData.test_y[:, 0]

# model [Linear classifier]
class evalContrVAE(nn.Module):
    def __init__(self):
        super(evalContrVAE, self).__init__()
        self.fcA = nn.Linear(Z_dim, y_dim)

    def forward(self, X):
        X = X.float()
        return self.fcA(X)

evalmodel = evalContrVAE()
if cuda:
    evalmodel = evalmodel.cuda()
evalCriterion = nn.CrossEntropyLoss()
evalmodel.train()

losses = []
running_loss = 0.0
running_corrects = 0

accuracy = 0.0
last_accuracy = 0.0
threshold = 3
decreasing = 0
evalsolver = optim.Adam(evalmodel.parameters(), lr=evallr, weight_decay=0.001)

# to store all the results
valAcc = []
testAcc = []
for it_eval in range(500):
    print('eval: '.format(it_eval))

    evalsolver.zero_grad()

    # forward train data
    output_val = evalmodel(train_z)

    # preprocessing for target variable
    target = trainLabel
    targetLong = Variable(torch.from_numpy(target.astype('float32'))).long()
    if cuda:
        targetLong = targetLong.cuda()

    # loss
    __, preds = torch.max(output_val.data, 1)
    evalLoss = evalCriterion(output_val, targetLong)
    evalLoss.backward(retain_variables=True)
    evalsolver.step()

    # statistics
    running_loss = evalLoss.data[0]
    running_corrects = torch.sum(preds == torch.from_numpy(target).long())
    trainAcc = float(running_corrects) / len(train_z)

    # forward Val data
    valOutput = evalmodel(val_z)
    __, predsVal = torch.max(valOutput.data, 1)
    hit = torch.sum(predsVal == torch.from_numpy(np.subtract(valLabel, 1)).long())
    accuracy = float(hit) / len(val_z)

    # forward Test data
    testOutput = evalmodel(test_z)
    __, predsTest = torch.max(testOutput.data, 1)
    hitTest = torch.sum(predsTest == torch.from_numpy(np.subtract(testLabel, 1)).long())
    totalTestAcc = float(hitTest) / len(test_z)

    #adaptive learning rate based on validation data
    if accuracy < last_accuracy:
        if decreasing > threshold:
            evallr = evallr / 2
        else:
            decreasing = decreasing + 1
    else:
        decreasing = 0
    valAcc.append(accuracy)
    testAcc.append(totalTestAcc)

torch.save(model.state_dict(), 'trainedModel/evalmodel_VAE.pt')
print(max(valAcc))
print(max(testAcc))
