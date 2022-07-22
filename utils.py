import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
from torch import tensor
from numpy.linalg import inv
import pandas as pd
from scipy.stats import norm
from scipy.cluster.vq import kmeans2
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import random

device = 'cpu'
# device = 'cuda:13'
def Pi(x, beta, mu):
        return np.asarray((np.exp(x @ beta + mu) / (1 + np.exp(x @ beta + mu))))
dtype = torch.FloatTensor
class dMEGA:
    def __init__(self, x, y):
        self.input_size = x[0].shape[1]
        var_name = []
        for i in range(self.input_size):
            var_name += ['X' + str(i+1)]
        self.var_name = var_name
        self.x = x
        self.y = y
        if isinstance(x[0], pd.DataFrame):
            self.var_name = x[0].columns
            self.x = [Variable(torch.from_numpy(np.array(data).astype(np.float32)).to(device)) for data in x]
            self.y = [Variable(torch.from_numpy(np.array(data).reshape(len(data),1).astype(np.float32)).to(device)) for data in y]
        self.site = len(x)
        self.criterion = nn.BCELoss()
        self.beta = Variable(torch.zeros(self.input_size,1).type(dtype).to(device), requires_grad=True)
        self.mu = [Variable(torch.randn(1,1).type(dtype).to(device), requires_grad=True) for i in range(self.site)]
        self.df = pd.DataFrame
        self.converge = False
        self.loss = 10**10
        self.iter = 0
        self.sample_size = sum([len(i) for i in self.y])

    def fit(self):
        old_mu=[np.array([[0]],dtype='float32')]*self.site
        learning_rate = 0.1
        for it_mu in range(50):
            mu = []
            for site in range(self.site):
#                 print(f'===Now in site:{site}===')
                params = [self.mu[site]]
#                 print(f'Local site{site} mu start at:{params}')
                solver = optim.Adam(params, lr=learning_rate) # 1.Adam (tried)
                old_loss = 10**10
                for n in range(1000):
                    y_pred = torch.sigmoid(self.x[site].mm(self.beta) + self.mu[site])
                    loss = self.criterion(y_pred, self.y[site])
                    if abs(old_loss - loss) < 1e-3:
                        break;
                    else:
                        old_loss = loss
                        loss.backward(retain_graph=True)
                        solver.step()
                        solver.zero_grad()
                mu += [self.mu[site].cpu().detach().numpy()]
            self.mu = [Variable(tensor(x).type(dtype).to(device), requires_grad=True) for x in mu]
            if max(abs(np.array(mu)-old_mu))<1e-6:
                break;
            else:
                old_mu = mu
            
            for epoch in range(100):
                score = 0
                for site in range(self.site):
                    y_pred = torch.sigmoid(self.x[site].mm(self.beta) + self.mu[site])
                    loss = self.criterion(y_pred, self.y[site])
                    score += grad(loss, self.beta, create_graph=True)[0] * len(self.y[site])/self.sample_size
                score = score.reshape(1,len(score))
                
                hess = []
                for i in range(score.shape[1]):
                    grad2 = grad(score[0][i], self.beta, retain_graph=True)[0].squeeze()
                    hess.append(grad2)
                hessian = torch.stack(hess) + torch.diag(torch.tensor(.0001).repeat(len(score))).to(device)

                inv_hess = torch.inverse(hessian)
                
                
                self.beta = self.beta - (score @ inv_hess).reshape(self.beta.shape)
                
                
                if max(abs((score @ inv_hess)[0])) < 1e-6:
                    self.converge = True
                    break;
        
        if self.converge:
            beta = self.beta.cpu().detach().numpy().reshape(self.input_size,1)
            x_concat = torch.cat(self.x).cpu().numpy()
            y_concat = torch.cat(self.y).cpu().numpy()
            s = []
            for site in range(self.site):
                score = Pi(self.x[site].cpu().numpy(), beta, self.mu[site].cpu().detach().numpy())
                s += [score * (1-score)]
            V = np.diagflat(np.concatenate(s))
#             V = np.diagflat(Pi(x_concat, beta) * (1 - Pi(x_concat, beta)))
            SE = np.sqrt(np.diag(inv(np.transpose(x_concat) @ V @ x_concat\
                                     + np.diagflat(np.repeat(0,len(beta)))))).reshape(self.input_size,1)
            Z = beta/SE
            P = 2 * norm.cdf(-1 * np.abs(Z))
            CI_025  = beta - 1.959964 * SE
            CI_975  = beta + 1.959964 * SE
            self.iter = epoch
            self.df = pd.DataFrame({'Coef': np.transpose(beta)[0], 'Std.Err': np.transpose(SE)[0],
                                    'z': np.transpose(Z)[0], 'P-value': np.transpose(P)[0],
                                    '[0.025': np.transpose(CI_025)[0], '0.975]': np.transpose(CI_975)[0]},
                                    index = self.var_name)
            return self
        else:
            print('=================================================\nThe federated GLMM algorithm failed to converge!!\n=================================================')
