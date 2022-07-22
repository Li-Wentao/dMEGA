import torch
from utils import dMEGA

# Load 3 toy datasets
X1 = torch.load('toy_data/X1.pt'); y1 = torch.load('toy_data/y1.pt')
X2 = torch.load('toy_data/X2.pt'); y2 = torch.load('toy_data/y2.pt')
X3 = torch.load('toy_data/X3.pt'); y3 = torch.load('toy_data/y3.pt')
X = [X1, X2, X3]
y = [y1, y2, y3]

# run 
model = dMEGA(X, y).fit()

# Fixed-effects coefficients
print(model.df)

# Mixed-effects coefficients
print(model.mu)