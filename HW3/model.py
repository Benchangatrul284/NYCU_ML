import torch
import torch.nn as nn

class SLP(nn.Module):
    def __init__(self,num_neorons=5):
        super(SLP, self).__init__()
        self.fc1 = nn.Linear(28*28, num_neorons)
        self.fc2 = nn.Linear(num_neorons, 10)
        
    def forward(self, x):
        x = x.view(-1, 28*28) # flatten the image
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MLP(nn.Module):
    def __init__(self,num_neorons=100):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, num_neorons)
        self.fc2 = nn.Linear(num_neorons, num_neorons)
        self.fc3 = nn.Linear(num_neorons, 10)
        
    def forward(self, x):
        x = x.view(-1, 28*28) # flatten the image
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DNN(nn.Module):
    def __init__(self,num_layers=5):
        # number of layers means the number of hidden layers
        super(DNN, self).__init__()
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(28*28, 100))
        for i in range(num_layers):
            self.fc.append(nn.Linear(100, 100))
        self.fc.append(nn.Linear(100, 10))
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        for i in range(len(self.fc)-1):
            x = torch.relu(self.fc[i](x))
        x = self.fc[-1](x)
        return x
        
if __name__ == '__main__':
    slp = SLP()
    mlp = MLP()
    dnn = DNN()
    print(slp)
    print(mlp)
    print(dnn)