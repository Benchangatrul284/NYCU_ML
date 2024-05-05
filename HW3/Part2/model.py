import torch
import torch.nn as nn

class HW2model(nn.Module):
    def __init__(self,num_features,num_teams,num_neurons=100,num_layers=5):
        super(HW2model, self).__init__()
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(num_features, num_neurons))
        for i in range(num_layers):
            self.fc.append(nn.Linear(num_neurons, num_neurons))
        self.fc.append(nn.Linear(num_neurons, num_teams))
        
    def forward(self, x):
        x = x.view(-1, 2)
        for i in range(len(self.fc)-1):
            x = torch.relu(self.fc[i](x))
        x = self.fc[-1](x)
        return x
    
    
if __name__ == '__main__':
    model = HW2model(2,5)