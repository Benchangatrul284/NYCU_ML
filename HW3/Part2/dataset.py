import torch
from torch.utils.data import Dataset
import pandas as pd

class HW2Dataset(Dataset):
    def __init__(self,path='HW2_training.csv'):
        self.data = pd.read_csv(path)
        self.num_teams = self.data['Team'].nunique()
        self.num_features = len(self.data.columns) - 1
        self.features = self.data.iloc[:,1:].values
        self.labels = self.data.iloc[:,0].values
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx],dtype=torch.float)
        label = torch.tensor(self.labels[idx],dtype=torch.long)
        return feature, label
    
if __name__ == '__main__':
    train_dataset = HW2Dataset(path = 'HW2_training.csv')
    test_dataset = HW2Dataset(path = 'HW2_testing.csv')
    print('Train data has {} samples'.format(train_dataset.__len__()))
    print('Test data has {} samples'.format(test_dataset.__len__()))