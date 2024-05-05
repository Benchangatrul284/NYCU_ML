import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
from model import SLP, MLP, DNN
from tqdm import tqdm
import matplotlib.pyplot as plt


def train_test_accuracy(lr, epochs, batch_size, num_neurons, model, device, percent=1.0,num_layers=5):
    '''
    input:    lr: learning rate
              epochs: number of epochs
              batch_size: batch size
              num_neurons: number of neurons in the hidden layer
              model: 'slp' or 'mlp' or 'dnn'
              device: 'cuda' or 'cpu'
              percent: the percentage of data used for training
    '''
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainset.data = trainset.data[:int(len(trainset)*percent)]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    if model == 'slp':
        classfier = SLP(num_neorons=num_neurons)
        print('You are using SLP with {} neurons'.format(num_neurons))
    elif model == 'mlp':
        classfier = MLP(num_neorons=num_neurons)
        print('You are using MLP with {} of training data'.format(percent))
    elif model == 'dnn':
        classfier = DNN(num_layers=num_layers)
        print('You are using DNN with {} hidden layers'.format(num_layers))
    else:
        raise ValueError('model should be either slp or mlp or dnn')
    
    classfier.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classfier.parameters(), lr=lr)
    
    classfier.train()
    for epoch in range(1,epochs + 1):
        print(f'Epoch {epoch}/{epochs}')
        classfier.train()
        train_loss = 0
        for images, labels in tqdm(trainloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = classfier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f'Training loss: {train_loss/len(trainloader)}')
    
    # after training the model, we need to test the model
    with torch.no_grad():
        classfier.eval()
        # calculate the accuracy of the training set
        correct = 0
        total = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            outputs = classfier(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_accuracy = correct/total
        print(f'Training accuracy: {train_accuracy}')
        
        # calculate the accuracy of the testing set
        correct = 0
        total = 0
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = classfier(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_accuracy = correct/total
        print(f'Testing accuracy: {test_accuracy}')
    return train_accuracy, test_accuracy
    
    
        
        
