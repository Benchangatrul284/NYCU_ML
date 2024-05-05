from model import HW2model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import HW2Dataset
import argparse
import numpy as np
from utils import confusion_matrix, plot_losses, plot_accuracies, plot_decision_boundary

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()

if __name__ == '__main__':
    train_dataset = HW2Dataset(path = 'HW2_training.csv')
    test_dataset = HW2Dataset(path = 'HW2_testing.csv')
    print('Train data has {} samples'.format(train_dataset.__len__()))
    print('Test data has {} samples'.format(test_dataset.__len__()))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    model = HW2model(num_features=train_dataset.num_features,num_teams=train_dataset.num_teams).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    for epoch in range(1,args.epoch+1):
        print(f'Epoch {epoch}/{args.epoch}')
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for features, labels in train_dataloader:
            features, labels = features.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_losses.append(train_loss/len(train_dataloader))
        train_accuracies.append(correct/total)
        print(f'Training loss: {train_loss/len(train_dataloader)}')
        print(f'Training accuracy: {correct/total}')
        
        # testing
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        for features, labels in test_dataloader:
            features, labels = features.to(args.device), labels.to(args.device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_losses.append(test_loss/len(test_dataloader))
        test_accuracies.append(correct/total)
        print(f'Testing loss: {test_loss/len(test_dataloader)}')
        print(f'Testing accuracy: {correct/total}')
    
    
    # plot the training and testing loss
    plot_losses(train_losses, test_losses)
    plot_accuracies(train_accuracies, test_accuracies)
    
    model.eval()
    # calculate the confusion matrix
    train_accuracy, train_confusion_matrix = confusion_matrix(train_dataset, model, args)
    print('Train accuracy: {}'.format(train_accuracy))
    test_accuracy, test_confusion_matrix = confusion_matrix(test_dataset, model, args)
    print('Test accuracy: {}'.format(test_accuracy))
    
    # plot the decision boundary
    plot_decision_boundary(model,args)