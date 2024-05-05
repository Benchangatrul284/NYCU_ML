import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_losses(train_losses, test_losses,path='loss.png'):
    '''
    input train_losses: list of training losses
    input test_losses: list of testing losses
    '''
    plt.cla()
    plt.plot(range(1,len(train_losses)+1), train_losses, label='Training loss')
    plt.plot(range(1,len(test_losses)+1), test_losses, label='Testing loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path)
    
def plot_accuracies(train_accuracies, test_accuracies,path='accuracy.png'):
    '''
    input train_accuracies: list of training accuracies
    input test_accuracies: list of testing accuracies
    '''
    plt.cla()
    plt.plot(range(1,len(train_accuracies)+1), train_accuracies, label='Training accuracy')
    plt.plot(range(1,len(test_accuracies)+1), test_accuracies, label='Testing accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(path)
    
def confusion_matrix(dataset, model, args):
    '''
    input true: true label (N,)
    input pred: predicted label (N,)
    input num_classes: number of classes
    returns confusion matrix (num_classes, num_classes)
    '''
    correct = 0
    train_features = torch.tensor(dataset.features,dtype=torch.float).to(args.device)
    train_logits = model(train_features)
    train_predictions = torch.argmax(train_logits, dim=1).cpu().numpy()
    correct = (train_predictions == dataset.labels).sum()
    accuracy = correct/len(dataset)
    
    confusion_matrix = np.zeros((dataset.num_teams,dataset.num_teams))
    for i in range(len(dataset)):
        confusion_matrix[int(train_predictions[i]),int(dataset.labels[i])] += 1
    return accuracy,confusion_matrix

def plot_decision_boundary(model,args,path='decision_boundary.png'):
    # plot the decision boundary
    x1, x2 = np.meshgrid(np.linspace(0, 100, 100), np.linspace(0, 100, 100)) # create a 100x100 grid
    x = np.array([x1.ravel(), x2.ravel()]).T # flatten and transpose to get a 2D array
    x = torch.tensor(x,dtype=torch.float).to(args.device)
    logits = model(x)
    predictions = torch.argmax(logits, dim=1).cpu().numpy()
    plt.cla()
    plt.contourf(x1, x2, predictions.reshape(x1.shape), alpha=0.5)
    plt.xlabel('Offensive')
    plt.ylabel('Defensive')
    plt.title('Decision boundary')
    plt.savefig(path)

