import argparse
import matplotlib.pyplot as plt
from train import train_test_accuracy
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epoch', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--device', type=str, default='cpu')
args = parser.parse_args()


if __name__ == '__main__':
    '''
    Problem 1: construct a DNN with one layer, plot the training and testing accuracy versus number of neurons
    '''
    train_accuracies = []
    test_accuracies = []
    num_neurons = [5,10,20,50,75,100]
    model = 'slp'
    for neurons in num_neurons:
        train_acc, test_acc = train_test_accuracy(lr=args.lr, epochs=args.epoch, batch_size=args.batch_size,num_neurons=neurons,model='slp',device=args.device)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
    # plot the training and testing accuracy
    plt.plot(num_neurons, train_accuracies, label='Training accuracy')
    plt.plot(num_neurons, test_accuracies, label='Testing accuracy')
    plt.xlabel('Number of neurons')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('problem1.png')
    
    '''
    Problem 2: construct a DNN with two layers, the number of neurons in each layer is 100
    '''
    train_accuracies = []
    test_accuracies = []
    percents = np.arange(0.1,1.1,0.1)
    for percent in percents:
        train_acc, test_acc = train_test_accuracy(lr=args.lr, epochs=args.epoch, batch_size=args.batch_size, num_neurons=100,model='mlp',device=args.device,percent=percent)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
    # plot the training and testing accuracy
    plt.cla()
    plt.plot(percents, train_accuracies, label='Training accuracy')
    plt.plot(percents, test_accuracies, label='Testing accuracy')
    plt.xlabel('Percentage of training data')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('problem2.png')
    
    '''
    Problem 3: construct a DNN with hidden layers, the number of neurons in each layer is 100
    '''
    train_accuracies = []
    test_accuracies = []
    hidden_layers = np.arange(1,6)
    for num_layers in hidden_layers:
        train_acc, test_acc = train_test_accuracy(lr=args.lr, epochs=args.epoch, batch_size=args.batch_size,num_neurons=100,model='dnn',device=args.device,num_layers=num_layers)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
    
    # plot the training and testing accuracy
    plt.cla()
    plt.plot(hidden_layers, train_accuracies, label='Training accuracy')
    plt.plot(hidden_layers, test_accuracies, label='Testing accuracy')
    plt.xlabel('Number of hidden layers')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('problem3.png')
    