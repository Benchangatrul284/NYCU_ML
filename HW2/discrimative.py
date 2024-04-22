import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--gif', type=str, default='plot')
parser.add_argument('--train_data', type=str, default='HW2_training.csv')
parser.add_argument('--test_data', type=str, default='HW2_testing.csv')
args = parser.parse_args()

def design_matrix(features):
    '''
    input: features (N,2)
    output: design matrix (N,3)
    '''
    return np.concatenate((np.ones((features.shape[0],1)), features), axis=1)

def grd_w(x,y,t):
    '''
    compute the gradient of w
    input x: design matrix (N,M)
    input y: predicted target (N,K)
    input t: true one hot target (N,K)
    '''
    def partial_w_j(x,y,t,j):
        '''
        compute the element of the gradient of w
        j is the index of the classes, compute the error of the j-th class
        returns a (M,1) vector
        '''
        w_k = 0
        for i in range(x.shape[0]):
            w_k += (y[i,j] - t[i,j])*x[i,:] # (M,)
            
        return w_k.reshape(-1,1)
    
    return np.concatenate([partial_w_j(x,y,t,j) for j in range(num_of_classes)], axis=0) # (MK,1)

def grd_grd(x,y,t):
    '''
    compute the Hassian matrix of w
    input x: design matrix (N,M)
    input y: predicted target (N,K)
    input t: true one hot target (N,K)
    returns a (MK,MK) matrix
    '''
    def partial_w_k_w_j(x,y,t,k,j):
        '''
        compute the block of the Hassian matrix of w
        k is the index of the classes, compute the error of the k-th class
        j is the index of the classes, compute the error of the j-th class
        returns a (M,M) matrix
        '''
        R = np.diag(y[:,k] * (int(k==j) - y[:,j])) #(N,N)
        w_k_w_j = x.T @ R @ x # (M,M)
        
        return w_k_w_j
    
    # j is the column index, k is the row index
    return np.block([[partial_w_k_w_j(x,y,t,k,j) for j in range(num_of_classes)] for k in range(num_of_classes)]) # (MK,MK)

def grd_grd(x,y,t):
    '''
    compute the Hassian matrix of w
    input x: design matrix (N,M)
    input y: predicted target (N,K)
    input t: true one hot target (N,K)
    returns a (MK,MK) matrix
    '''
    def partial_w_k_w_j(x,y,t,k,j):
        '''
        compute the block of the Hassian matrix of w
        k is the index of the classes, compute the error of the k-th class
        j is the index of the classes, compute the error of the j-th class
        returns a (M,M) matrix
        '''
        w_k_w_j = np.zeros((x.shape[1],x.shape[1]))
        for i in range(x.shape[0]):
            w_k_w_j += y[i,k]*(int(k==j) - y[i,j])*np.outer(x[i,:],x[i,:])
        
        return w_k_w_j
    
    # j is the column index, k is the row index
    return np.block([[partial_w_k_w_j(x,y,t,k,j) for j in range(num_of_classes)] for k in range(num_of_classes)]) # (MK,MK)
      
def accuracy(y,t):
    '''
    compute the accuracy
    input y: predicted target (N,K)
    input t: true one hot target (N,K)
    '''
    correct = 0
    for i in range(y.shape[0]):
        if np.argmax(y[i]) == np.argmax(t[i]):
            correct += 1
    return correct/y.shape[0]

def forward_pass(x,w):
    '''
    input x: design matrix (N,M)
    input w: weight matrix (M,K)
    '''
    predictions = np.matmul(x,w) # (N,K)
    predictions = predictions - np.max(predictions, axis=1, keepdims=True)
    predictions = np.exp(predictions) / np.sum(np.exp(predictions), axis=1).reshape(-1,1)
    return predictions

def plot_decision_boundary(data,w,path,epoch):
    '''
    input: w: weight matrix (M,K)
    input: train_features: (N,M)
    input: train_targets: (N,K)
    '''
    plt.clf()
    # plot the decision boundary
    x1, x2 = np.meshgrid(np.linspace(0, 100, 100), np.linspace(0, 100, 100)) # create a 100x100 grid
    x = np.array([x1.ravel(), x2.ravel()]).T # flatten and transpose to get a 2D array
    x = design_matrix(x) # (N,M)
    predictions = forward_pass(x,w) # (N,K)
    print(predictions)
    predictions = np.argmax(predictions, axis=1)
    plt.contourf(x1,x2, predictions.reshape(100,100))
    plt.scatter(data['Offensive'], data['Defensive'], c=data['Team'])
    if epoch is not None:
        plt.title(f'decision boundary at epoch {epoch}')
    plt.savefig(path)

def plot_accuracy(train_accuracy, test_accuracy):
    plt.figure()
    plt.plot(train_accuracy, label='Train Accuracy')
    plt.plot(test_accuracy, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy.png')  
    
def create_gif(path):
    import imageio
    import os
    import natsort
    images = []
    full_path = os.path.join(os.getcwd(), path)
    print(full_path)
    for filename in natsort.natsorted(os.listdir(full_path)):
        images.append(imageio.v2.imread(os.path.join(full_path, filename)))
    imageio.v2.mimsave('decision_boundary.gif', images, duration=1)


def reshape_weight(w):
    '''
    reshape the weight matrix to (M,K)
    '''
    w_new = np.zeros((num_features,num_of_classes))
    for i in range(num_of_classes* num_features):
        w_new[i % num_features, i // num_features] = w[i]
    return w_new

def cross_entropy(predictions, targets):
    '''
    input: predictions (N,K)
    input: targets (N,K)
    '''
    return -np.mean(targets*np.log(predictions))

def plot_loss(train_losses, test_losses):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')

def confusion_matrix(true, pred, num_classes):
    '''
    input true: true label (N,)
    input pred: predicted label (N,)
    input num_classes: number of classes
    returns confusion matrix (num_classes, num_classes)
    '''
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(true)):
        confusion_matrix[int(true[i]), int(pred[i])] += 1
    
    
    return confusion_matrix
   
def plot_confusion_matrix(confusion_matrix, path,train=True):
    plt.figure()
    plt.imshow(confusion_matrix)
    plt.colorbar()
    plt.xlabel('Prediction')
    plt.ylabel('True')
    if train:
        plt.title('Train Confusion matrix')
    else:
        plt.title('Test Confusion matrix')
    plt.savefig(path)
    
if __name__ == '__main__':
    import os
    import shutil
    if os.path.exists('plot'):
        shutil.rmtree('plot')

    os.makedirs('plot')
    
    # compute the number of players of each team
    train_data = pd.read_csv(args.train_data)
    test_data = pd.read_csv(args.test_data)
    num_of_classes = train_data['Team'].nunique()
    num_features = len(train_data.columns)
    
    train_features = design_matrix(train_data[['Offensive', 'Defensive']].values) #(N,3)
    train_targets = train_data['Team'].values.astype(int)
    
    test_features = design_matrix(test_data[['Offensive', 'Defensive']].values) #(N,3)
    test_targets = test_data['Team'].values.astype(int)
    
    # create a one-hot encoding of the target
    num_teams = train_data['Team'].nunique()
    train_targets_onehot = np.zeros((train_targets.shape[0], num_teams))
    train_targets_onehot[np.arange(train_targets.shape[0]), train_targets] = 1
    
    test_targets_onehot = np.zeros((test_targets.shape[0], num_teams))
    test_targets_onehot[np.arange(test_targets.shape[0]), test_targets] = 1
    
    w_old = np.zeros((num_features, num_of_classes))
    print('Initial w:', w_old)
    w_new = w_old
    
    train_accuracies = []
    test_accuracies = []
    old_train_accuracy = accuracy(np.matmul(train_features,w_old), train_targets_onehot)
    print('Initial train accuracy:', old_train_accuracy)
    train_accuracies.append(old_train_accuracy)
    old_test_accuracy = accuracy(np.matmul(test_features,w_old), test_targets_onehot)
    test_accuracies.append(old_test_accuracy)
    print('Initial test Accuracy:', old_test_accuracy)
    
    train_losses = []
    test_losses = []
    train_loss = cross_entropy(forward_pass(train_features, w_old), train_targets_onehot)
    test_loss = cross_entropy(forward_pass(test_features, w_old), test_targets_onehot)
    print('Initial train loss:', train_loss)
    print('Initial test loss:', test_loss)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    
    for i in range(args.epochs):
        print('Epoch:', i)
        if i % 1 == 0:
            plot_decision_boundary(data=train_data, w=w_old, path=f'plot/train_decision_boundary_epoch_{i}.png',epoch=i)
        # do the forward pass
        y = forward_pass(train_features, w_old) # (N,K)
        # compute the gradient of w
        partial_w = grd_w(train_features, y, train_targets_onehot) # (MK,1)
        Hassian_w = grd_grd(train_features, y, train_targets_onehot) # (MK,MK)
        # reshape w to (MK,1)
        w_old = w_old.T.reshape(-1,1)
        w_new = w_old - np.linalg.pinv(Hassian_w) @ partial_w
        w_new = w_old - 1e-6*partial_w
        # reshape w back to (M,K)
        w_old = reshape_weight(w_old)
        w_new = reshape_weight(w_new)
        
        # compute the new y to compute the accuracy and loss
        new_train_accuracy = accuracy(np.matmul(train_features,w_new), train_targets_onehot)
        print('Train Accuracy:', new_train_accuracy)
        train_loss = cross_entropy(forward_pass(train_features, w_new), train_targets_onehot)
        print('Train Loss:', train_loss)
        new_test_accuracy = accuracy(np.matmul(test_features,w_new), test_targets_onehot)
        print('Test Accuracy:', new_test_accuracy)
        test_loss = cross_entropy(forward_pass(test_features, w_new), test_targets_onehot)
        print('Test Loss:', test_loss)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(new_train_accuracy)
        test_accuracies.append(new_test_accuracy)
        
        old_test_accuracy = new_test_accuracy
        old_train_accuracy = new_train_accuracy
        w_old = w_new
        w = w_old
        
        
    plot_decision_boundary(data=test_data, w=w, path='test_decision_boundary.png',epoch=None)
    
    create_gif(args.gif)
    plot_loss(train_losses, test_losses)
    plot_accuracy(train_accuracies, test_accuracies)
    train_predictions = forward_pass(train_features, w)
    test_predictions = forward_pass(test_features, w)
    train_confusion_matrix = confusion_matrix(train_data['Team'], np.argmax(train_predictions, axis=1), num_of_classes)
    test_confusion_matrix = confusion_matrix(test_data['Team'], np.argmax(test_predictions, axis=1), num_of_classes)
    print(train_confusion_matrix)
    print(test_confusion_matrix)
    plot_confusion_matrix(train_confusion_matrix, 'train_confusion_matrix.png')
    plot_confusion_matrix(test_confusion_matrix, 'test_confusion_matrix.png',train=False)