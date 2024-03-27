import pandas as pd
import numpy as np
import os
import shutil
import argparse


def normalize(features: np.array):
    """
    input: features: np.array of shape (n_samples, n_features)
    output: normalized_features: np.array of shape (n_samples, n_features)
        mean: float
        std: float
    """
    return (features - features.mean(axis=0)) / features.std(axis=0), features.mean(axis=0), features.std(axis=0)

def design_matrix_single_feature(features: np.array, M, k=2):
    '''
    input: given a features matrix of shape (n_samples, n_features)
    only get the the third feature
    output is a matrix of shape (n_samples, M)
    '''
    s = 0.1
    N = features.shape[0]
    phi = np.zeros((N, M))
    features = features[:,k]
    
    for n in range(N):
        for j in range(M):
            mu_j = 3* (-M+1 +2*(j-1)*(M-1)/(M-2+1e-9)) / M
            phi[n, j] = 1 if j == 0 else sigmoid((features[n]-mu_j)/s)
                  
    return phi

def design_matrix(features: np.array, M):
    '''
    input: given a features matrix of shape (n_samples, n_features)
    a feature is transformed into a new feature vector of shape (M, 1) by applying the basis function
    output is a matrix of shape (n_samples, n_features* M)
    '''
    s = 0.1
    N = features.shape[0]
    K = features.shape[1]
    
    # create a matrix of shape (N, K, M)
    phi = np.zeros((N, K*M))
    
    phi = np.concatenate([design_matrix_single_feature(features, M, k) for k in range(K)], axis=1)

    # for n in range(N):
    #     for k in range(K*M):
    #         f = k // M # the index of the feature
    #         j = k % M # the index of the basis function
    #         mu_j = 3* (-M+1 +2*(j-1)*(M-1)/(M-2)) / M
    #         phi[n,k] = 1 if j == 0 else sigmoid( (features[n,f]-mu_j) /s)
    
    print(phi.shape)
    return phi
    

def plot_fitting_curve(features: np.array,k: int,predictions : np.array, targets : np.array,m: int):
    '''
    input: features: np.array of shape (n_samples, n_features)
        k: int, the index of the feature to plot (=2 for danceability)
        targets: np.array of shape (n_samples, 1)
        weights: np.array of shape (M, 1)
        M: int, the number of basis functions
    '''
    
    import matplotlib.pyplot as plt
    plt.cla()
    feature = features[:,k]
    plt.scatter(feature, targets, label='true data')
    plt.scatter(feature, predictions, label='Predicted')
    # add the plot title
    plt.title('Fitting curve for M =  {} at test data'.format(m))
    plt.legend()
    plt.savefig(os.path.join('plot','M_{}.png'.format(m)))
    

def generate(train_features: np.array, test_features: np.array, train_targets: np.array, M: int, lambda_: float = 0.1):
    '''
    input:  train_features: np.array of shape (n_samples, n_features)
            test_features: np.array of shape (n_samples, n_features)
            train_targets: np.array of shape (n_samples, 1)
            M: int, the number of basis functions
            lambda_: float, the regularization parameter
    output: train_phi: np.array of shape (n_samples, K*M)
            test_phi: np.array of shape (n_samples, K*M)
            weights: np.array of shape (K*M, 1)
            train_predictions: np.array of shape (n_samples, 1)
            test_predictions: np.array of shape (n_samples, 1)
    '''
    
    train_phi = design_matrix(train_features, M=M)
    test_phi = design_matrix(test_features, M=M)
    
    print('train_phi has shape {}'.format(train_phi.shape)) # N, K*M
    print('test_phi has shape {}'.format(test_phi.shape)) # N, K*M
    if lambda_ == 0:
        weights = np.linalg.pinv(train_phi) @ train_targets
    else:
        weights = np.linalg.inv(train_phi.T @ train_phi + lambda_ * np.eye(M*train_features.shape[1])) @ train_phi.T @ train_targets
    
    print('weights has shape {}'.format(weights.shape)) # K*M, 1
    
    train_predictions = train_phi @ weights
    test_predictions = test_phi @ weights
    
    return train_phi, test_phi, weights, train_predictions, test_predictions

def mse(predictions: np.array, targets: np.array):
    '''
    input: predictions: np.array of shape (n_samples, 1)
           targets: np.array of shape (n_samples, 1)
    output: mse: float
    '''
    return np.mean((predictions - targets)**2)

def accuracy(predictions: np.array, targets: np.array):
    # modify the test_target is that the 0 becomes 1
    targets = [targets[i] if targets[i] != 0 else 1 for i in range(len(targets))]
    accuracy = 1 - np.mean(np.abs((predictions - targets)/targets))
    return accuracy

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def plot_error(train_error: np.array, test_error: np.array):
    import matplotlib.pyplot as plt
    plt.cla()
    plt.plot(M, train_error, label='train error')
    plt.plot(M, test_error, label='test error')
    plt.legend()
    plt.title('Train and Test Error')
    plt.savefig('error.png')

def plot_accuracy(train_accuracy: np.array, test_accuracy: np.array):
    import matplotlib.pyplot as plt
    plt.cla()
    plt.plot(M, train_accuracy, label='train accuracy')
    plt.plot(M, test_accuracy, label='test accuracy')
    plt.legend()
    plt.title('Train and Test Accuracy')
    plt.savefig('accuracy.png')

def plot_weights(weights: np.array,m: int):
    import matplotlib.pyplot as plt
    plt.cla()
    '''
    the weights are of shape (K*M, 1)
    for each feature, plot the weights of the basis functions
    '''
    sum = []
    for k in range(11):
        sum.append(np.abs(weights[k*m:(k+1)*m]).sum())
    plt.bar(range(11), sum)
    plt.title('Sum of weights for each feature at M = {}'.format(m))
    plt.savefig(f'weights at {m}.png')
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda_', type=float, default=0.1)
    args = parser.parse_args()
    df = pd.read_csv('HW1.csv')
    targets = np.array(df['song_popularity'])
    features = np.array(df.drop('song_popularity', axis=1))

    # split into testing and training data
    train_features = features[:10000]
    test_features = features[10000:]
    train_targets = targets[:10000]
    test_targets = targets[10000:]
    
    print('train_features has {} samples and {} features'.format(*train_features.shape))
    print('train_targets has {} samples'.format(*train_targets.shape))
    print('test_features has {} samples and {} features'.format(*test_features.shape))
    print('test_targets has {} samples'.format(*test_targets.shape))
    
    # normalize the features
    normalized_train_features, mean, std = normalize(train_features)
    normalized_test_features = (test_features - mean) / std
    
    # ============================================ #
    if os.path.exists('plot'):
        shutil.rmtree('plot')
    os.mkdir('plot')
    
    train_error = []
    test_error = []
    train_accuracy = []
    test_accuracy = []
    M = [5,10,15,20,25,30]
    for m in M:
        print('M = {} .......'.format(m))
        train_phi, test_phi, weights, train_predictions, test_predictions = generate(normalized_train_features, normalized_test_features,train_targets,M=m,lambda_= args.lambda_)
        
        # compute the error
        plot_fitting_curve(normalized_test_features, 2, test_predictions, test_targets, m=4)
        plot_weights(weights,m)
        train_error.append(mse(train_predictions, train_targets))
        test_error.append(mse(test_predictions, test_targets))
        train_accuracy.append(accuracy(train_predictions, train_targets))
        test_accuracy.append(accuracy(test_predictions, test_targets))
        
    plot_error(train_error, test_error)
    plot_accuracy(train_accuracy, test_accuracy)