import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type=str, default='HW2_training.csv')
parser.add_argument('--test_data', type=str, default='HW2_testing.csv')
args = parser.parse_args()

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

if __name__ == '__main__':
    # Load data
    train_data = pd.read_csv(args.train_data)
    test_data = pd.read_csv(args.test_data)
    
    # plot the distribution of train data
    plt.figure()
    plt.scatter(train_data['Offensive'], train_data['Defensive'], c=train_data['Team'])
    plt.xlabel('Offensive')
    plt.ylabel('Defensive')
    plt.title('Train data')
    plt.savefig('train_data.png')

    # plot the distribution of test data
    plt.figure()
    plt.scatter(test_data['Offensive'], test_data['Defensive'], c=test_data['Team'])
    plt.xlabel('Offensive')
    plt.ylabel('Defensive')
    plt.title('Test data')
    plt.savefig('test_data.png')

    # compute the number of players of each team
    num_teams = train_data['Team'].nunique()
    num_features = len(train_data.columns) - 1
    c = [0] * num_teams
    for i in train_data['Team']:
        c[int(i)] += 1
    
    # compute the average of each feature of each team
    team_feature = np.zeros((num_teams, num_features))
    for i in range(num_teams):
        team_feature[i,0] = train_data[train_data['Team'] == i]['Offensive'].mean()
        team_feature[i,1] = train_data[train_data['Team'] == i]['Defensive'].mean()

    # compute the covariance matrix
    cov = np.zeros((num_features, num_features))
    for i in range(len(train_data)):
        team = int(train_data['Team'][i])
        x = np.array([train_data['Offensive'][i], train_data['Defensive'][i]])
        x = x - team_feature[team]
        x = x.reshape(-1,1)
        cov += np.dot(x, x.T)
    cov /= len(train_data)
    # print(cov)

    # for each class, compute wk and wko (different classes is seperated by different rows)
    w = np.zeros((num_teams, num_features))
    w0 = np.zeros(num_teams)
    for i in range(num_teams):
        w[i] = np.dot(np.linalg.inv(cov), team_feature[i])
        w0[i] = -0.5 * np.dot(np.dot(team_feature[i], np.linalg.inv(cov)), team_feature[i]) + np.log(c[i] / len(train_data))
    
    # use train data to compute the accuracy
    train_predictions = np.zeros(len(train_data))
    train_logits = np.zeros((len(train_data), num_teams))
    correct = 0
    for i in range(len(train_data)):
        target = int(train_data['Team'][i])
        x = np.array([train_data['Offensive'][i], train_data['Defensive'][i]])
        for j in range(num_teams):
            train_logits[i,j] = np.dot(w[j], x) + w0[j]
        # apply softmax
        predict = np.exp(train_logits[i]) / np.sum(np.exp(train_logits[i]))
        train_predictions[i] = np.argmax(predict)

        if train_predictions[i] == target:
            correct += 1
        
    print(correct / len(train_data))
    
    
     # calculate the confusion matrix of training data
    train_confusion_matrix = confusion_matrix(train_data['Team'], train_predictions, num_teams)
    print(train_confusion_matrix)
    # predict the test data
    test_predictions = np.zeros(len(test_data))
    test_logits = np.zeros((len(test_data), num_teams))
    correct = 0
    
    for i in range(len(test_data)):
        target = int(test_data['Team'][i])
        x = np.array([test_data['Offensive'][i], test_data['Defensive'][i]])
        for j in range(num_teams):
            test_logits[i,j] = np.dot(w[j], x) + w0[j]
        # apply softmax
        predict = np.exp(test_logits[i]) / np.sum(np.exp(test_logits[i]))
        test_predictions[i] = np.argmax(predict)

        if test_predictions[i] == target:
            correct += 1
    
    print(correct / len(test_data))
    
    
    # calculate the confusion matrix of testing data
    test_confusion_matrix = confusion_matrix(test_data['Team'], test_predictions, num_teams)
    print(test_confusion_matrix)
    # plot the confusion matrix
    plt.figure()
    plt.imshow(train_confusion_matrix)
    plt.colorbar()
    plt.xlabel('Prediction')
    plt.ylabel('True')
    plt.title('Train confusion matrix')
    plt.savefig('train_confusion_matrix.png')
    
    # plot the confusion matrix
    plt.figure()
    plt.imshow(test_confusion_matrix)
    plt.colorbar()
    plt.xlabel('Prediction')
    plt.ylabel('True')
    plt.title('Test confusion matrix')
    plt.savefig('test_confusion_matrix.png')
    
    # plot the decision boundary
    x1, x2 = np.meshgrid(np.linspace(0, 100, 100), np.linspace(0, 100, 100)) # create a 100x100 grid
    x = np.array([x1.ravel(), x2.ravel()]).T # flatten and transpose to get a 2D array
    predictions = np.zeros(x.shape[0])
    logit = np.zeros((x.shape[0], num_teams))
    for i in range(x.shape[0]):
        for j in range(num_teams):
            logit[i,j] = np.dot(w[j], x[i]) + w0[j]
        # get the class with the highest probability
        predictions[i] = np.argmax(logit[i])
    
    plt.figure()
    plt.contourf(x1, x2, predictions.reshape(x1.shape), alpha=0.5)
    # plt.scatter(train_data['Offensive'], train_data['Defensive'], c=train_data['Team'])
    plt.xlabel('Offensive')
    plt.ylabel('Defensive')
    plt.title('Decision boundary')
    plt.savefig('decision_boundary.png')
    