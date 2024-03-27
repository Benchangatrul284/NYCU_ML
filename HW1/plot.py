import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

if __name__ == '__main__':
    
    if os.path.exists('plot'):
        shutil.rmtree('plot')
    os.mkdir('plot')
    
    M = [5,10,15,20,25,30]
    
    # plot fitting curve
    for m in M:
        plt.cla()
        df = pd.read_csv('fitting_curve_{}.csv'.format(m))
        x = np.array(df['danceability'])
        y = np.array(df['Predictions'])
        t = np.array(df['Test_target'])
        plt.scatter(x,t, label='target', color='blue')
        plt.scatter(x,y, label='prediction', color='red')
        plt.title('Fitting Curve for M = {} without regularlization'.format(m))
        plt.legend()
        plt.xlabel('danceability')
        plt.ylabel('song popularity')
        plt.savefig(os.path.join('plot2','fitting_curve_{}_without_regularization.png'.format(m)))
        
        plt.cla()
        df = pd.read_csv('fitting_curve_{}.csv'.format(m))
        x = np.array(df['danceability'])
        y = np.array(df['Predictions Regularization'])
        t = np.array(df['Test_target'])
        plt.scatter(x,t, label='target', color='blue')
        plt.scatter(x,y, label='prediction', color='red')
        plt.title('Fitting Curve for M = {} with regularlization'.format(m))
        plt.xlabel('danceability')
        plt.ylabel('song popularity')
        plt.legend()
        plt.savefig(os.path.join('plot2','fitting_curve_{}_with_regularization.png'.format(m)))
        
        
    # plot validation error
    df = pd.read_csv('cross_validation.csv')
    validation_error = np.array(df['Validation Error'])
    plt.cla()
    plt.plot(range(1,31), validation_error, label='validation error')
    plt.title('Validation Error')
    plt.legend()
    plt.xlabel('M')
    plt.ylabel('MSE Error')
    plt.savefig(os.path.join('plot2','validation_error.png'))
    
    # plot error and accuracy
    df = pd.read_csv('result.csv')
    train_error = np.array(df['Train Error'])
    test_error = np.array(df['Test Error'])
    train_accuracy = np.array(df['Train Accuracy'])
    test_accuracy = np.array(df['Test Accuracy'])
    train_error_reg = np.array(df['Train Error Regularization'])
    test_error_reg = np.array(df['Test Error Regularization'])
    train_accuracy_reg = np.array(df['Train Accuracy Regularization'])
    test_accuracy_reg = np.array(df['Test Accuracy Regularization'])
    
    fig, axs = plt.subplots(1,2,figsize=(10,5))

    axs[0].plot([5,10,15,20,25,30], train_error, label='train error')
    axs[0].plot([5,10,15,20,25,30], test_error, label='test error')
    axs[0].set_title('Train and Test Error')
    axs[0].legend()
    axs[0].set_xlabel('M')
    axs[0].set_ylabel('MSE Error')
    
    axs[1].plot([5,10,15,20,25,30], train_accuracy, label='train accuracy')
    axs[1].plot([5,10,15,20,25,30], test_accuracy, label='test accuracy')
    axs[1].set_title('Train and Test Accuracy')
    axs[1].legend()
    axs[1].set_xlabel('M')
    axs[1].set_ylabel('Accuracy')
    plt.tight_layout()
    
    plt.savefig(os.path.join('plot2','error and accuracy without regularization.png'))
    
    fig, axs = plt.subplots(1,2,figsize=(10,5))

    axs[0].plot([5,10,15,20,25,30], train_error_reg, label='train error')
    axs[0].plot([5,10,15,20,25,30], test_error_reg, label='test error')
    axs[0].set_title('Train and Test Error')
    axs[0].set_xlabel('M')
    axs[0].set_ylabel('MSE Error')
    axs[0].legend()

    axs[1].plot([5,10,15,20,25,30], train_accuracy_reg, label='train accuracy')
    axs[1].plot([5,10,15,20,25,30], test_accuracy_reg, label='test accuracy')
    axs[1].set_title('Train and Test Accuracy')
    axs[1].set_xlabel('M')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join('plot2','error and accuracy with regularization.png'))
    
    # plot demo
    
    # for m in M:
    #     plt.cla()
    #     plt.clf()
    #     plt.figure(figsize=(5,5))
    #     df = pd.read_csv('demo_predictions_{}.csv'.format(m))
    #     x = np.array(df['danceability'])
    #     y = np.array(df['Predictions'])
    #     t = np.array(df['Test_target'])
    #     plt.scatter(x,t, label='test data', color='blue')
    #     plt.scatter(x,y, label='prediction', color='red')
    #     plt.title('Demo Fitting Curve for M = {} without regularlization'.format(m))
    #     plt.legend()
    #     plt.xlabel('danceability')
    #     plt.ylabel('song popularity')
    #     plt.savefig(os.path.join('plot2','demo_fitting_curve_{}_without_regularization.png'.format(m)))
        
    #     plt.cla()
    #     df = pd.read_csv('fitting_curve_{}.csv'.format(m))
    #     x = np.array(df['danceability'])
    #     y = np.array(df['Predictions Regularization'])
    #     t = np.array(df['Test_target'])
    #     plt.scatter(x,t, label='test data', color='blue')
    #     plt.scatter(x,y, label='prediction', color='red')
    #     plt.title('Demo Fitting Curve for M = {} with regularlization'.format(m))
    #     plt.xlabel('danceability')
    #     plt.ylabel('song popularity')
    #     plt.legend()
    #     plt.savefig(os.path.join('plot2','demo_fitting_curve_{}_with_regularization.png'.format(m)))
        
    # df = pd.read_csv('demo_result.csv')
    # x = np.array(df['M'])
    # y = np.array(df['Demo Accuracy'])
    # y_r = np.array(df[' Demo Accuracy Regularization'])
    # plt.cla()
    # plt.plot(x,y, label='demo accuracy')
    # plt.plot(x,y_r, label='demo accuracy regularization')
    # plt.title('Demo Accuracy')
    # plt.xlabel('M')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.savefig(os.path.join('plot2','demo_accuracy.png'))
    
    
    
    
    
    
    