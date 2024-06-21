import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# plot loss curve
def plot_loss_curve(history):
    plt.cla()
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x.get('valid_loss') for x in history]
    plt.plot(train_losses, '-bx',label='train')
    plt.plot(val_losses, '-rx',label='valid')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. No. of epochs')
    plt.legend()
    plt.savefig('loss_curve.png')
# plot accuracy curve
def plot_accuracy_curve(history):
    plt.cla()
    psnrs = [x.get('accuracy') for x in history]
    plt.plot(psnrs, '-bx')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('accuracy vs. No. of epochs');
    plt.savefig('accuracy.png')
# plot lr curve
def plot_lr_curve(history):
    plt.cla()
    lrs = [x.get('lrs') for x in history]
    plt.plot(lrs, '-bx')
    plt.xlabel('epoch')
    plt.ylabel('lr')
    plt.title('lr vs. No. of epochs');
    plt.savefig('lr_curve.png')

def test_accuracy(model, test_dl):
    correct = 0
    total = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        for (img, label) in test_dl:
            img = img.to(device)
            label = label.to(device)
            _, output = model(img)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
            total += label.shape[0]
    return correct / total
def plot_confusion_matrix(model, test_dl, num_classes=2):
    plt.cla()
    plt.figure()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    confusion_matrix = torch.zeros(num_classes, num_classes, device=device)
    for (img, label) in test_dl:
        img = img.to(device)
        label = label.to(device)
        _, output = model(img)
        pred = output.argmax(dim=1, keepdim=True)
        for t, p in zip(label.view(-1), pred.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
    # Move the confusion matrix to CPU for plotting
    confusion_matrix = confusion_matrix.cpu()
    plt.imshow(confusion_matrix.numpy(), interpolation='nearest', cmap=plt.cm.Blues)
    plt.xlabel('Prediction')
    plt.ylabel('True')
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.colorbar()
    plt.title('Test confusion matrix')
    plt.savefig('test_confusion_matrix.png')
        
        
def loss_fn_kd(outputs, labels, teacher_outputs, alpha=0.5,T=2):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    """
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss

