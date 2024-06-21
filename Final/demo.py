import torch
import torch.nn as nn
import os
#import your model here
from model import lightNN
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from utils import plot_confusion_matrix, test_accuracy

def show_results(imgs, labels, preds):
    num_imgs = len(imgs)
    fig, axs = plt.subplots(4, 4, figsize=(10, 10))
    correct = 0
    for i in range(num_imgs):
        row = i // 4
        col = i % 4
        axs[row, col].imshow(imgs[i].permute(1, 2, 0))
        label = 'child' if labels[i] == 1 else 'adult'
        pred = 'child' if preds[i] == 1 else 'adult'
        correct += (label == pred)
        axs[row, col].set_title('label: {}, pred: {}'.format(label, pred),color='r' if label != pred else 'b')
    plt.tight_layout()
    plt.savefig('results.png')
    print('accuracy: {} out of {}'.format(correct, num_imgs))

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trans = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                ])
    test_ds = ImageFolder(root=os.path.join(os.getcwd(),'./dataset/test'), transform=trans) #real world images
    test_dl = DataLoader(test_ds, batch_size=16,shuffle=True)
    model = lightNN().to(device)
    model.load_state_dict(torch.load('checkpoint/student_v1.pth')['model'],strict=False)
    model.eval()
    for img,label in test_dl:
        img = img.to(device)
        _,output = model(img)
        pred = output.argmax(dim=1, keepdim=True)
        show_results(img.cpu(), label.cpu(),pred.cpu())
        break
    
    plot_confusion_matrix(model, test_dl,num_classes=2)
    print(test_accuracy(model, test_dl))