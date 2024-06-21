import argparse
import os

from model import DeepNN, MobileNetV3_small
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from utils import plot_accuracy_curve, plot_loss_curve, plot_lr_curve
from torchvision.transforms import v2
from torchvision.transforms.autoaugment import AutoAugmentPolicy
from effnet import EfficientNetV2

parser = argparse.ArgumentParser()
parser.add_argument('--resume', default='', help='path to latest checkpoint')
parser.add_argument('--export', default='teacher.pth', help='path to save checkpoint')
parser.add_argument('--epoch', default=100, help='number of epochs to train')
parser.add_argument('--batch_size', default=16, help='batch size')
parser.add_argument('--lr', default=1e-5, help='learning rate')
args = parser.parse_args()

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def adjust_learning_rate(epoch, T_max=args.epoch, eta_min=0, lr_init=args.lr):
    lr = eta_min + (lr_init - eta_min) * (1 + math.cos(math.pi * epoch / T_max)) / 2
    if epoch >= T_max:
        lr = eta_min
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train():
    history = []
    train_loss = 0
    best_accuracy = 0
    start_epoch = 1
    #loading pretrained models
    if args.resume:
        if os.path.isfile(args.resume):
            print("===> loading models '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model'])
            start_epoch = checkpoint['epoch']
            best_accuracy = checkpoint['best_accuracy']
            history = checkpoint['history']
            print("checkpoint loaded: epoch = {}, accuracy = {}".format(start_epoch, best_accuracy))
        else:
            print("===> no models found at '{}'".format(args.resume))

   
    for epoch in range(start_epoch,args.epoch + 1):
        adjust_learning_rate(epoch)
        result = {'train_loss': [], 'valid_loss': [], 'lrs': [], 'accuracy': []}
        print('Epoch: {}'.format(epoch))
        print('learning rate: {:.6f}'.format(optimizer.param_groups[0]['lr']))
        model.train()
        for (img,label) in tqdm(train_dl):
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            _,output = model(img)
            # output = model(img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_dl) # average loss per batch
        
        model.eval()
        with torch.no_grad():
            # compute validation loss and validation accuracy
            valid_loss = 0
            correct = 0
            for (img,label) in tqdm(valid_dl):
                img = img.to(device)
                label = label.to(device)
                _,output = model(img)
                # output = model(img)
                loss = criterion(output, label)
                valid_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()
            val_accuracy = correct/len(val_ds)
            
            valid_loss = valid_loss / len(valid_dl) # average loss per batch
            
            # compute test accuracy
            correct = 0
            for (img,label) in tqdm(test_dl):
                img = img.to(device)
                label = label.to(device)
                _,output = model(img)
                # output = model(img)
                pred = output.argmax(dim=1, keepdim=True) #returns the index of the maximum value
                correct += pred.eq(label.view_as(pred)).sum().item()
            
        accuracy = correct/len(test_ds)
        result['train_loss'].append(train_loss)
        result['valid_loss'].append(valid_loss)
        result['accuracy'].append(accuracy)
        result['lrs'].append(optimizer.param_groups[0]['lr'])
        print('Train Loss: {:.4f}'.format(train_loss))
        print('Val Loss: {:.4f}'.format(valid_loss))
        print('Val Accuracy: {:.4f}'.format(val_accuracy))
        print('Test Accuracy: {:.4f}'.format(accuracy))
        
        history.append(result)

        if accuracy > best_accuracy:
            model_folder = "checkpoint"
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
            best_accuracy = accuracy
            model_out_path = os.path.join(model_folder, args.export)
            state = {"epoch": epoch,
                    "model": model.state_dict(),
                    "best_accuracy": best_accuracy,
                    "history": history}
            torch.save(state, model_out_path)
            print("===> Checkpoint saved to {}".format(model_out_path))

        plot_loss_curve(history)
        plot_accuracy_curve(history)
        plot_lr_curve(history)



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ===============================transformation==================================
    trans = v2.Compose([
                        v2.Resize((185,160)),
                        v2.ToImage(), 
                        v2.ToDtype(torch.float32, scale=True),
                        ])
    trans1 = v2.Compose([
                        v2.Resize((185,160)),
                        # blur
                        v2.GaussianBlur(kernel_size=3),
                        # horizontal flip
                        v2.RandomHorizontalFlip(p=0.5),
                        v2.ToImage(), 
                        v2.ToDtype(torch.float32, scale=True),
                        ])
    trans2 = v2.Compose([
                        v2.Resize((185,160)),
                        v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                        v2.ToImage(), 
                        v2.ToDtype(torch.float32, scale=True),
                        ])

    trans3 = v2.Compose([
                        v2.Resize((185,160)),
                        v2.AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
                        v2.ToImage(), 
                        v2.ToDtype(torch.float32, scale=True),
                        ])
    
    trans4 = v2.Compose([
                        v2.Resize((185,160)),
                        v2.AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
                        v2.ToImage(), 
                        v2.ToDtype(torch.float32, scale=True),
                        ])

    trans5 = v2.Compose([
                        v2.Resize((185,160)),
                        v2.AutoAugment(policy=AutoAugmentPolicy.SVHN),
                        v2.ToImage(), 
                        v2.ToDtype(torch.float32, scale=True),
                        ])
    
    trans6 = v2.Compose([
                        v2.Resize((185,160)),
                        v2.RandAugment(num_ops=2),
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                        ])
    trans7 = v2.Compose([
                        v2.Resize((185,160)),
                        v2.RandAugment(num_ops=2),
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                        ])
    trans8 = v2.Compose([
                        v2.Resize((185,160)),
                        v2.RandAugment(num_ops=2),
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                        ])
    trans9 = v2.Compose([
                        v2.Resize((185,160)),
                        v2.RandAugment(num_ops=2),
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                        ])
    trans10 = v2.Compose([
                        v2.Resize((185,160)),
                        v2.RandAugment(num_ops=2),
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                        ])

    
    transformation = [trans1,trans2,trans3,trans4,trans5,trans6,trans7,trans8,trans9,trans10]
    
    # ===============================dataset==================================
    full_ds = ImageFolder(root=os.path.join(os.getcwd(),'./dataset/train'), transform=trans) #train and valid
    
    for trans in transformation:
        full_ds1 = ImageFolder(root=os.path.join(os.getcwd(),'./dataset/train'), transform=trans) #train and valid
        full_ds = torch.utils.data.ConcatDataset([full_ds1,full_ds])
    
    print(f'total training dataset: {len(full_ds)}')
    test_ds = ImageFolder(root=os.path.join(os.getcwd(),'./dataset/test'), transform=trans1) #test image
    
    #split train_ds into train and val
    train_size = int(len(full_ds)*0.95)
    val_size = len(full_ds)-train_size
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size])
    
    # ================================================================
    # dataloader
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,num_workers=12, shuffle=True,drop_last=True)
    valid_dl = DataLoader(val_ds, batch_size=8, shuffle=True,num_workers=12,drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=8, shuffle=True,num_workers=12,drop_last=True)
    # model
    model = DeepNN().to(device)
    # model = effnetv2_s(number_of_classes=2).to(device)
    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train()
    
    