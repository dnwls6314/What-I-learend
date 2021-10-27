
import torch
import torch.nn as nn
import dataset
from model import LeNet5, CustomMLP, Regularized_LeNet5
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import time


def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """
    trn_loss_sum = 0; acc_sum = 0
    
    for idx, (imgs, labels) in enumerate(trn_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(imgs)
        trn_loss = criterion(output, labels)
        trn_loss.backward()
        optimizer.step()
        
        trn_loss_sum += trn_loss.item()
        
        accuracy = (torch.argmax(output,dim=1)==labels).sum()/len(imgs)
        acc_sum += accuracy
        
    trn_loss = trn_loss_sum/len(trn_loader)
    
    acc = acc_sum.item()/len(trn_loader)
    
    return trn_loss, acc

def test(model, tst_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """
    tst_loss_sum = 0; acc_sum = 0;
    
    for idx, (imgs, labels) in enumerate(tst_loader):
        with torch.no_grad():
            imgs = imgs.to(device)
            labels = labels.to(device)
            output = model(imgs)
            tst_loss = criterion(output, labels)
                    
            tst_loss_sum += tst_loss.item()
            
            accuracy = (torch.argmax(output,dim=1)==labels).sum()/len(imgs)
            acc_sum += accuracy
        
    tst_loss = tst_loss_sum/len(tst_loader)
    acc = acc_sum.item()/len(tst_loader)
    
    return tst_loss, acc


def main():
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    # data loading
    train_data_dir="C:/Users/Woojin/OneDrive - 서울과학기술대학교/2021_1/인공신경망과 딥러닝/과제/mnist-classification/mnist-classification/data/train.tar"
    test_data_dir="C:/Users/Woojin/OneDrive - 서울과학기술대학교/2021_1/인공신경망과 딥러닝/과제/mnist-classification/mnist-classification/data/test.tar"
    normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))])
    
    train_data = dataset.MNIST(data_dir=train_data_dir, transform=normalize)
    test_data = dataset.MNIST(data_dir=test_data_dir, transform=normalize)
    
    train_loader = DataLoader(train_data, batch_size=64)
    test_loader = DataLoader(test_data, batch_size=64)
    
    ###################################################### LENET-5
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    train_epoch = 10
    
    LeNet5_model = LeNet5().to(device)    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(LeNet5_model.parameters(), lr=0.01, momentum=0.9)
    
    LeNet5_cost_function = criterion.to(device)
    LeNet5_optimizer = optimizer
    
    print("LeNet5 training")
    LeNet5_time = time.time()
    LeNet5_trn_loss, LeNet5_trn_acc, LeNet5_tst_loss, LeNet5_tst_acc = [],[],[],[]
    
    for epoch in range(train_epoch):
        train_loss, train_acc = train(LeNet5_model, train_loader, device, LeNet5_cost_function, LeNet5_optimizer)
        test_loss, test_acc = test(LeNet5_model, test_loader, device, LeNet5_cost_function)
        
        LeNet5_trn_loss.append(train_loss)
        LeNet5_trn_acc.append(train_acc)
        LeNet5_tst_loss.append(test_loss)
        LeNet5_tst_acc.append(test_acc)
        
        print('epochs {}, training loss {},  training accuracy {}, validation loss {},  validation accuracy {}.'.format(epoch, train_loss, train_acc, test_loss, test_acc))
        
        if epoch+1 == 10:
            print('LeNet5 time spent: {}'.format(time.time()-LeNet5_time))
            
    ###################################################### CustomMLP
    Custom_model = CustomMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(Custom_model.parameters(), lr=0.01, momentum=0.9)
    
    Custom_cost_function = criterion.to(device)
    Custom_optimizer = optimizer
    
    print("CustomMLP training")
    Custom_time = time.time()
    Custom_trn_loss, Custom_trn_acc, Custom_tst_loss, Custom_tst_acc = [],[],[],[]
    
    for epoch in range(train_epoch):
        train_loss, train_acc = train(Custom_model, train_loader, device, Custom_cost_function, Custom_optimizer)
        test_loss, test_acc = test(Custom_model, test_loader, device, Custom_cost_function)
        
        Custom_trn_loss.append(train_loss)
        Custom_trn_acc.append(train_acc)
        Custom_tst_loss.append(test_loss)
        Custom_tst_acc.append(test_acc)
        
        print('epochs {}, training loss {},  training accuracy {}, validation loss {},  validation accuracy {}.'.format(epoch, train_loss, train_acc, test_loss, test_acc))
        
        if epoch+1 == 10:
            print('CustomMLP time spent: {}'.format(time.time()-Custom_time))
            
    ###################################################### Regularized LeNet5
    Regularized_LeNet5_model = Regularized_LeNet5().to(device)    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(Regularized_LeNet5_model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.05)
    
    Regularized_LeNet5_cost_function = criterion.to(device)
    Regularized_LeNet5_optimizer = optimizer
    
    print("Regularized_LeNet5 training")
    Regularized_LeNet5_time = time.time()
    Regularized_LeNet5_trn_loss, Regularized_LeNet5_trn_acc, Regularized_LeNet5_tst_loss, Regularized_LeNet5_tst_acc = [],[],[],[]
    
    for epoch in range(train_epoch):
        train_loss, train_acc = train(Regularized_LeNet5_model, train_loader, device, Regularized_LeNet5_cost_function, Regularized_LeNet5_optimizer)
        test_loss, test_acc = test(Regularized_LeNet5_model, test_loader, device, Regularized_LeNet5_cost_function)
        
        Regularized_LeNet5_trn_loss.append(train_loss)
        Regularized_LeNet5_trn_acc.append(train_acc)
        Regularized_LeNet5_tst_loss.append(test_loss)
        Regularized_LeNet5_tst_acc.append(test_acc)
        
        print('epochs {}, training loss {},  training accuracy {}, validation loss {},  validation accuracy {}.'.format(epoch, train_loss, train_acc, test_loss, test_acc))
        
        if epoch+1 == 10:
            print('Regularized_LeNet5 time spent: {}'.format(time.time()-Regularized_LeNet5_time))
            
            
            
    ############################################ Visualize difference between models
    trn_loss = [LeNet5_trn_loss, Custom_trn_loss, Regularized_LeNet5_trn_loss]
    trn_acc = [LeNet5_trn_acc, Custom_trn_acc, Regularized_LeNet5_trn_acc]
    tst_loss = [LeNet5_tst_loss, Custom_tst_loss, Regularized_LeNet5_tst_loss]
    tst_acc = [LeNet5_tst_acc, Custom_tst_acc, Regularized_LeNet5_tst_acc]
    train_test_plot(trn_loss, trn_acc, tst_loss, tst_acc)


def train_test_plot(trn_loss, trn_acc, val_loss, val_acc) :
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16,10))
    
    # draw train loss graph 
    axes[0, 0].plot(trn_loss[0], label='LeNet5'); axes[0, 0].plot(trn_loss[1], label='CustomMLP'); axes[0, 0].plot(trn_loss[2], label='Regularized LeNet5')
    axes[0, 0].set_title('train loss function')
    axes[0, 0].legend()
    
    # draw train acc graph
    axes[0, 1].plot(trn_acc[0], label='LeNet5'); axes[0, 1].plot(trn_acc[1], label='CustomMLP'); axes[0, 1].plot(trn_acc[2], label='Regularized LeNet5')
    axes[0, 1].set_title('train accuracy function')
    axes[0, 1].legend()
    
    # draw validation loss graph
    axes[1, 0].plot(val_loss[0], label='LeNet5'); axes[1, 0].plot(val_loss[1], label='CustomMLP'); axes[1, 0].plot(val_loss[2], label='Regularized LeNet5')
    axes[1, 0].set_title('validation loss function')
    axes[1, 0].legend()    
 
    # draw validation acc graph
    axes[1, 1].plot(val_acc[0], label='LeNet5'); axes[1, 1].plot(val_acc[1], label='CustomMLP'); axes[1, 1].plot(val_acc[2], label='Regularized LeNet5')
    axes[1, 1].set_title('validation accuracy function')
    axes[1, 1].legend()
    
    plt.savefig('output.png')

    
if __name__ == '__main__':
    main()
