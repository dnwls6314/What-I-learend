
# import some packages you need here
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt

import io
import tarfile


class MNIST(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir, transform):
        self.data_dir=data_dir
        self.tars = tarfile.open(name=data_dir, mode='r')
        self.files = self.tars.getmembers()
        self.labels = pd.DataFrame([[t.name[-11:-6], t.name, int(t.name[-5])] for t in self.files[1:]], columns=['idx', 'name', 'label'])
        self.transform = transform
        
        
    def __len__(self):
        length = len(self.labels)
        return length
        
    def __getitem__(self, idx):
        image = self.tars.extractfile(self.labels.iloc[idx, 1]).read()
        img = Image.open(io.BytesIO(image))
        label = torch.tensor(self.labels.iloc[idx, 2])
        
        if self.transform:
            img = self.transform(img)
            
        
        return img, label
    
    

if __name__ == '__main__':
    
    # write test codes to verify your implementations
    data_dir="C:/Users/Woojin/OneDrive - 서울과학기술대학교/2021_1/인공신경망과 딥러닝/과제/mnist-classification/mnist-classification/data/train.tar"
    
    normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))])
    
    dataset = MNIST(data_dir=data_dir, transform=normalize)
    # dataset = MNIST(data_dir="..//data//train.tar", transform=normalize)
    
    train_data, val_data, test_data = random_split(dataset, [40000, 10000, 10000])
    
    train_loader = DataLoader(train_data, batch_size=64)
    val_loader = DataLoader(val_data, batch_size=64)
    test_loader = DataLoader(test_data)
    
    image, label = next(iter(train_loader))
    
    sample_img = np.array(image[0]).reshape((28,28))
    sample_label = label[0]
    plt.xticks()
    plt.yticks()
    plt.imshow(sample_img)
    plt.title('labeled : {}'.format(sample_label))
    
