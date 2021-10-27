
import torch.nn as nn
from collections import OrderedDict

class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self):
        super(LeNet5, self).__init__()
        
        self.convnet = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2, stride=1)), # parameter: 5*5*6
            ('relu1', nn.ReLU()),
            ('sub2', nn.MaxPool2d(kernel_size=2)),
            ('conv3', nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)), # parameter: 5*5*6*16
            ('relu3', nn.ReLU()),
            ('sub4', nn.MaxPool2d(kernel_size=2))
        ]))
        
        self.classifier = nn.Sequential(OrderedDict([
            ('conv5', nn.Linear(in_features=16*5*5, out_features=120)), # parameter: 16*5*5*120
            ('relu5', nn.ReLU()),
            ('f6', nn.Linear(in_features=120, out_features=84)), # parameter: 120*84
            ('relu6', nn.ReLU()),
            ('f7', nn.Linear(in_features=84, out_features=10)), # parameter: 84*10
            ('sftmx7', nn.Softmax(dim=1))
        ]))
        
        # forwarding parameter = (5*5*6) + (5*5*6*16) + (16*5*5*120) + (120*84) + (84*10) = 61,470
        # total parameter: 61,470 * 2 = 122,940 by backpropagation

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(-1, 16*5*5)
        output = self.classifier(output)
        return output


class CustomMLP(nn.Module):
    """ Your custom MLP model

        - Note that the number of model parameters should be about the same
          with LeNet-5
    """

    def __init__(self):
        super(CustomMLP, self).__init__()
        
        self.myMLP = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(in_features=28*28, out_features=70)), # parameter: 28*28*70 = 54,880
            ('dropout1', nn.Dropout(p=0.5)),
            ('relu1', nn.ReLU()),
            ('linear2', nn.Linear(in_features=70, out_features=64)), # parameter: 70*64
            ('dropout2', nn.Dropout(p=0.5)),
            ('relu2', nn.ReLU()),
            ('linear3', nn.Linear(in_features=64, out_features=24)), # parameter: 64*24
            ('dropout3', nn.Dropout(p=0.5)),
            ('relu3', nn.ReLU()),
            ('linear4', nn.Linear(in_features=24, out_features=16)), # parameter: 24*16
            ('dropout4', nn.Dropout(p=0.5)),
            ('relu4', nn.ReLU()),
            ('linear5', nn.Linear(in_features=16, out_features=10)), # parameter: 16*10
            ('sftmx5', nn.Softmax(dim=1)),
        ]))
        
                
        # forwarding parameter = (28*28*70) + (70*64) + (64*24) + (24*16) + (16*10) = 61,440
        # total parameter: 61,440 * 2 = 122,880 by backpropagation

    def forward(self, img):
        img = img.view(-1, 28*28)
        output = self.myMLP(img)
        
        return output
    

class Regularized_LeNet5(nn.Module):
    
    def __init__(self):
        super(Regularized_LeNet5, self).__init__()
        
        self.convnet = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2, stride=1)), # parameter: 5*5*6
            ('relu1', nn.ReLU()),
            ('sub2', nn.MaxPool2d(kernel_size=2)),
            ('conv3', nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)), # parameter: 5*5*6*16
            ('relu3', nn.ReLU()),
            ('sub4', nn.MaxPool2d(kernel_size=2))
        ]))
        
        self.classifier = nn.Sequential(OrderedDict([
            ('conv5', nn.Linear(in_features=16*5*5, out_features=120)), # parameter: 16*5*5*120
            ('dropout5', nn.Dropout(p=0.4)),
            ('relu5', nn.ReLU()),
            ('f6', nn.Linear(in_features=120, out_features=84)), # parameter: 120*84
            ('dropout6', nn.Dropout(p=0.4)),
            ('relu6', nn.ReLU()),
            ('f7', nn.Linear(in_features=84, out_features=10)), # parameter: 84*10
            ('sftmx7', nn.Softmax(dim=1))
        ]))
        
        # forwarding parameter = (5*5*6) + (5*5*6*16) + (16*5*5*120) + (120*84) + (84*10) = 61,470
        # total parameter: 61,470 * 2 = 122,940 by backpropagation

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(-1, 16*5*5)
        output = self.classifier(output)
        return output
