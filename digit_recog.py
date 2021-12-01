import torch.nn as nn
class neural_1(nn.Module):
    def __init__(self):
        super(neural_1,self).__init__()
        self.convolutional = nn.Sequential(
                                            nn.Conv2d(1,16,kernel_size=(3,3)),
                                            nn.ReLU(),
                                            nn.MaxPool2d(3,stride=1),
                                            nn.Conv2d(16,32,kernel_size=(3,3)),
                                            nn.ReLU(),
                                            nn.MaxPool2d(5,stride=2),
                                            nn.Conv2d(32,10,kernel_size=(3,3)),
                                            nn.ReLU(),
                                            nn.MaxPool2d(5,stride=2)
                )
        
        self.linear = nn.Sequential(nn.Linear(1210,1000),
                                   nn.ReLU(),
                                   nn.Linear(1000,500),
                                   nn.ReLU(),
                                   nn.Linear(500,250),
                                   nn.ReLU(),
                                   nn.Dropout(0.5),
                                   nn.Linear(250,10),
                                   nn.Sigmoid())
    def forward(self,x):
        out = self.convolutional(x)
        out = out.view(out.size(0),-1)
        out = self.linear(out)
        
        return out