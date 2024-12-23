import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F


class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.ReLU = nn.ReLU()
        self.c1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.s1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.c2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.s2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.flatten = nn.Flatten()
        self.f1 = nn.Linear(1024*128, 512)
        self.f2 = nn.Linear(512, 206)
        

        
        for m in self.modules():
            #initialise parameters
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0) 
            
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0) 

    def forward(self,x):
        x = self.ReLU(self.c1(x))
        x = self.s1(x)
        x = self.ReLU(self.c2(x))
        x = self.s2(x)
        x = self.flatten(x)
        x = self.ReLU(self.f1(x))
        x = F.dropout(x, 0)
        x = self.f2(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cnn().to(device)
    print(summary(model, (3, 128, 128)))