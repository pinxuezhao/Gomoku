import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_dim=4, output_dim=4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_layer_filters = 64
        self.conv_layer_kernel_size = (3,3)
        self.residual_layer_num = 2
        self.value_head_hidden_layer_size = 3

        self.learning_rate = 0.1
        self.momentum = 0.9

        self.backbone = self.build_model()
        self.p_net = Policy_Head()
        self.v_net = Value_Head()

    def build_model(self):
        module_list = []
        module_list.append(Conv_Layer(4, self.conv_layer_filters,(3,3), (1,1), (1,1)))
        for _ in range(self.residual_layer_num):
            module_list.append(Residual_Layer(self.conv_layer_filters))
        return nn.Sequential(*module_list)

    def forward(self, x): 
        x = self.backbone(x)
        p = self.p_net(x)
        v = self.v_net(x)
        return (v, p)
        
class Conv_Layer(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv_layer = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU()
                )

    def forward(self, x):
        return self.conv_layer(x)

class Residual_Layer(nn.Module):
    
    def __init__(self, num_channels):
        super().__init__()
        self.conv_1 = Conv_Layer(num_channels, num_channels, (3,3), (1,1), (1,1))
        self.conv_2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.bn = nn.BatchNorm2d(num_features=num_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return x+self.relu(self.bn(self.conv_2(self.conv_1(x))))

class Value_Head(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_0 = Conv_Layer(64, 1, (1,1), (1,1), (0,0))
        self.linear_0 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.linear_1 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv_0(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear_0(x)
        x = self.relu(x)
        x = self.linear_1(x)
        
        return torch.tanh(x)
        
class Policy_Head(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_0 = Conv_Layer(64, 1, (1,1), (1,1), (0,0))
        self.linear = nn.Linear(64, 64)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_0(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return self.softmax(x)
"""
net = Net()
x = torch.ones(1,4,8,8)
print(net(x))
print(torch.sum(net(x)[1]))
"""
