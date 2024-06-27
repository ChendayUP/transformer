import torch.nn as nn
print('\n')
class testNet(nn.Module):
    def __init__(self):
        super(testNet, self).__init__()
        self.combine = []
        self.combine.append(nn.Linear(100,50))
        self.combine.append(nn.Linear(50,25))
 
net = testNet()
print(net)
class testNet2(nn.Module):
    def __init__(self):
        super(testNet2, self).__init__()
        self.combine = nn.ModuleList()
        self.combine.append(nn.Linear(100,50))
        self.combine.append(nn.Linear(50,25))
 
net2 = testNet2()
print(net2)
