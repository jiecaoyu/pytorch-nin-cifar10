import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import data
import cPickle as pickle
import numpy

from torch.autograd import Variable

trainset = data.dataset(root='./data', train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
        shuffle=True, num_workers=2)

testset = data.dataset(root='./data', train=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
        shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.classifier = nn.Sequential(
                nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(160,  96, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Dropout(0.5),

                nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                nn.Dropout(0.5),

                nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(192,  10, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),

                )

    def forward(self, x):
        x = self.classifier(x)
        x = x.view(x.size(0), 10)
        return x


model = Net()
print model
model.cuda()

pretrained=False
if pretrained:
    params = pickle.load(open('data/params', 'r'))
    index = -1
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            index = index + 1
            weight = torch.from_numpy(params[index])
            m.weight.data.copy_(weight)
            index = index + 1
            bias = torch.from_numpy(params[index])
            m.bias.data.copy_(bias)
else:
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.05)
            m.bias.data.normal_(0, 0.0)



criterion = nn.CrossEntropyLoss()
param_dict = dict(model.named_parameters())
params = []

base_lr = 0.1

for key, value in param_dict.items():
    if key == 'classifier.20.weight':
        params += [{'params':[value], 'lr':0.1 * base_lr, 
            'momentum':0.95, 'weight_decay':0.0001}]
    elif key == 'classifier.20.bias':
        params += [{'params':[value], 'lr':0.1 * base_lr, 
            'momentum':0.95, 'weight_decay':0.0000}]
    elif 'weight' in key:
        params += [{'params':[value], 'lr':1.0 * base_lr,
            'momentum':0.95, 'weight_decay':0.0001}]
    else:
        params += [{'params':[value], 'lr':2.0 * base_lr,
            'momentum':0.95, 'weight_decay':0.0000}]


optimizer = optim.SGD(params, lr=0.1, momentum=0.9)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.data[0],
                optimizer.param_groups[1]['lr']))
            

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in testloader:
        data, target = Variable(data.cuda()), Variable(target.cuda())
        output = model(data)
        test_loss += criterion(output, target).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss * 128., correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))

def adjust_learning_rate(optimizer, epoch):
    if epoch%80==0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

def print_std():
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            print torch.std(m.weight.data)

for epoch in range(1, 320):
    adjust_learning_rate(optimizer, epoch)
    train(epoch)
    test()
