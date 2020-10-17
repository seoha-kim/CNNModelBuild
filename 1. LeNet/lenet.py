import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from collections import OrderedDict
import os
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
transform = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_EPOCH = 10
PRINT_EVERY = 1000
SAVE_EVERY = 1
save_path = "./model/"

trainset = MNIST(root='../data/cifar', train=True, download=True, transform=transform)
testset = MNIST(root='../data/cifar', train=False, download=True, transform=transform)
train_loader = DataLoader(trainset, shuffle=True, num_workers=8, batch_size=BATCH_SIZE)
test_loader = DataLoader(testset, num_workers=8, batch_size=BATCH_SIZE)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.c1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(3, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))
        self.c2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

        self.c3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu3', nn.ReLU())
        ]))

        self.f4 = nn.Sequential(OrderedDict([
            ('f4', nn.Linear(120, 84)),
            ('relu4', nn.ReLU())
        ]))

        self.f5 = nn.Sequential(OrderedDict([
            ('f5', nn.Linear(84, 10)),
            ('softmax5', nn.LogSoftmax(dim=-1))
        ]))


    def forward(self, x):
        out = self.c1(x)
        x = self.c2(out)
        out = self.c2(out)
        out += x
        out = self.c3(out)
        out = out.view(x.size(0), -1)
        out = self.f4(out)
        out = self.f5(out)
        return out

def save_model(epoch, model, optimizer, PATH):
    torch.save({"path" : PATH,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, './epoch_{}.tar'.format(str(epoch)))

def train(model, data_loader, criterion, optimizer):
    print("Training...")
    loss_arr = []
    for epoch in range(NUM_EPOCH):
        for i, (imgs, labels) in enumerate(data_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, argmax = torch.max(outputs, 1)
            accuracy = (labels==argmax).float().mean()

            if i % PRINT_EVERY == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%".format(
                    epoch+1, NUM_EPOCH, i+1, len(data_loader), loss.item(), accuracy.item()*100
                ))
                loss_arr.append(loss.cpu().detach().numpy())

        if (epoch+1) % SAVE_EVERY == 0:
            if not os.path.exists("./model"):
                os.makedirs("./model")
            save_model(epoch+1, model, optimizer, save_path)
            print("checkpoint saved")

    plt.plot(loss_arr)
    plt.show()

def test(model, data_loader):
    print("Testing..")
    model.eval()
    correct = 0; total= 0
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(data_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, argmax = torch.max(outputs, 1)
            total += imgs.size(0)
            correct += (labels==argmax).sum().item()
        print("Test accuracy for {} images: {:.2f}%".format(total, correct / total * 100))


if __name__ == "__main__":
    model = LeNet().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-3)

    for i in model.named_children():
        print(i)

    train(model, train_loader, criterion, optimizer)
    # checkpoint = torch.load(save_path)
    # model.load_state_dict(checkpoint['model_state_dict'])

    test(model, test_loader)

