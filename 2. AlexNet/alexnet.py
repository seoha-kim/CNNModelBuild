import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from collections import OrderedDict
import os
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


LEARNING_RATE = 0.001
NUM_EPOCH = 10
PRINT_EVERY = 1000
SAVE_EVERY = 1
BATCH_SIZE = 8
save_path = "./model/"

trainset = CIFAR10(root='../data/cifar', train=True, download=True, transform=transform)
testset = CIFAR10(root='../data/cifar', train=False, download=True, transform=transform)
train_loader = DataLoader(trainset, shuffle=True, num_workers=8, batch_size=BATCH_SIZE)
test_loader = DataLoader(testset, num_workers=8, batch_size=BATCH_SIZE)

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 2 * 2, 4096, bias=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096, bias=1),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes, bias=1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def save_model(epoch, model, optimizer, PATH):
    torch.save({"path" : PATH,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dixt()
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
            save_model(epoch+1, model, optimizer, "./model/")
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
    model = AlexNet().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-3)

    for i in model.named_children():
        print(i)

    train(model, train_loader, criterion, optimizer)
    # checkpoint = torch.load(save_path)
    # model.load_state_dict(checkpoint['model_state_dict'])

    test(model, test_loader)

