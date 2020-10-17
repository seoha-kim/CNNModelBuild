import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

LEARNING_RATE = 0.001
NUM_EPOCH = 10
PRINT_EVERY = 1000
SAVE_EVERY = 1
BATCH_SIZE = 1
save_path = "./model/"

trainset = CIFAR10(root='../data/cifar', train=True, download=True, transform=transform)
testset = CIFAR10(root='../data/cifar', train=False, download=True, transform=transform)
train_loader = DataLoader(trainset, shuffle=True, num_workers=8, batch_size=BATCH_SIZE)
test_loader = DataLoader(testset, num_workers=8, batch_size=BATCH_SIZE)

def conv_2_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model

def conv_3_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model

class VGG(nn.Module):
    def __init__(self, base_dim, num_classes=10):
        super(VGG, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(3, base_dim),
            conv_2_block(base_dim, 2*base_dim),
            conv_3_block(2*base_dim, 4*base_dim),
            conv_3_block(4*base_dim, 8*base_dim),
            conv_3_block(8*base_dim, 8*base_dim)
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(8*base_dim*7*7, 100),
            nn.ReLU(True),
            nn.Linear(100, 20),
            nn.ReLU(True),
            nn.Linear(20, num_classes),
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
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
    model = VGG(base_dim=16).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-3)

    for i in model.named_children():
        print(i)

    train(model, train_loader, criterion, optimizer)
    # checkpoint = torch.load(save_path)
    # model.load_state_dict(checkpoint['model_state_dict'])

    test(model, test_loader)