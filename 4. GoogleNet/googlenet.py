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

def conv_1(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 1, 1),
        nn.ReLU(),
    )
    return model

def conv_1_3(in_dim, mid_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, mid_dim, 1, 1),
        nn.ReLU(),
        nn.Conv2d(mid_dim, out_dim, 3, 1, 1),
        nn.ReLU()
    )
    return model

def conv_1_5(in_dim, mid_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, mid_dim, 1, 1),
        nn.ReLU(),
        nn.Conv2d(mid_dim, out_dim, 5, 1, 2),
        nn.ReLU()
    )
    return model

def max_3_1(in_dim, out_dim):
    model = nn.Sequential(
        nn.MaxPool2d(3, 1, 1),
        nn.Conv2d(in_dim, out_dim, 1, 1),
        nn.ReLU()
    )
    return model

class inception_module(nn.Module):
    def __init__(self, in_dim, out_dim_1, mid_dim_3, out_dim_3, mid_dim_5, out_dim_5, pool):
        super(inception_module, self).__init__()
        self.conv_1 = conv_1(in_dim, out_dim_1)
        self.conv_1_3 = conv_1_3(in_dim, mid_dim_3, out_dim_3)
        self.conv_1_5 = conv_1_5(in_dim, mid_dim_5, out_dim_5)
        self.max_3_1 = max_3_1(in_dim, pool)

    def forward(self, x):
        out_1 = self.conv_1(x)
        out_2 = self.conv_1_3(x)
        out_3 = self.conv_1_5(x)
        out_4 = self.max_3_1(x)
        output = torch.cat([out_1, out_2, out_3, out_4], 1)
        return output

class GoogLeNet(nn.Module):
    def __init__(self, base_dim, num_classes=10):
        super(GoogLeNet, self).__init__()
        self.num_classes = num_classes
        self.layer_1 = nn.Sequential(
            nn.Conv2d(3, base_dim, 7, 2, 3),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(base_dim, base_dim*3, 3, 1, 1),
            nn.MaxPool2d(3, 2, 1),
        )

        self.layer_2 = nn.Sequential(
            inception_module(base_dim*3, 64, 96, 128, 16, 32, 32),
            inception_module(base_dim*4, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(3, 2, 1)
        )

        self.layer_3 = nn.Sequential(
            inception_module(480, 192, 96, 208, 16, 48, 64),
            inception_module(512, 160, 112, 224, 24, 64, 64),
            inception_module(512, 128, 128, 256, 24, 64, 64),
            inception_module(512, 112, 144, 288, 32, 64, 64),
            inception_module(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(3, 2, 1),
        )

        self.layer_4 = nn.Sequential(
            inception_module(832, 256, 160, 320, 32, 128, 128),
            inception_module(832, 384, 192, 384, 48, 128, 128),
            nn.AvgPool2d(7,1),
        )

        self.layer_5 = nn.Dropout2d(0.4)
        self.fc_layer = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = out.view(BATCH_SIZE, -1)
        out = self.fc_layer(out)
        return out


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
    model = GoogLeNet(base_dim=64).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-3)

    for i in model.named_children():
        print(i)

    train(model, train_loader, criterion, optimizer)
    # checkpoint = torch.load(save_path)
    # model.load_state_dict(checkpoint['model_state_dict'])

    test(model, test_loader)