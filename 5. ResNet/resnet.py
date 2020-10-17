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

def conv_block_1(in_dim, out_dim, act_fn, stride=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride),
        act_fn
    )
    return model

def conv_block_3(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        act_fn
    )
    return model

class BottleNeck(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, act_fn, down=False):
        super(BottleNeck, self).__init__()
        self.down = down

        # 특성 지도의 크기가 감소하는 경우
        if self.down:
            self.layer = nn.Sequential(
                conv_block_1(in_dim, mid_dim, act_fn, 2),
                conv_block_3(mid_dim, mid_dim, act_fn),
                conv_block_1(mid_dim, out_dim, act_fn),
            )
            self.downsample = nn.Conv2d(in_dim, out_dim, 1, 2)

        # 특성 지도의 크기가 그대로인 경우
        else:
            self.layer = nn.Sequential(
                conv_block_1(in_dim, mid_dim, act_fn),
                conv_block_3(mid_dim, mid_dim, act_fn),
                conv_block_1(mid_dim, out_dim, act_fn),
            )

        # 더하기를 위해 차원을 맞춰주는 부
        self.dim_equalizer = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, x):
        if self.down:
            downsample = self.downsample(x)
            out = self.layer(x)
            out = out + downsample
        else:
            out = self.layer(x)
            if x.size() is not out.size():
                x = self.dim_equalizer(x)
            out = out + x
        return out

class ResNet(nn.Module):
    def __init__(self, base_dim, num_classes=2):
        super(ResNet, self).__init__()
        self.act_fn = nn.ReLU()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(3, base_dim, 7, 2, 3),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
        )

        self.layer_2 = nn.Sequential(
            BottleNeck(base_dim, base_dim, base_dim*4, self.act_fn),
            BottleNeck(base_dim*4, base_dim, base_dim*4, self.act_fn),
            BottleNeck(base_dim*4, base_dim, base_dim*4, self.act_fn, down=True)
        )

        self.layer_3 = nn.Sequential(
            BottleNeck(base_dim*4, base_dim*2, base_dim*8, self.act_fn),
            BottleNeck(base_dim*8, base_dim*2, base_dim*8, self.act_fn),
            BottleNeck(base_dim*8, base_dim*2, base_dim*8, self.act_fn),
            BottleNeck(base_dim*8, base_dim*2, base_dim*8, self.act_fn, down=True)
        )

        self.layer_4 = nn.Sequential(
            BottleNeck(base_dim * 8, base_dim * 4, base_dim * 16, self.act_fn),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.act_fn),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.act_fn),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.act_fn),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.act_fn),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.act_fn, down=True),
        )
        self.layer_5 = nn.Sequential(
            BottleNeck(base_dim*16, base_dim*8, base_dim*32, self.act_fn),
            BottleNeck(base_dim*32, base_dim*8, base_dim*32, self.act_fn),
            BottleNeck(base_dim*32, base_dim*8, base_dim*32, self.act_fn),
        )
        self.avgpool = nn.AvgPool2d(7,1)
        self.fc_layer = nn.Linear(base_dim*32, num_classes)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = self.avgpool(out)
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
    model = ResNet(base_dim=64).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-3)

    for i in model.named_children():
        print(i)

    train(model, train_loader, criterion, optimizer)
    # checkpoint = torch.load(save_path)
    # model.load_state_dict(checkpoint['model_state_dict'])

    test(model, test_loader)