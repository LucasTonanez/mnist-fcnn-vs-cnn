import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
LR = 1e-3

MEAN, STD = 0.1307, 0.3081

def accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

class FCNet(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, train_loader, val_loader, epochs):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

        tr_acc = accuracy(model, train_loader)
        va_acc = accuracy(model, val_loader)
        print(f"Epoch {ep:2d}/{epochs} | train_acc={tr_acc:.4f} | val_acc={va_acc:.4f}")

def main():
    base_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,))
    ])

    aug_tf = transforms.Compose([
        transforms.RandomAffine(
            degrees=10,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05)
        ),
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,))
    ])

    ds_plain = datasets.MNIST(root="data", train=True, download=True, transform=base_tf)
    ds_aug   = datasets.MNIST(root="data", train=True, download=True, transform=aug_tf)
    ds_test  = datasets.MNIST(root="data", train=False, download=True, transform=base_tf)

    # 50k train / 10k val split
    gen = torch.Generator().manual_seed(RANDOM_STATE)
    train_size, val_size = 50000, 10000

    train_plain, val_plain = random_split(ds_plain, [train_size, val_size], generator=gen)
    train_aug,   val_aug   = random_split(ds_aug,   [train_size, val_size], generator=gen)

    test_loader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

    # FCNN (no augmentation)
    print("\n=== FCNN ===")
    fcnn = FCNet()
    train_loader = DataLoader(train_plain, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_plain,   batch_size=BATCH_SIZE, shuffle=False)
    train_model(fcnn, train_loader, val_loader, epochs=10)
    print("FCNN train_acc:", accuracy(fcnn, train_loader))
    print("FCNN val_acc:", accuracy(fcnn, val_loader))
    print("FCNN test_acc:", accuracy(fcnn, test_loader))

    # CNN (augmentation on train)
    print("\n=== CNN ===")
    cnn = CNNNet()
    train_loader = DataLoader(train_aug, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_aug,   batch_size=BATCH_SIZE, shuffle=False)
    train_model(cnn, train_loader, val_loader, epochs=15)
    print("CNN train_acc:", accuracy(cnn, train_loader))
    print("CNN val_acc:", accuracy(cnn, val_loader))
    print("CNN test_acc:", accuracy(cnn, test_loader))

if __name__ == "__main__":
    main()