import os
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import FashionCNN


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / len(data_loader), 100 * correct / total


def save_graph(train_values, test_values, title, ylabel, filename):
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(train_values) + 1)
    plt.plot(epochs, train_values, marker="o", label="Train")
    plt.plot(epochs, test_values, marker="s", label="Test")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main():
    set_seed(42)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    log_path = os.path.join(base_dir, "logs.txt")
    acc_graph_path = os.path.join(base_dir, "accuracy_graph.png")
    loss_graph_path = os.path.join(base_dir, "loss_graph.png")
    model_save_path = os.path.join(base_dir, "fashion_cnn.pth")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Fashion MNIST PyTorch CNN Training Log\n")
        f.write("=" * 50 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    train_dataset = datasets.FashionMNIST(
        root=os.path.join(base_dir, "data"),
        train=True,
        download=True,
        transform=train_transform,
    )
    train_eval_dataset = datasets.FashionMNIST(
        root=os.path.join(base_dir, "data"),
        train=True,
        download=True,
        transform=test_transform,
    )
    test_dataset = datasets.FashionMNIST(
        root=os.path.join(base_dir, "data"),
        train=False,
        download=True,
        transform=test_transform,
    )

    use_cuda = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=2,
        pin_memory=use_cuda,
    )
    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=use_cuda,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=use_cuda,
    )

    model = FashionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    epochs = 30
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-5,
    )

    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        _, train_acc = evaluate(model, train_eval_loader, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        log = (
            f"Epoch [{epoch:02d}/{epochs}] "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_acc:.2f}%, "
            f"Test Loss: {test_loss:.4f}, "
            f"Test Acc: {test_acc:.2f}%"
        )
        print(log)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log + "\n")

    final_log = (
        "\nFinal Result\n"
        f"Train Accuracy: {train_acc_list[-1]:.2f}%\n"
        f"Test Accuracy: {test_acc_list[-1]:.2f}%\n"
    )
    print(final_log)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(final_log)

    save_graph(
        train_acc_list,
        test_acc_list,
        "Fashion MNIST PyTorch CNN Accuracy",
        "Accuracy (%)",
        acc_graph_path,
    )
    save_graph(
        train_loss_list,
        test_loss_list,
        "Fashion MNIST PyTorch CNN Loss",
        "Loss",
        loss_graph_path,
    )
    torch.save(model.state_dict(), model_save_path)

    print("Saved files:")
    print(" - logs.txt")
    print(" - accuracy_graph.png")
    print(" - loss_graph.png")
    print(" - fashion_cnn.pth")


if __name__ == "__main__":
    main()
