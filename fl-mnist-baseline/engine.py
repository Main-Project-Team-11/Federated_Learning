import torch
import torch.nn as nn

def train(net, train_loader, epochs, device):
    """Trains `net` locally for `epochs` epochs on this client's train_loader."""
    net.to(device)
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def test(net, data_loader, device):
    """Evaluates `net` on `data_loader`. Returns (average_loss, accuracy)."""
    net.to(device)
    net.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total, loss_sum = 0, 0, 0.0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss_sum += criterion(outputs, labels).item() * labels.size(0)
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total == 0:
        raise ValueError("data_loader must contain at least one sample")

    return loss_sum / total, correct / total
