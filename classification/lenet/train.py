import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from model import LeNet
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

train_set = CIFAR10(root=r"E:\CodeWorld\deepl\data", train=True, download=True, transform=transform)
test_set = CIFAR10(root=r"E:\CodeWorld\deepl\data", train=False, download=True, transform=transform)

train_dataloader = DataLoader(train_set, batch_size=36, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_set, batch_size=5000, shuffle=False, num_workers=0)

device = "cuda:0"

net = LeNet().to(device)
loss_func = CrossEntropyLoss()
optimizer = SGD(params=net.parameters(), lr=0.001, weight_decay=0.0005)

if __name__ == '__main__':
    net.train()

    for epoch in range(1,101):

        running_loss = 0.0

        for step, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            output = net(inputs)
            loss = loss_func(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step % 500 == 0 and step != 0:
                print('[%d, %5d] train_loss: %.3f' %(epoch + 1, step + 1, running_loss / 500))
                running_loss = 0.0

    torch.save(net.state_dict(), f"ckpt/lenet_cifar10_epoch_{epoch}.pth")