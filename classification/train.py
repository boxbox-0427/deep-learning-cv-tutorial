import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from alexnet import AlexNet
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

train_set = CIFAR10(root=r"E:\CodeWorld\deepl\data", train=True, download=True, transform=transform)
test_set = CIFAR10(root=r"E:\CodeWorld\deepl\data", train=False, download=True, transform=transform)

train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)

device = "cuda:0"

net = AlexNet().to(device)
loss_func = CrossEntropyLoss()
optimizer = Adam(params=net.parameters(), lr=0.0001, weight_decay=0.00005)

if __name__ == '__main__':
    net.train()

    for epoch in range(1,101):

        running_loss = 0.0

        for step, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = net(inputs)
            loss = loss_func(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step % 500 == 0 and step != 0:
                print('[Epoch:%d, Iter:%5d] train_loss: %.3f' %(epoch + 1, step + 1, running_loss / 500))
                running_loss = 0.0

    torch.save(net.state_dict(), f"{type(net).__name__.lower()}/ckpt/{type(net).__name__.lower()}_cifar10_epoch_{epoch}.pth")