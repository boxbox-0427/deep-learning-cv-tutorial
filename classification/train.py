import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from backbone import GoogLeNet
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from dataset import MyDataSet

from log import logger

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize((224,224), antialias=True)
    ]
)

train_set = MyDataSet(root=r"../data/animal image dataset/archive/animals/animals", transform=transform)
train_dataloader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=0)

device = "cuda:0"

net = GoogLeNet(num_classes=len(train_set.label), init_weights=True).to(device)
loss_func = CrossEntropyLoss()
optimizer = Adam(params=net.parameters(), lr=0.001, weight_decay=0.0005)

if __name__ == '__main__':
    net.train()

    for epoch in range(1,101):

        running_loss = 0.0

        for step, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # for normal networks
            # output = net(inputs)
            # loss = loss_func(output, labels)
            # loss.backward()

            # for googlenet
            logits, aux_logits1, aux_logits2 = net(inputs)

            loss0 = loss_func(logits, labels)
            loss1 = loss_func(aux_logits1, labels)
            loss2 = loss_func(aux_logits2, labels)
            loss = loss0 * 0.4 + loss1 * 0.3 + loss2 * 0.3

            loss.backward()

            optimizer.step()
            running_loss += loss.item()
            # logger.info("[Iter:%d] train_loss for single img %.3f" %(step+1, loss.item()))

        logger.info('[Epoch:%d] train_loss: %.3f' %(epoch, running_loss / step))
        running_loss = 0.0

    torch.save(net.state_dict(), f"ckpt/googlenet_animal_epoch_{epoch}.pth")