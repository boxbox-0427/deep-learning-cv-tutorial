import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize

from backbone import GoogLeNet
from classification.dataset import MyDataSet
import matplotlib.pyplot as plt
import numpy as np

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck') for cifar-10
device = "cuda:0"
net = GoogLeNet(num_classes=5, init_weights=True).to(device)
net.load_state_dict(torch.load(f"ckpt/googlenet_crop_epoch_100.pth"))
net.eval()

mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)
unnormalize = Normalize(mean=(-1 * mean[0] / std[0], -1 * mean[1] / std[1], -1 * mean[2] / std[2]), std=(1 / std[0], 1 / std[1], 1 / std[2]))

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

train_set = MyDataSet(root=r"../data/Agriculturecropimages/archive/kag2", transform=transform)
train_dataloader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
iter_loader = iter(train_dataloader)

def predict():

    plt.figure(figsize=(12,5), dpi=300)

    for i in range(10):
        plt.subplot(2, 5, i+1)
        img, label = next(iter_loader)
        img = img.to(device)

        with torch.no_grad():
            outputs = net(img)
            pred = torch.max(outputs, dim=1)[1].item()

        pil_img = transforms.ToPILImage()(unnormalize(img.squeeze(0)))
        pil_arr = np.array(pil_img)

        plt.imshow(pil_arr)
        plt.title(f"Ground Truth: {train_set.label[label.item()]} \n Pred Result: {train_set.label[pred]}")
    plt.subplots_adjust(hspace=0.5)
    plt.show()

if __name__ == '__main__':
    predict()
