import torch
import torchvision.transforms as transforms
from torchvision.transforms import Normalize

from model import LeNet
from train import test_dataloader
import matplotlib.pyplot as plt
import numpy as np

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
device = "cuda:0"
net = LeNet().to(device)
net.load_state_dict(torch.load("ckpt/lenet_cifar10_epoch_100.pth"))
net.eval()
iter_loader = iter(test_dataloader)
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)
unnormalize = Normalize(mean=(-1 * mean[0] / std[0], -1 * mean[1] / std[1], -1 * mean[2] / std[2]), std=(1 / std[0], 1 / std[1], 1 / std[2]))

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
        plt.title(f"Ground Truth: {classes[label.item()]} \n Pred Result: {classes[pred]}")
    plt.subplots_adjust(hspace=0.5)
    plt.show()

if __name__ == '__main__':
    predict()
