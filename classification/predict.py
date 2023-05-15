import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize

from backbone import ResNet101, DenseNet201
from classification.dataset import MyDataSet
import matplotlib.pyplot as plt
import numpy as np
from log import logger

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck') for cifar-10

mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)
unnormalize = Normalize(mean=(-1 * mean[0] / std[0], -1 * mean[1] / std[1], -1 * mean[2] / std[2]),
                        std=(1 / std[0], 1 / std[1], 1 / std[2]))

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize((224, 224), antialias=True)
    ]
)

test_set = MyDataSet(root=r"../data/animal image dataset/archive/animals/animals", transform=transform)
test_dataloader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)
iter_loader = iter(test_dataloader)

device = "cuda:0"
net = DenseNet201(num_classes=len(test_set.label)).to(device)
net.load_state_dict(torch.load(f"ckpt/resnet101_animal_epoch_100.pth"))
net.eval()


def predict():
    plt.figure(figsize=(12, 5), dpi=300)

    for i in range(10):
        plt.subplot(2, 5, i + 1)
        img, label = next(iter_loader)
        img = img.to(device)

        with torch.no_grad():
            outputs = net(img)
            pred = torch.max(outputs, dim=1)[1].item()

        pil_img = transforms.ToPILImage()(unnormalize(img.squeeze(0)))
        pil_arr = np.array(pil_img)

        plt.imshow(pil_arr)
        plt.title(f"Ground Truth: {test_set.label[label.item()]} \n Pred Result: {test_set.label[pred]}")
    plt.subplots_adjust(hspace=0.5)
    plt.show()

    logger.info("Predict Done!")


if __name__ == '__main__':
    predict()
