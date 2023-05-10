import torch
import torchvision.transforms as transforms
from PIL import Image

from model import LeNet

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
device = "cuda:0"
net = LeNet().to(device)
net.load_state_dict(torch.load("ckpt/lenet_cifar10_epoch_10"))

def predict():
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    img = Image.open("demo.jpeg")
    img = transform(img).to(device)
    img = torch.unsqueeze(img, dim=0)

    with torch.no_grad():
        outputs = net(img)
        pred = torch.max(outputs, dim=1)[1].item()
    print(classes[int(pred)])

if __name__ == '__main__':
    predict()
