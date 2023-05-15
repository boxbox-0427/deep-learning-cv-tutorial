from torch.utils.data import Dataset
from PIL import Image
import os
from log import logger

class MyDataSet(Dataset):

    def __init__(self, root, transform):
        super(MyDataSet, self).__init__()
        self.label = os.listdir(root)
        self.data = []
        self.transform = transform

        for i in self.label:
            path = os.listdir(os.path.join(root, i))

            for j in path:
                self.data.append(
                        [os.path.join(root, i, j), i]
                    )
        logger.info("Dataset Loaded!")

    def __getitem__(self, index):
        data: list = self.data[index]

        return self.transform(Image.open(data[0])), self.label.index(data[1])

    def __len__(self):
        return len(self.data)
