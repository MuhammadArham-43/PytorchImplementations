from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import ToTensor, Normalize, Compose, Resize
from torchvision.datasets import CIFAR10

import matplotlib.pyplot as plt
import numpy as np


class CIFARDataset(Dataset):
    def __init__(self, train: bool = True, image_size=(512, 512)) -> None:
        super(CIFARDataset, self).__init__()
        self.cifar = CIFAR10(
            root="data/cifar",
            train=train,
            download=True,
            transform=Compose([Resize(image_size), ToTensor(),
                              Normalize(mean=(0.5,), std=(0.5,))]),
        )

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, index):
        img, label = self.cifar[index]
        return img, label


if __name__ == "__main__":
    dataset = CIFARDataset(train=True)
    cifar_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    images, labels = next(iter(cifar_dataloader))
    # print(images.shape, labels.shape)
    # print(labels)

    fig = plt.figure(figsize=(28, 28))
    rows = 1
    cols = 4
    for i in range(4):
        # print(images[i].shape, labels[i].item())
        fig.add_subplot(rows, cols, i + 1)

        image = images[i].squeeze().detach().cpu().numpy()
        image = np.moveaxis(image, 0, 2)

        plt.imshow(image)
        plt.title(labels[i].item())

    plt.show()
