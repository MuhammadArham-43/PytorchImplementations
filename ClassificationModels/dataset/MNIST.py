from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import ToTensor, Normalize, Compose, Resize
from torchvision.datasets import MNIST

import matplotlib.pyplot as plt
from PIL import Image


class MNISTDataset(Dataset):
    def __init__(self, train: bool = True, output_size=(227,227)) -> None:
        super(MNISTDataset, self).__init__()
        self.mnist = MNIST(
            root="data",
            train=train,
            download=True,
            transform=Compose([Resize(output_size), ToTensor(), Normalize(mean=(0.5,), std=(0.5,))]),
        )

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index):
        img, label = self.mnist[index]
        return img, label


if __name__ == "__main__":
    BATCH_SIZE = 4
    dataset = MNISTDataset(train=True)
    mnist_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    images, labels = next(iter(mnist_dataloader))
    # print(images.shape, labels.shape)
    # print(labels)

    fig = plt.figure(figsize=(28, 28))
    rows = 1
    cols = 4
    for i in range(BATCH_SIZE):
        print(images[i].shape, labels[i].item())
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(images[i].squeeze())
        plt.title(labels[i].item())

    plt.show()
