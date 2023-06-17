import torch
import torch.nn as nn
import torch.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from models.AlexNet import AlexNet
from dataset.CIFAR import CIFARDataset

from torch.utils.data import DataLoader

from tqdm import tqdm

if __name__ == "__main__":
    INPUT_SIZE = (227, 227)
    BATCH_SIZE = 8
    IN_CHANNELS = 3
    NUM_CLASSES = 10
    LEARNING_RATE = 0.01
    NUM_EPOCHS = 10
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset = CIFARDataset(train=True, image_size=INPUT_SIZE)
    test_dataset = CIFARDataset(train=False, image_size=INPUT_SIZE)

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = AlexNet(num_classes=NUM_CLASSES, in_channels=IN_CHANNELS)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    writer = SummaryWriter()

    step = 0
    for epoch in range(NUM_EPOCHS):
        for idx, (images, labels) in enumerate(tqdm(train_dataloader)):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            writer.add_scalar("Loss / Iteration", loss, step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

        print(f'EPOCH {epoch + 1} / {NUM_EPOCHS}: Loss: {loss}')
        writer.add_scalar('Loss / Epoch', loss, epoch)
        writer.flush()
