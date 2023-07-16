import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
from model.unet import UNet
from dataset.forestData import ForestData

import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

if __name__ == "__main__":

    META_DATA_FILE = 'data/ForestData/Forest Segmented/meta_data.csv'
    IMAGES_DIR_PATH = 'data/ForestData/Forest Segmented/images'
    MASK_DIR_PATH = 'data/ForestData/Forest Segmented/masks'

    MODEL_SAVE_DIR = 'results/models'
    IMG_SAVE_DIR = 'results'
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(IMG_SAVE_DIR, exist_ok=True)

    BATCH_SIZE = 4
    NUM_EPOCHS = 100
    IMAGE_WIDTH = 256
    IMAGE_HEIGHT = 256
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LEARNING_RATE = 0.01

    image_transforms = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])

    dataset = ForestData(
        META_DATA_FILE, IMAGES_DIR_PATH, MASK_DIR_PATH, transforms=image_transforms
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    unet = UNet(in_channels=3, out_channels=1).to(DEVICE)

    lossFn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(unet.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    writer = SummaryWriter()

    step = 0

    images, masks = next(iter(dataloader))
    
    images = images.to(DEVICE)
    masks = masks.float().unsqueeze(1).to(DEVICE)
    for epoch in range(NUM_EPOCHS):
        # for idx, (images, masks) in enumerate(tqdm(dataloader)):
        unet.train()
        predictions = unet(images)
        loss = lossFn(predictions, masks)
        writer.add_scalar('Loss / Iteration', loss, step)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % 10 == 0:
            unet.eval()
            val_image = images[0]
            val_mask = masks[0]
            # print(val_image.shape, val_mask.shape)
            pred = unet(val_image.unsqueeze(0).to(DEVICE))
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float()
            # print(pred.shape)
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1.imshow(transforms.ToPILImage()(val_image))
            ax2.imshow(transforms.ToPILImage()(val_mask))
            ax3.imshow(transforms.ToPILImage()(pred.squeeze(0)))
            fig.savefig(os.path.join(IMG_SAVE_DIR, 'result.png'))
            plt.close(fig)
            unet.train()
        
        if step % 100 == 0:
            torch.save(unet.state_dict(), os.path.join(MODEL_SAVE_DIR, 'unet.pth'))
        step += 1

        print(f'EPOCH {epoch}: BCE LOSS: {loss}')
        writer.add_scalar('Loss / Epoch', loss, epoch)
