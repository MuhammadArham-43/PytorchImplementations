import torch
from model.unet import UNet

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import os

if __name__ == "__main__":
    model = UNet(3,1)
    model.load_state_dict(torch.load('results/models/unet.pth'))
    meta_data = 'data/ForestData/Forest Segmented/meta_data.csv'
    images_dir = 'data/ForestData/Forest Segmented'
    meta_data = pd.read_csv(meta_data)
    row = meta_data.iloc[1267]
    image_path = os.path.join(images_dir, 'images', row['image'])
    mask_path = os.path.join(images_dir, 'masks', row['mask'])
    img_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    image = img_transforms(image)
    
    model.eval()
    
    prediction = model(image.unsqueeze(0))
    print(prediction.shape)
    fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    ax1.imshow(transforms.ToPILImage()(image))
    ax2.imshow(mask)
    ax3.imshow(transforms.ToPILImage()(torch.sigmoid((prediction.squeeze(1).detach().cpu()) > 0.5).float()))
    # fig.show()
    fig.savefig('infer_result.png')
    print('result saved')
    
        