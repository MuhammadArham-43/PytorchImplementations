from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np


class ForestData(Dataset):
    def __init__(
        self,
        meta_data_file: str,
        images_dir_path: str,
        masks_dir_path: str,
        transforms=None
    ) -> None:
        super(ForestData, self).__init__()
        self.meta_data = pd.read_csv(meta_data_file)
        self.images_dir = images_dir_path
        self.masks_dir = masks_dir_path
        self.transforms = transforms
        # print(self.meta_data.iloc[0]['image'], self.meta_data.iloc[0]['mask'])

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, index):
        row = self.meta_data.iloc[index]
        image_path = row['image']
        mask_path = row['mask']
        image = Image.open(os.path.join(
            self.images_dir, image_path)).convert('RGB')
        mask = Image.open(os.path.join(self.masks_dir, mask_path)).convert('L')

        image = np.array(image)
        mask = np.array(mask)
        # mask[mask == 255.0] = 1.0
        if self.transforms is not None:
            augmentations = self.transforms(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        mask = mask / 255.0
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0
        return image, mask


if __name__ == "__main__":
    meta_file_path = '/media/muhammad_arham/F/BlogTest/PytorchTutorials/ImageSegmentation/data/ForestData/Forest Segmented/meta_data.csv'
    images_dir = '/media/muhammad_arham/F/BlogTest/PytorchTutorials/ImageSegmentation/data/ForestData/Forest Segmented/images'
    masks_dir = '/media/muhammad_arham/F/BlogTest/PytorchTutorials/ImageSegmentation/data/ForestData/Forest Segmented/masks'
    image_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    data = ForestData(
        meta_file_path,
        images_dir,
        masks_dir,
        transforms=image_transforms
    )

    image, mask = data.__getitem__(10)
    print(image.size())
    print(mask.size())
    print(mask)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    print(mask)
    ax1.imshow(transforms.ToPILImage()(image))
    ax2.imshow(transforms.ToPILImage()(mask))
    plt.show()
