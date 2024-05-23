import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace('.jpg', '_mask.gif'))

        image = np.array(Image.open(img_path).convert('RGB'))   # just ensuring RGB
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)     # as masks are greyscale
        # convert to numpy array as we are going to use albumentations, so after reading the image using PIL, convert it
        # similarly for the greyscale masks to be readable

        mask[mask == 255.0] = 1.0   # for sigmoid type of activation that we use in model
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)  # this defines the augmentations
            image = augmentations['image']  # get the augmented image
            mask = augmentations['mask']    # get the augmented mask

        return image, mask
