import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from model import UNet
from utils import load_checkpoint, save_checkpoint, get_loaders, check_accuracy, save_predictions_as_imgs

# Hyperparameters
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  # 1280 in the original dataset
IMAGE_WIDTH = 240   # 1918 in the original dataset
PIN_MEMORY = True   # <GPU related>
LOAD_MODEL = False
TRAIN_IMG_DIR = 'data/train_images/'
TRAIN_MASK_DIR = 'data/train_masks/'
VAL_IMG_DIR = 'data/val_images/'
VAL_MASK_DIR = 'data/val_masks/'


def train_fn(loader, model, optimizer, loss_fn):
    """
    main training loop
    """
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        # print('batch_idx', batch_idx)
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)   # don't forget to un-squeeze along the channels dimension

        # forward
        predictions = model(data)
        loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # Adam step
        optimizer.step()

        loop.set_postfix(loss=loss.item())


def main():
    """
    driver function for training, saving ckpts, and saving some images
    define the transformations and other parameters to generalize the dataloaders for training
    """
    train_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ],
    )
    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ],
    )
    model = UNet(3, 1).to(DEVICE)   # change out_channels for multi-class classification
    loss_fn = nn.BCEWithLogitsLoss()    # we are not doing a sigmoid on our output
    # if we had done return torch.sigmoid(self.conv(x)) in our forward method of UNet Class, then no need of WithLogits
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transforms,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY, # <GPU related>
    )

    # simple flag to check if we have to load the prev model or not, useful for future purposes
    if LOAD_MODEL:
        load_checkpoint(torch.load('my_checkpoint.pth.tar'), model)

    for epoch in range(NUM_EPOCHS):
        # print('epoch', epoch)
        if epoch == 0:
            print('Training started')

        train_fn(train_loader, model, optimizer, loss_fn)

        # Save the model - the 2 most important things we need while saving a model
        # are the tensors ("parameters") of the model & optimizers; this is represented by state_dict
        # try printing it to understand it better
        # print(model.state_dict())
        checkpoint = {'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      }
        save_checkpoint(checkpoint) # check utils.py file to understand how saving is being done

        # Check Accuracy
        check_accuracy(val_loader, model, DEVICE)   
        # check utils.py file to understand how accuracy checking is being done

        # Save some sample Images
        save_predictions_as_imgs(val_loader, model, folder='saved_images/')
        # Once the training is done for the first time, you should see a new folder called 'saved_images', 
        # with images like pred_0.png etc, which contain the masks


if __name__ == '__main__':
    main()
