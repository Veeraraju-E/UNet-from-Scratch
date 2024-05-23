# UNet-from-Scratch
- this repo contains the implementation of UNET architecture from scratch for the task of Image Segmentation on the Carvana Dataset
- you can find the dataset at https://www.kaggle.com/c/carvana-image-masking-challenge/data

# Data Structure
- ensure that the data is in the following format - 
```bash
├───data
│   ├───train_images
│   ├───train_masks
│   ├───val_images
│   └───val_masks
├───dataset.py
├───model.py
├───train.py
├───utils.py
└───saved_images
```
- note that I have only a small proportion of my train images and masks in my val images and masks - 48 of them, to be precise. 
