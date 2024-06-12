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
# Training Results
- 3 epochs take ~6 hrs
- After epoch 1 - val Dice score : ```0.9727```, val loss = ```0.187```, val acc = ```98.99```
- After epoch 2 - val Dice score : ```0.9820```, val loss = ```0.139```, val acc = ```99.35```
- After epoch 3 - val Dice score : ```0.9811```, val loss = ```0.106```, val acc = ```99.29```
- We can see that just after 3 epochs, we get some really good results, check out the ```saved_images``` folder to compare ground truth and predicted masks.
