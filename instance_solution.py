# %% [markdown]
# # Exercise 05: Instance Segmentation
#
# <hr style="height:2px;">
#
# In this notebook, we adapt our 2D U-Net for better nuclei segmentations in the Kaggle Nuclei dataset.
#
# Written by Valentyna Zinchenko, Constantin Pape and William Patton.

# %% [markdown]
# <div class="alert alert-danger">
# Please use kernel <code>05-semantic-segmentation</code> for this exercise.
# </div>

# %% [markdown]
# <hr style="height:2px;">
#
# ## Import Packages
#%%
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import datetime

from glob import glob
from natsort import natsorted
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

#from torch.utils.tensorboard import SummaryWriter
#from torchsummary import summary
from tqdm import tqdm

# %%
# make sure gpu is available. Please call a TA if this cell fails
assert torch.cuda.is_available()

# %% [markdown]
# ## Visualize the data:

# %% [markdown]
# <div class="alert alert-block alert-success">
# <h2> Checkpoint 0 </h2>

# %% [markdown]
# <hr style="height:2px;">
#
# ## Section 1: Signed Distance Transform

# %% [markdown]
# Create the  DataSet

# %% tags=["solution"]
# write a function to calculate SDT
from scipy.ndimage import distance_transform_edt

def compute_sdt(labels: np.ndarray, constant: float = 0.5, scale: int = 5):
    """Function to compute a signed distance transform."""

    inner = distance_transform_edt(binary_erosion(labels))
    outer = distance_transform_edt(np.logical_not(labels))

    distance = (inner - outer) + constant

    distance = np.tanh(distance / scale)

    return distance

#%% [markdown]
# take the dataset from local.py and add create SDT target function

#%% tags [solution]
# take the dataset from local.py and add create SDT target function
class InstanceDataset(Dataset):
    """A PyTorch dataset to load cell images and nuclei masks"""

    def __init__(self, root_dir, transform=None, img_transform=None):
        self.root_dir = root_dir  # the directory with all the training samples
        self.samples = os.listdir(root_dir)  # list the samples
        self.transform = (
            transform  # transformations to apply to both inputs and targets
        )
        self.img_transform = img_transform  # transformations to apply to raw image only
        #  transformations to apply just to inputs
        self.inp_transforms = transforms.Compose(
            [
                transforms.Grayscale(),  # some of the images are RGB
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]), # 0.5 = mean and 0.5 = variance 
            ]
        )

    # get the total number of samples
    def __len__(self):
        return len(self.samples)

    # fetch the training sample given its index
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.samples[idx], "image.tif")
        # we'll be using Pillow library for reading files
        # since many torchvision transforms operate on PIL images
        image = Image.open(img_path)
        image = self.inp_transforms(image)
        mask_path = os.path.join(self.root_dir, self.samples[idx], "mask.tif")
        mask = transforms.ToTensor()(Image.open(mask_path))
        if self.transform is not None:
            # Note: using seeds to ensure the same random transform is applied to
            # the image and mask
            seed = torch.seed()
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask = self.transform(mask)
        if self.img_transform is not None:
            image = self.img_transform(image)
        return image, mask
    
    def create_sdt_target(self, mask):
        # TODO: fill in with SDT function

#%%
# Adjust the Unet to this new prediction
# Which loss function should we use
# train the model

from unet import UNet

train_data = InstanceDataset("nuclei_train_data", transforms.RandomCrop(256))
train_loader = DataLoader(train_data, batch_size=5, shuffle=True)
val_data = InstanceDataset("nuclei_val_data", transforms.RandomCrop(256))
val_loader = DataLoader(val_data, batch_size=5)

unet = UNet(...)
loss = 
optimizer = torch.optim.Adam(unet.parameters())

for epoch in range(10):
    train(
        unet,
        train_loader,
        optimizer,
        loss,
        epoch,
        device="cuda",
    )

#%% 
# take a look at the results

# %% [markdown]
# <div class="alert alert-block alert-success">
# <h2> Checkpoint 0 </h2>
# %% [markdown]
# <hr style="height:2px;">
#
# ## Section 2: Post-Processing
# What is watershed?
# How do we get seed points?

# %% [markdown]
# <hr style="height:2px;">
#
# ## Section 3: Evaluation
# Which evaluation metric should we use
# Use this website to pick a good one

# %% [markdown]
# <hr style="height:2px;">
#
# ## Section 4: Affinities
# add create affinities method to the dataset
# make needed changes to the model
# train model
# post-processing
# evaluation metric

# %% [markdown]
# <hr style="height:2px;">
#
# ## Section 5: Pre-Trained Models
# try running cellpose from script

