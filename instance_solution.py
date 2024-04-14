# %% [markdown]
# # Exercise 05: Instance Segmentation
#
# <hr style="height:2px;">
#
# In this notebook, we adapt our 2D U-Net for better nuclei segmentations in the Kaggle Nuclei dataset.
#

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
from PIL import Image

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
# For instance segmentation, the specific value of the label is arbitrary
# This means we cannot train directly on those labels
# image table of different labels --> say all are equivalently good


# %% [markdown]
# <hr style="height:2px;">
#
# ## Section 1: Signed Distance Transform
# what is the signed distance transform, with image examples

# %% [markdown]
# Create the  DataSet

# %% tags=["solution"]
# write a function to calculate SDT
from scipy.ndimage import distance_transform_edt, binary_erosion

def compute_sdt(labels: np.ndarray, constant: float = 0.5, scale: int = 5):
    """Function to compute a signed distance transform."""

    inner = distance_transform_edt(binary_erosion(labels))
    outer = distance_transform_edt(np.logical_not(labels))

    distance = (inner - outer) + constant

    distance = np.tanh(distance / scale)

    return distance

#%%
# small box to visualize the signed distance transform
from local import NucleiDataset, show_random_dataset_image
train_data = NucleiDataset("nuclei_train_data", transforms.RandomCrop(256))

idx = np.random.randint(0, len(train_data))  # take a random sample
img, mask = train_data[idx]  # get the image and the nuclei masks

f, axarr = plt.subplots(1, 2)  # make two plots on one figure
axarr[0].imshow(img[0])  # show the image
axarr[1].imshow(compute_sdt(mask[0]), interpolation=None)  # show the masks
_ = [ax.axis("off") for ax in axarr]  # remove the axes
print("Image size is %s" % {img[0].shape})
plt.show()

#%% tags [markdown]
# Questions
# 1. What is the purpose of the tanh function in computing our signed distance transform?
# 2. What is the effect of changing the scale value? what do you think is a good default value? why?


#%% tags [solution]
# take the dataset from local.py and add create SDT target function
# 1. add a create_sdt_target function to this class
# 2. Change the __get_item__ method to return the sdt output rather than the mask
#   2a. think about how transformations will affect the SDT vs mask
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
            sdt_mask = self.create_sdt_target(mask)
        if self.img_transform is not None:
            image = self.img_transform(image)
        return image, sdt_mask
    
    def create_sdt_target(self, mask):
        sdt_target = compute_sdt(mask, constant=0.5, scale=5)
        return sdt_target

#%%
# Adjust the Unet to this new prediction
# Which loss function should we use
# train the model
from local import train, show_random_dataset_image

from unet import UNet

train_data = InstanceDataset("nuclei_train_data", transforms.RandomCrop(256))
train_loader = DataLoader(train_data, batch_size=5, shuffle=True)
val_data = InstanceDataset("nuclei_val_data", transforms.RandomCrop(256))
val_loader = DataLoader(val_data, batch_size=5)

show_random_dataset_image(train_data)

#%%
# add the augmentations that you want
unet = UNet()

# choose a loss function (here are a few options to consider)
loss = torch.nn.BCELoss()

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
# evaluate model and plot resulting distance transform

# This looks good, but it isn't quite what we want yet

# %% [markdown]
# <div class="alert alert-block alert-success">
# <h2> Checkpoint 1 </h2>
# %% [markdown]
# <hr style="height:2px;">
#
# ## Section 2: Post-Processing
# What is watershed?
# How do we get seed points?

#%%
from scipy.ndimage import label_cc, maximum_filter
from scipy.segmentation import watershed

def watershed_from_boundary_distance(
        boundary_distances: np.ndarray,
        boundary_mask: np.ndarray,
        id_offset: float = 0,
        min_seed_distance: int = 10
        ):
    """Function to compute a watershed from boundary distances."""

    # get our seeds 
    # make them write a function to find maximum values?
    max_filtered = maximum_filter(boundary_distances, min_seed_distance)
    maxima = max_filtered==boundary_distances
    seeds, n = label_cc(maxima)

    if n == 0:
        return np.zeros(boundary_distances.shape, dtype=np.uint64), id_offset

    seeds[seeds!=0] += id_offset

    # calculate our segmentation
    segmentation = watershed(
        boundary_distances.max() - boundary_distances,
        seeds,
        mask=boundary_mask)
    
    return segmentation

def get_boundary_mask(pred, threshold):
    boundary_mask = pred > threshold
    return boundary_mask
#%%
net.eval()
for idx, (image, mask) in enumerate(val_loader):
    image = image.to(device)
    logits = net(image)
    pred = activation(logits)
        
    image = np.squeeze(image.cpu())
    mask = np.squeeze(mask.cpu().numpy())
        
    pred = np.squeeze(pred.cpu().detach().numpy())
    
    # feel free to try different thresholds
    thresh = np.mean(pred)
    
    # get boundary mask
    boundary_mask = get_boundary_mask(pred, thresh=thresh)

    seg = watershed_from_boundary_distance(
        pred,
        boundary_mask,
        id_offset=0
        min_seed_distance=10)


# %% [markdown]
# <hr style="height:2px;">
#
# ## Section 3: Evaluation
# Which evaluation metric should we use
# Use this website to pick a good one

https://metrics-reloaded.dkfz.de/problem-category-selection

# Which metric do you think is best for this dataset

# 3 -5 example problems with images, want them to choose which metric is best for each

# 1. cells that are mostly round, but have long protrusions, where our goal is to quantify the number of protrusions

# 2. mixed population of 2 cells, where our goal is to segment 
# 1 population form the other and count the number of cells of that population

# 3. Cells with many small specs where the goal is to segment and count the number of specs


# %% [markdown]
# <hr style="height:2px;">
#
# ## Section 4: Affinities
#%%
# add create affinities method to the dataset
def compute_affinities(seg: np.ndarray, nhood: list):

    nhood = np.array(nhood)

    shape = seg.shape
    nEdge = nhood.shape[0]
    dims = nhood.shape[1]
    aff = np.zeros((nEdge,) + shape, dtype=np.int32)

    for e in range(nEdge):
        aff[e, \
          max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
          max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1])] = \
                      (seg[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                          max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1])] == \
                        seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                          max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1])] ) \
                      * ( seg[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                          max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1])] > 0 ) \
                      * ( seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                          max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1])] > 0 )
                          

    return aff

# make needed changes to the model
# train model
# post-processing
# evaluation metric

# %% [markdown]
# <hr style="height:2px;">
#
# ## Section 5: Pre-Trained Models
# try running cellpose from script

#%%
import cellpose


