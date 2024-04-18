# %% [markdown]
# # Exercise 05: Instance Segmentation
#
# <hr style="height:2px;">
#
# - So far, we were only interested in `semantic` classes, e.g. foreground / background, cell types, person / car, etc. 
# But in many cases we not only want to know if a certain pixel belongs to an object, but also to which unique object (i.e. the task of `instance segmentation`).
#
# - For isolated objects, this is trivial, all connected foreground pixels form one instance, yet often instances are very close together or even overlapping. Thus we need to think a bit more how to formulate the targets / loss of our network.
#
# - Further, in instance segmentation the specific value of each label is arbitrary. Here, `Mask 1` and `Mask 2` below would be equivalent even though the values of pixels on individual cells is different.
#
# | Image | Mask 1| Mask 2|
# | :-: | :-: | :-: |
# | ![image](static/01_instance_img.png) | ![mask1](static/02_instance_teaser.png) | ![mask2](static/03_instance_teaser.png) |
#
#
# - Therefore we must devise an auxilliary task to train the model on, which we then can post process into our final instance segmentation.

# %% [markdown]
# <hr style="height:2px;">
#
# ## Import Packages
# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import datetime
from PIL import Image

from glob import glob
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

device = 'cpu'

#from torch.utils.tensorboard import SummaryWriter
#from torchsummary import summary
from tqdm import tqdm

# %%
# make sure gpu is available. Please call a TA if this cell fails
assert torch.cuda.is_available()

# %% [markdown]
# <hr style="height:2px;">
#
# ## Section 1: Signed Distance Transform (SDT)
# What is the signed distance transform?

# %% [markdown]
# ![image](static/04_instance_sdt.png)
#

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 1.1</b>: Write a function to compute the signed distance transform
# </div>
# %%
from scipy.ndimage import distance_transform_edt, binary_erosion

def compute_sdt(labels: np.ndarray, constant: float = ..., scale: int = ...):
    """Function to compute a signed distance transform."""

    # compute the distance transform inside and outside of the objects

    # create the signed distance transform 

    # scale the distances so that they are between -1 and 1 (hint: np.tanh)

    # be sure to return your solution as type 'float'
    return 


# %% tags=["solution"]
# write a function to calculate SDT
# maybe change to a forloop that iterates over all labels, would work for both instance and semantic labels
from scipy.ndimage import distance_transform_edt, binary_erosion

def compute_sdt(labels: np.ndarray, constant: float = 0.5, scale: int = 5):
    """Function to compute a signed distance transform."""

    # compute the distance transform inside and outside of the objects
    inner = distance_transform_edt(binary_erosion(labels))
    outer = distance_transform_edt(np.logical_not(labels))

    # create the signed distance transform 
    distance = (inner - outer) + constant

    # scale the distances so that they are between -1 and 1 (hint: np.tanh)
    distance = np.tanh(distance / scale)

    # be sure to return your solution as type 'float'
    return distance.astype(float)

# %% [markdown]
# Below is a small function to visualize the signed distance transform (SDT), use it to validate your function.
#
# Make sure that your SDT output is what you expect
# %%
# visualize the signed distance transform
from local import NucleiDataset
# import an example dataset
train_data = NucleiDataset("nuclei_train_data", transforms.RandomCrop(256))

idx = np.random.randint(0, len(train_data))  # take a random sample
img, mask = train_data[idx]  # get the image and the nuclei masks

f, axarr = plt.subplots(1, 2)  # make two plots on one figure
axarr[0].imshow(img[0])  # show the image
axarr[1].imshow(compute_sdt(mask[0]), interpolation=None)  # show the masks
_ = [ax.axis("off") for ax in axarr]  # remove the axes
print("Image size is %s" % {img[0].shape})
plt.show()

# %% tags [markdown]
# Questions
# 1. Why do we need to normalize the distances between -1 and 1?
#   -
# 2. What is the effect of changing the scale value? What do you think is a good default value?
#   -

# %% [markdown] tags=["solution"]
# Questions
# 1. Why do we need to normalize the distances between -1 and 1?
#   -
# 2. What is the effect of changing the scale value? What do you think is a good default value?
#   -

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 1.2</b>: Modify the dataset to produce the paired raw and SDT images
#   1. use the compute_sdt function to fill in the create_sdt_target method
#   2. modify the __get_item__ method to return an SDT output rather than a mask
#       - be sure that all final outputs in a torch tensor
#       - Think about the order transformations are applied to the mask/SDT
# </div>

# %%
class InstanceDataset(Dataset):
    """A PyTorch dataset to load cell images and nuclei masks"""

    def __init__(self, root_dir, transform=None, img_transform=None, return_mask=False):
        self.root_dir = root_dir  # the directory with all the training samples
        self.samples = os.listdir(root_dir)  # list the samples
        self.return_mask = return_mask
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

        # Modify this function to return an image and sdt pair

        img_path = os.path.join(self.root_dir, self.samples[idx], "image.tif")
        # we'll be using Pillow library for reading files
        # since many torchvision transforms operate on PIL images
        image = Image.open(img_path)
        image = self.inp_transforms(image)
        mask_path = os.path.join(self.root_dir, self.samples[idx], "label.tif")
        mask = Image.open(mask_path)
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
        if self.return_mask is True:
            return image, mask, sdt
        else:
            return image, sdt
        
    
    def create_sdt_target(...):

       # fill in function

        return 
# %% tags=["solution"]
class InstanceDataset(Dataset):
    """A PyTorch dataset to load cell images and nuclei masks"""

    def __init__(self, root_dir, transform=None, img_transform=None, return_mask=False):
        self.root_dir = root_dir  # the directory with all the training samples
        self.samples = os.listdir(root_dir)  # list the samples
        self.return_mask = return_mask
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
        mask_path = os.path.join(self.root_dir, self.samples[idx], "label.tif")
        mask = Image.open(mask_path)
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
        if self.return_mask is True:
            return image, transforms.ToTensor()(mask), sdt_mask
        else:
            return image, sdt_mask
    
    def create_sdt_target(self, mask):
        sdt_target_array = compute_sdt(mask, constant=0.5, scale=5)
        sdt_target = transforms.ToTensor()(sdt_target_array)
        return sdt_target.float()

# %% [markdown]
# ### Test your function
# - Create training and validation datasets and data loaders
# - use `show_random_dataset_image` to verify that your dataset solution is correct
#   - output should show 2 images: the raw image and the SDT
# %%
from local import show_random_dataset_image

train_data = InstanceDataset("nuclei_train_data", transforms.RandomCrop(256))
train_loader = DataLoader(train_data, batch_size=5, shuffle=True)
val_data = InstanceDataset("nuclei_val_data", transforms.RandomCrop(256))
val_loader = DataLoader(val_data, batch_size=5)

show_random_dataset_image(train_data)

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 1.3</b>: Train the Unet
# </div>
# %% [markdown]
# Hints:
#   - loss function - [torch losses](https://pytorch.org/docs/stable/nn.html#loss-functions)
#   - optimizer - [torch optimizers](https://pytorch.org/docs/stable/optim.html)
#   - final_activation - there are a few options (only one is the best)
#       - [sigmoid](https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html)
#       - [tanh](https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html#torch.nn.Tanh)
#       - [relu](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU)

# %%
from local import train
import torch
from unet import UNet

# initialize the model
unet = UNet(
    depth=,
    in_channels=,
    num_fmaps=,
    fmap_inc_factor=,
    downsample_factor=,
    kernel_size=,
    padding=,
    upsample_mode=,
    final_activation=, 
    out_channels=,
)

# choose a loss function

# choose an optimizer

# write a training loop and train for 10 epochs

# %% tags=["solution"]
from local import train
from unet import UNet


unet = UNet(

    depth=3,
    in_channels=1,
    out_channels=1, 
    final_activation='Tanh',
    num_fmaps = 64,
    fmap_inc_factor= 2,
    downsample_factor= 2,
    kernel_size= 3,
    padding= "same",
    upsample_mode= "nearest",
)

loss = torch.nn.MSELoss()

optimizer = torch.optim.Adam(unet.parameters())

for epoch in range(5):
    train(
        unet,
        train_loader,
        optimizer,
        loss,
        epoch,
        log_interval=1,
        device="cpu", # make sure to set to cuda
    )

# %% [markdown]
# Next run inference using our trained model and visualize some random samples

# %%
unet.eval()
idx = np.random.randint(0, len(val_data))  # take a random sample
image, mask = val_data[idx]  # get the image and the nuclei masks

image = image.to(device)
pred = unet(torch.unsqueeze(image, dim=0))
    
image = np.squeeze(image.cpu())
mask = np.squeeze(mask.cpu().numpy())
    
pred = np.squeeze(pred.cpu().detach().numpy())

f, axarr = plt.subplots(1, 3)  # make two plots on one figure
axarr[0].imshow(image)  # show the image
axarr[1].imshow((mask), interpolation=None)  # show the masks
axarr[2].imshow(pred) # show the prediction
_ = [ax.axis("off") for ax in axarr]  # remove the axes
plt.show()

# %% [markdown]
# <div class="alert alert-block alert-success">
# <h2> Checkpoint 1 </h2>
#
# At this point we have a model that does what we told it too, but not yet a segmentation.
#
# In the next section we will perform some post-processing and get our segmentations.
# %% [markdown]
# <hr style="height:2px;">
#
# ## Section 2: Post-Processing
# - See here for a nice overview: [scikit-image watershed](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html)
# - Given the distance transform (the output of our model), we first need to find the local maxima that will be used as seed points
# - The watershed algorithm then expands each seed out in a local "basin" until the segments touch.

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 2.1</b>: write a function to find the local maxima of the distance transform
# </div>

# %% [markdown]
# hint: look at the imports
# hint: it is possible to write this function by only adding 2 lines
# %%
from scipy.ndimage import label, maximum_filter

def find_local_maxima(distance_transform, min_seed_distance):

    # fill in function here...
    
    # uniquely label the local maxima 
    seeds, n = label(...)

    return seeds, n
# %% tags=["solution"]
from scipy.ndimage import label, maximum_filter
def find_local_maxima(distance_transform, min_seed_distance):
    max_filtered = maximum_filter(distance_transform, min_seed_distance)
    maxima = max_filtered==distance_transform
    seeds, n = label(maxima)

    return seeds, n

# %% [markdown]
# We now use this function to find the seeds for the watershed
# %%
from skimage.segmentation import watershed

def watershed_from_boundary_distance(
        boundary_distances: np.ndarray,
        boundary_mask: np.ndarray,
        id_offset: float = 0,
        min_seed_distance: int = 10
        ):
    """Function to compute a watershed from boundary distances."""

    
    seeds, n = find_local_maxima(boundary_distances, min_seed_distance)

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

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 2.2</b>: get the watershed segmentation
# </div>

# %%
idx = np.random.randint(0, len(val_data))  # take a random sample
image, mask = val_data[idx]  # get the image and the nuclei masks

# hint: look at how we got predictions in the inference box above
# hint: make sure set the model to evaluation


# chose a threshold value to use to get the boundary mask
thresh = ...

# get boundary mask
boundary_mask = ...

# get the segmentation
seg = watershed_from_boundary_distance(...)

# plot the results
f, axarr = plt.subplots(1, 3)  # make two plots on one figure
axarr[0].imshow(image)  # show the image
axarr[1].imshow((mask), interpolation=None)  # show the masks
axarr[2].imshow((seg))
_ = [ax.axis("off") for ax in axarr]  # remove the axes
plt.show()

# %% tags=["solution"]
# If we notice this exercise is running long, we can just give them this part

unet.eval()
idx = np.random.randint(0, len(val_data))  # take a random sample
image, mask = val_data[idx]  # get the image and the nuclei masks

image = image.to(device)
pred = unet(torch.unsqueeze(image, dim=0))
    
image = np.squeeze(image.cpu())
mask = np.squeeze(mask.cpu().numpy())
    
pred = np.squeeze(pred.cpu().detach().numpy())

# feel free to try different thresholds
thresh = np.mean(pred)

# get boundary mask
boundary_mask = get_boundary_mask(pred, threshold=thresh)

seg = watershed_from_boundary_distance(
    pred,
    boundary_mask,
    id_offset=0,
    min_seed_distance=5)

f, axarr = plt.subplots(1, 3)  # make two plots on one figure
axarr[0].imshow(image)  # show the image
axarr[1].imshow((mask), interpolation=None)  # show the masks
axarr[2].imshow((seg))
_ = [ax.axis("off") for ax in axarr]  # remove the axes
plt.show()

# %% [markdown]
# Questions:
# 1. what is the effect of the min_seed_distance parameter in watershed? 
#   - experiment with different values

# %% [markdown]
# <div class="alert alert-block alert-success">
# <h2> Checkpoint 2 </h2>

# %% [markdown]
# <hr style="height:2px;">
#
# ## Section 3: Evaluation
# Many different evaluation metrics exist, and which one you should use is dependant on the specifics of the data.

# [This website](https://metrics-reloaded.dkfz.de/problem-category-selection) has a good summary of different options.

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 3.1</b>: Pick the best metric to use
# </div>
# %% [markdown]
# Which of the following should we use for our dataset?:
#   1) [Dice](https://metrics-reloaded.dkfz.de/metric?id=dsc)
#   2) [Average Precision](https://metrics-reloaded.dkfz.de/metric?id=average_precision)
#   3) [Sensitivity](https://metrics-reloaded.dkfz.de/metric?id=sensitivity) and [Specificity](https://metrics-reloaded.dkfz.de/metric?id=specificity@target_value)
# 

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 3.2</b>: evaluate for the validation dataset
# </div>
# %%
from local import evaluate
# need to re-initialize the dataloader to return masks in addition to SDTs
val_data = InstanceDataset("nuclei_val_data", transforms.RandomCrop(256), return_mask=True)
val_loader = DataLoader(val_data, batch_size=5)

unet.eval()

for idx, (image, mask, sdt) in enumerate(val_loader):
    image = image.to(device)
    pred = unet(image)
        
    image = np.squeeze(image.cpu())
    gt_labels = np.squeeze(mask.cpu().numpy())
        
    pred = np.squeeze(pred.cpu().detach().numpy())

    # feel free to try different thresholds
    thresh = np.mean(pred)

    # get boundary mask
    boundary_mask = get_boundary_mask(pred, threshold=thresh)

    pred_labels = watershed_from_boundary_distance(
        pred,
        boundary_mask,
        id_offset=0,
        min_seed_distance=5)

    ap, precision, recall, tp, fp, fn = evaluate(gt_labels, pred_labels)
# %% [markdown]
# <hr style="height:2px;">
#
# ## Section 4: Affinities
# %% [markdown]
# What are affinities?
#Here we consider not just the pixel but also its direct neighbors (in 2D the left neighbor and the upper neighbor are sufficient, 
# right and down are redundant with the next pixel's left and upper neighbor). Imagine there is an edge between two pixels if they 
# are in the same class and no edge if not. If we then take all pixels that are directly and indirectly connected by edges, we get an 
# instance. Essentially, we label edges between neighboring pixels as “connected” or “cut”, rather than labeling the pixels themselves. 
# This representation can be useful especially in the case of smaller objects that would otherwise be classified as background pixels. 
# We can use mean squared error.

# %% [markdown]
# ![image](static/05_instance_affinity.png)

# %% [markdown]
# Similar to the pipeline used for SDTs, we first need to modify the dataset to produce affinities.

#%%
# create a new dataset for affinities
from local import compute_affinities
class affinityDataset(Dataset):
    """A PyTorch dataset to load cell images and nuclei masks"""

    def __init__(self, root_dir, transform=None, img_transform=None, return_mask=False):
        self.root_dir = root_dir  # the directory with all the training samples
        self.samples = os.listdir(root_dir)  # list the samples
        self.return_mask = return_mask
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
        mask_path = os.path.join(self.root_dir, self.samples[idx], "label.tif")
        mask = Image.open(mask_path)
        if self.transform is not None:
            # Note: using seeds to ensure the same random transform is applied to
            # the image and mask
            seed = torch.seed()
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask = self.transform(mask)
            aff_mask = self.create_aff_target(mask)
        if self.img_transform is not None:
            image = self.img_transform(image)
        if self.return_mask is True:
            return image, transforms.ToTensor()(mask), aff_mask
        else:
            return image, aff_mask
    
    def create_aff_target(self, mask):
        aff_target_array = compute_affinities(np.asarray(mask), [[0,1],[1,0]])
        aff_target = torch.from_numpy(aff_target_array)
        return aff_target.float()

# %% [markdown]
# next we initialize the datasets and data loaders
# %%
# Initialize the datasets
from local import train
from unet import UNet

train_data = affinityDataset("nuclei_train_data", transforms.RandomCrop(256))
train_loader = DataLoader(train_data, batch_size=5, shuffle=True)
val_data = affinityDataset("nuclei_val_data", transforms.RandomCrop(256))
val_loader = DataLoader(val_data, batch_size=5)

show_random_dataset_image(train_data)

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 4.1</b>: Train a model on affinities
# </div>
# %%
# repurpose your training loop for the SDTs

# Think carefully about your loss and final activation, 
# the best for SDT is no necessarily the best for affinities


#%% tags=["solutions"]
unet = UNet(
    depth=3,
    in_channels=1,
    num_fmaps=64,
    fmap_inc_factor=3,
    downsample_factor=([2,2]),
    kernel_size=3,
    padding="same",
    upsample_mode="nearest",
    final_activation=torch.nn.Sigmoid(), # different from SDTs
    out_channels=2,
)

# choose a loss function 
loss = torch.nn.MSELoss()

optimizer = torch.optim.Adam(unet.parameters())

for epoch in range(5):
    train(
        unet,
        train_loader,
        optimizer,
        loss,
        epoch,
        log_interval=1,
        device="cpu", # make sure to set to cuda
    )
# %% [markdown]
# <hr style="height:2px;">
#
# ## Section 5: Pre-Trained Models
# try running cellpose from script

# %%
import cellpose


