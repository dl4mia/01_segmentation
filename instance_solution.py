# %% [markdown]
# # Exercise 05: Instance Segmentation
#
# So far, we were only interested in `semantic` classes, e.g. foreground / background etc.
# But in many cases we not only want to know if a certain pixel belongs to a specific class, but also to which unique object (i.e. the task of `instance segmentation`).
#
# For isolated objects, this is trivial, all connected foreground pixels form one instance, yet often instances are very close together or even overlapping. Thus we need to think a bit more how to formulate the targets / loss of our network.
#
# Furthermore, in instance segmentation the specific value of each label is arbitrary. Here, `Mask 1` and `Mask 2` below would be equivalent even though the values of pixels on individual cells is different.
#
# | Image | Mask 1| Mask 2|
# | :-: | :-: | :-: |
# | ![image](static/01_instance_img.png) | ![mask1](static/02_instance_teaser.png) | ![mask2](static/03_instance_teaser.png) |
#
# Therefore, we must devise an auxilliary task to train the model on, which we then can post process into our final instance segmentation.

# %% [markdown]
# ## Import Packages
# %%
import matplotlib.pyplot as plt
from matplotlib import gridspec, ticker
from matplotlib.colors import ListedColormap
import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.ndimage import distance_transform_edt, binary_erosion
from local import train, NucleiDataset
from unet import UNet
from local import show_random_dataset_image
from tqdm import tqdm
import urllib.request
from skimage.filters import threshold_otsu


# %%
device = "mps"  # 'cuda', 'cpu', 'mps'
# make sure gpu is available. Please call a TA if this cell fails
# assert torch.cuda.is_available()

# %%
# Download a custom label color map
urllib.request.urlretrieve(
    "https://tinyurl.com/labelcmap",
    "cmap_60.npy",
)
label_cmap = ListedColormap(np.load("cmap_60.npy"))

# %% [markdown]
# ## Section 1: Signed Distance Transform (SDT)
# <i>What is the signed distance transform?</i> <br> Signed Distance Transform indicates the distance to the boundary of objects. <br> It should be positive for pixels inside objects and negative for pixels outside objects (i.e. in the background).<br> As an example, here, you see the SDT (right) of the target mask (middle), below.

# %% [markdown]
# ![image](static/04_instance_sdt.png)
#

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 1.1</b>: Write a function to compute the signed distance transform.
# </div>
# %%
# Write a function to calculate SDT.
# (Hint: Use `distance_transform_edt` and `binary_erosion` which are imported in the first cell.)


def compute_sdt(labels: np.ndarray, constant: float = 0.5, scale: int = 5):
    """Function to compute a signed distance transform."""

    # compute the distance transform inside and outside of the objects

    # create the signed distance transform

    # scale the distances so that they are between -1 and 1 (hint: np.tanh)

    # be sure to return your solution as type 'float'
    return


# %% tags=["solution"]
# Write a function to calculate SDT.
# (Hint: Use `distance_transform_edt` and `binary_erosion` which are imported in the first cell.)


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
# Below is a small function to visualize the signed distance transform (SDT). <br> Use it to validate your function.
# %%
# Visualize the signed distance transform using the function you wrote above.
train_data = NucleiDataset("nuclei_train_data", transforms.RandomCrop(256))
idx = np.random.randint(len(train_data))  # take a random sample
img, mask = train_data[idx]  # get the image and the nuclei masks

fig = plt.figure(constrained_layout=False, figsize=(10, 3))
spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
ax1 = fig.add_subplot(spec[0, 0])
ax1.set_xlabel("Image", fontsize=20)
plt.imshow(img[0], cmap="magma")
ax2 = fig.add_subplot(spec[0, 1])
ax2.set_xlabel("SDT", fontsize=20)
plt.imshow(compute_sdt(mask[0]), cmap="magma")
_ = [ax.set_xticks([]) for ax in [ax1, ax2]]
_ = [ax.set_yticks([]) for ax in [ax1, ax2]]
plt.tight_layout()
plt.show()


# %% tags [markdown]
# <b>Questions</b>:
# 1. Why do we need to normalize the distances between -1 and 1?
#   -
# 2. What is the effect of changing the scale value? What do you think is a good default value?
#   -

# %% [markdown] tags=["solution"]
# <b>Questions</b>:
# 1. Why do we need to normalize the distances between -1 and 1? <br>
#   It allows for better targets for the model and enables better training.<br>
# 2. What is the effect of changing the scale value? What do you think is a good default value?<br>
#   Increasing the scale is equivalent to having a wider boundary region.

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 1.2</b>: <br>
#     Modify the `SDTDataset` class below to produce the paired raw and SDT images.<br>
#   1. Use the `compute_sdt` function to fill in the `create_sdt_target` method.<br>
#   2. Modify the `__get_item__` method to return an SDT output rather than a label mask.<br>
#       - Ensure that all final outputs are of torch tensor type.<br>
#       - Think about the order in which transformations are applied to the mask/SDT.<br>
# </div>


# %%
class SDTDataset(Dataset):
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
                transforms.Normalize([0.5], [0.5]),  # 0.5 = mean and 0.5 = variance
            ]
        )

    # get the total number of samples
    def __len__(self):
        return len(self.samples)

    # fetch the training sample given its index
    def __getitem__(self, idx):

        # Modify this function to return an image and sdt pair

        img_path = os.path.join(self.root_dir, self.samples[idx], "image.tif")
        # we'll be using the Pillow library for reading files
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

    def create_sdt_target(self):
        # TODO: Fill in function

        return


# %% tags=["solution"]
class SDTDataset(Dataset):
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
                transforms.Normalize([0.5], [0.5]),  # 0.5 = mean and 0.5 = variance
            ]
        )

    # get the total number of samples
    def __len__(self):
        return len(self.samples)

    # fetch the training sample given its index
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.samples[idx], "image.tif")
        # We'll be using the Pillow library for reading files
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
#
# Next, we will create a training dataset and data loader.
# We will use `show_random_dataset_image` (imported in the first cell) to verify that our dataset solution is correct. The output would show 2 images: the raw image and the corresponding SDT.
# %%
train_data = SDTDataset("nuclei_train_data", transforms.RandomCrop(256))
train_loader = DataLoader(train_data, batch_size=5, shuffle=True)
val_data = SDTDataset("nuclei_val_data", transforms.RandomCrop(256))
val_loader = DataLoader(val_data, batch_size=5, shuffle=False)
show_random_dataset_image(train_data)

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 1.3</b>: Train the U-Net.
# </div>
# %% [markdown]
# In this task, initialize the UNet.<br>
# <u>Hints</u>:<br>
#   - Loss function - [torch losses](https://pytorch.org/docs/stable/nn.html#loss-functions)
#   - Optimizer - [torch optimizers](https://pytorch.org/docs/stable/optim.html)
#   - Final_activation - there are a few options (only one is the best)
#       - [sigmoid](https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html)
#       - [tanh](https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html#torch.nn.Tanh)
#       - [relu](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU)

# %%
# initialize the model
unet = UNet(
    depth=...,
    in_channels=...,
    num_fmaps=...,
    fmap_inc_factor=...,
    downsample_factor=...,
    padding=...,
    final_activation=...,
    out_channels=...,
)

# Choose a loss function

# Choose an optimizer

# Write a training loop and train for 10 epochs

# %% tags=["solution"]
unet = UNet(
    depth=1,
    in_channels=1,
    out_channels=1,
    final_activation="Tanh",
    num_fmaps=64,
    fmap_inc_factor=3,
    downsample_factor=2,
    padding="same",
    upsample_mode="nearest",
)

loss = torch.nn.MSELoss()
learning_rate = 1e-4
optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)


for epoch in range(10):
    train(
        unet,
        train_loader,
        optimizer,
        loss,
        epoch,
        log_interval=10,
        device=device,
    )

# %% [markdown]
# Next, let's run the inference using our trained model and visualize some random samples.

# %%
unet.eval()
idx = np.random.randint(len(val_data))  # take a random sample
image, mask = val_data[idx]  # get the image and the nuclei masks
image = image.to(device)
pred = unet(torch.unsqueeze(image, dim=0))
image = np.squeeze(image.cpu())
mask = np.squeeze(mask.cpu().numpy())
pred = np.squeeze(pred.cpu().detach().numpy())


fig = plt.figure(constrained_layout=False, figsize=(10, 3))
spec = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)
ax1 = fig.add_subplot(spec[0, 0])
ax1.set_xlabel("Image", fontsize=20)
plt.imshow(image, cmap="magma")
ax2 = fig.add_subplot(spec[0, 1])
ax2.set_xlabel("SDT", fontsize=20)
plt.imshow(mask, cmap="magma")
ax3 = fig.add_subplot(spec[0, 2])
ax3.set_xlabel("PREDICTION", fontsize=20)
t = plt.imshow(pred, cmap="magma")
cbar = fig.colorbar(t, fraction=0.046, pad=0.04)
tick_locator = ticker.MaxNLocator(nbins=3)
cbar.locator = tick_locator
cbar.update_ticks()
_ = [ax.set_xticks([]) for ax in [ax1, ax2, ax3]]  # remove the xticks
_ = [ax.set_yticks([]) for ax in [ax1, ax2, ax3]]  # remove the yticks
plt.tight_layout()
plt.show()

# %% [markdown]
# <div class="alert alert-block alert-success">
# <h2> Checkpoint 1 </h2>
#
# At this point we have a model that does what we told it too, but do not yet have a segmentation. <br>
# In the next section, we will perform some post-processing and obtain segmentations from our predictions.
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
# <u>Hint</u>: Look at the imports. <br>
# <u>Hint</u>: It is possible to write this function by only adding 2 lines.
# %%
from scipy.ndimage import label, maximum_filter


def find_local_maxima(distance_transform, min_seed_distance):

    # Use `maximum_filter` to perform a maximum filter convolution on the distance_transform

    # Uniquely label the local maxima
    seeds, n = label(...)

    return seeds, n


# %% tags=["solution"]


def find_local_maxima(distance_transform, min_seed_distance):
    # Use `maximum_filter` to perform a maximum filter convolution on the distance_transform
    max_filtered = maximum_filter(distance_transform, min_seed_distance)
    maxima = max_filtered == distance_transform
    # Uniquely label the local maxima
    seeds, n = label(maxima)

    return seeds, n


# %% [markdown]
# We now use this function to find the seeds for the watershed.
# %%
from skimage.segmentation import watershed


def watershed_from_boundary_distance(
    boundary_distances: np.ndarray,
    inner_mask: np.ndarray,
    id_offset: float = 0,
    min_seed_distance: int = 10,
):
    """Function to compute a watershed from boundary distances."""

    seeds, n = find_local_maxima(boundary_distances, min_seed_distance)

    if n == 0:
        return np.zeros(boundary_distances.shape, dtype=np.uint64), id_offset

    seeds[seeds != 0] += id_offset

    # calculate our segmentation
    segmentation = watershed(
        boundary_distances.max() - boundary_distances, seeds, mask=inner_mask
    )

    return segmentation


def get_inner_mask(pred, threshold):
    inner_mask = pred > threshold
    return inner_mask


# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 2.2</b>: <br> Get the watershed segmentation.
# </div>

# %%
idx = np.random.randint(len(val_data))  # take a random sample
image, mask = val_data[idx]  # get the image and the nuclei masks

# Hint: make sure set the model to evaluation

# Choose a threshold value to use to get the boundary mask
thresh = ...

# Get inner mask
inner_mask = ...

# Get the segmentation
seg = watershed_from_boundary_distance(...)


# %% tags=["solution"]
idx = np.random.randint(len(val_data))  # take a random sample
image, mask = val_data[idx]  # get the image and the nuclei masks

# Hint: make sure set the model to evaluation
unet.eval()

image = image.to(device)
pred = unet(torch.unsqueeze(image, dim=0))

image = np.squeeze(image.cpu())
mask = np.squeeze(mask.cpu().numpy())
pred = np.squeeze(pred.cpu().detach().numpy())

# %% tags=["solution"]
# Choose a threshold value to use to get the boundary mask.
# Feel free to play around with the threshold.
threshold = threshold_otsu(pred)
print(f"Foreground threshold is {threshold}")

# Get inner mask
inner_mask = get_inner_mask(pred, threshold=threshold)

# Get the segmentation
seg = watershed_from_boundary_distance(pred, inner_mask, min_seed_distance=20)

# %% tags=["solution"]
# Visualize the results
fig = plt.figure(constrained_layout=False, figsize=(10, 3))
spec = gridspec.GridSpec(ncols=4, nrows=1, figure=fig)
ax1 = fig.add_subplot(spec[0, 0])
ax1.imshow(image)  # show the image
ax1.set_xlabel("Image", fontsize=20)
ax2 = fig.add_subplot(spec[0, 1])
ax2.imshow(mask)  # show the masks
ax2.set_xlabel("SDT", fontsize=20)
ax3 = fig.add_subplot(spec[0, 2])
t = ax3.imshow(pred)
ax3.set_xlabel("Pred.", fontsize=20)
tick_locator = ticker.MaxNLocator(nbins=3)
cbar = fig.colorbar(t, fraction=0.046, pad=0.04)
cbar.locator = tick_locator
cbar.update_ticks()
ax4 = fig.add_subplot(spec[0, 3])
ax4.imshow(seg, cmap=label_cmap, interpolation="none")
ax4.set_xlabel("Seg.", fontsize=20)
_ = [ax.set_xticks([]) for ax in [ax1, ax2, ax3, ax4]]  # remove the xticks
_ = [ax.set_yticks([]) for ax in [ax1, ax2, ax3, ax4]]  # remove the yticks
plt.tight_layout()
plt.show()

# %% [markdown]
# Questions:
# 1. What is the effect of the `min_seed_distance` parameter in watershed?
#       - Experiment with different values.

# %% [markdown]
# <div class="alert alert-block alert-success">
# <h2> Checkpoint 2 </h2>

# %% [markdown]
# <hr style="height:2px;">
#
# ## Section 3: Evaluation
# Many different evaluation metrics exist, and which one you should use is dependant on the specifics of the data.
#
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
# <b>Task 3.2</b>: <br> Evaluate metrics for the validation dataset.
# </div>
# %%
from local import evaluate

# Need to re-initialize the dataloader to return masks in addition to SDTs.
val_dataset = SDTDataset("nuclei_val_data", return_mask=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
unet.eval()

ap_list, precision_list, recall_list, tp_list, fp_list, fn_list = [], [], [], [], [], []
for idx, (image, mask, sdt) in enumerate(tqdm(val_dataloader)):
    image = image.to(device)
    pred = unet(image)

    image = np.squeeze(image.cpu())
    gt_labels = np.squeeze(mask.cpu().numpy())
    pred = np.squeeze(pred.cpu().detach().numpy())

    # feel free to try different thresholds
    thresh = threshold_otsu(pred)

    # get boundary mask
    inner_mask = get_inner_mask(pred, threshold=thresh)

    pred_labels = watershed_from_boundary_distance(
        pred, inner_mask, id_offset=0, min_seed_distance=20
    )
    ap, precision, recall, tp, fp, fn = evaluate(gt_labels, pred_labels)
    ap_list.append(ap)
    precision_list.append(precision)
    recall_list.append(recall)
    tp_list.append(tp)
    fp_list.append(fp)
    fn_list.append(fn)
print(f"Mean Accuracy is {np.mean(ap_list):.3f}")
print(f"Mean Precision is {np.mean(precision_list):.3f}")
print(f"Mean Recall is {np.mean(recall_list):.3f}")
print(f"Mean True Positives is {np.mean(tp_list):.3f}")
print(f"Mean False Positive is {np.mean(fp_list):.3f}")
print(f"Mean False Negatives is {np.mean(fn_list):.3f}")
# %% [markdown]
# <hr style="height:2px;">
#
# ## Section 4: Affinities
# %% [markdown]
# <i>What are affinities? </i><br>
# Here we consider not just the pixel but also its direct neighbors. <br> Imagine there is an edge between two pixels if they
# are in the same class and no edge if not. If we then take all pixels that are directly and indirectly connected by edges, we get an instance. Essentially, we label edges between neighboring pixels as “connected” or “cut”, rather than labeling the pixels themselves. <br>
# Here, below we show the affinity in x + affinity in y.

# %% [markdown]
# ![image](static/05_instance_affinity.png)

# %% [markdown]
# Similar to the pipeline used for SDTs, we first need to modify the dataset to produce affinities.

# %%
# create a new dataset for affinities
from local import compute_affinities


class AffinityDataset(Dataset):
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
                transforms.Normalize([0.5], [0.5]),  # 0.5 = mean and 0.5 = variance
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
        aff_target_array = compute_affinities(np.asarray(mask), [[0, 1], [1, 0]])
        aff_target = torch.from_numpy(aff_target_array)
        return aff_target.float()


# %% [markdown]
# Next we initialize the datasets and data loaders.
# %%
# Initialize the datasets

train_data = AffinityDataset("nuclei_train_data", transforms.RandomCrop(256))
train_loader = DataLoader(train_data, batch_size=5, shuffle=True)
show_random_dataset_image(train_data)

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 4.1</b>: Train a model with affinities as targets.
# </div>
# %% [markdown]
# Repurpose the training loop which you used for the SDTs. <br>
# Think carefully about your loss and final activation. <br>
# (The best for SDT is not necessarily the best for affinities.)


# %% tags=["solutions"]
unet = UNet(
    depth=1,
    in_channels=1,
    num_fmaps=64,
    fmap_inc_factor=3,
    downsample_factor=2,
    padding="same",
    upsample_mode="nearest",
    final_activation="Sigmoid",  # different from SDTs
    out_channels=2,
)

# choose a loss function
loss = torch.nn.MSELoss()

learning_rate = 1e-4
optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)

for epoch in range(20):
    train(
        unet,
        train_loader,
        optimizer,
        loss,
        epoch,
        log_interval=10,
        device=device,
    )

# %% [markdown]
# Let's next look at a prediction on a random image.

# %%
val_data = AffinityDataset("nuclei_val_data", transforms.RandomCrop(256))
val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

unet.eval()
idx = np.random.randint(len(val_data))  # take a random sample
image, mask = val_data[idx]  # get the image and the nuclei masks
image = image.to(device)
pred = unet(torch.unsqueeze(image, dim=0))
image = np.squeeze(image.cpu())
mask = np.squeeze(mask.cpu().numpy())
pred = np.squeeze(pred.cpu().detach().numpy())


fig = plt.figure(constrained_layout=False, figsize=(10, 3))
spec = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)
ax1 = fig.add_subplot(spec[0, 0])
ax1.set_xlabel("Image", fontsize=20)
plt.imshow(image, cmap="magma")
ax2 = fig.add_subplot(spec[0, 1])
ax2.set_xlabel("AFFINITY", fontsize=20)
plt.imshow(mask[0] + mask[1], cmap="magma")
ax3 = fig.add_subplot(spec[0, 2])
ax3.set_xlabel("PREDICTION", fontsize=20)
t = plt.imshow(pred[0] + pred[1], cmap="magma")
cbar = fig.colorbar(t, fraction=0.046, pad=0.04)
tick_locator = ticker.MaxNLocator(nbins=3)
cbar.locator = tick_locator
cbar.update_ticks()
_ = [ax.set_xticks([]) for ax in [ax1, ax2, ax3]]  # remove the xticks
_ = [ax.set_yticks([]) for ax in [ax1, ax2, ax3]]  # remove the yticks
plt.tight_layout()
plt.show()

# %% [markdown]
# Let's also evaluate the model performance.

# %%
val_dataset = AffinityDataset("nuclei_val_data", return_mask=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
unet.eval()

ap_list, precision_list, recall_list, tp_list, fp_list, fn_list = [], [], [], [], [], []
for idx, (image, mask, sdt) in enumerate(tqdm(val_loader)):
    image = image.to(device)
    pred = unet(image)

    image = np.squeeze(image.cpu())
    gt_labels = np.squeeze(mask.cpu().numpy())
    pred = np.squeeze(pred.cpu().detach().numpy())

    # feel free to try different thresholds
    thresh = threshold_otsu(pred)
    inner_mask = 0.5 * (pred[0] + pred[1]) > thresh
    boundary_distances = distance_transform_edt(inner_mask)

    pred_labels = watershed_from_boundary_distance(
        boundary_distances, inner_mask, min_seed_distance=20
    )
    ap, precision, recall, tp, fp, fn = evaluate(gt_labels, pred_labels)
    ap_list.append(ap)
    precision_list.append(precision)
    recall_list.append(recall)
    tp_list.append(tp)
    fp_list.append(fp)
    fn_list.append(fn)
print(f"Mean Accuracy is {np.mean(ap_list):.3f}")
print(f"Mean Precision is {np.mean(precision_list):.3f}")
print(f"Mean Recall is {np.mean(recall_list):.3f}")
print(f"Mean True Positives is {np.mean(tp_list):.3f}")
print(f"Mean False Positive is {np.mean(fp_list):.3f}")
print(f"Mean False Negatives is {np.mean(fn_list):.3f}")

# %% [markdown]
# <hr style="height:2px;">
#
# ## Bonus: Pre-Trained Models
# Try running a pretrained `cellpose` model using the script below.

# %%
# !pip install cellpose
from cellpose import models

model = models.Cellpose(model_type="nuclei")
channels = [[0, 0]]

ap_list, precision_list, recall_list, tp_list, fp_list, fn_list = [], [], [], [], [], []
for idx, (image, mask, _) in enumerate(tqdm(val_loader)):
    gt_labels = np.squeeze(mask.cpu().numpy())
    image = np.squeeze(image.cpu().numpy())
    pred_labels, _, _, _ = model.eval([image], diameter=None, channels=channels)

    ap, precision, recall, tp, fp, fn = evaluate(gt_labels, pred_labels[0])
    ap_list.append(ap)
    precision_list.append(precision)
    recall_list.append(recall)
    tp_list.append(tp)
    fp_list.append(fp)
    fn_list.append(fn)
print(f"Mean Accuracy is {np.mean(ap_list):.3f}")
print(f"Mean Precision is {np.mean(precision_list):.3f}")
print(f"Mean Recall is {np.mean(recall_list):.3f}")
print(f"Mean True Positives is {np.mean(tp_list):.3f}")
print(f"Mean False Positive is {np.mean(fp_list):.3f}")
print(f"Mean False Negatives is {np.mean(fn_list):.3f}")

# %%
