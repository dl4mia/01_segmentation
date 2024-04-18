
import os
import imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from mpl_toolkits.axes_grid1 import make_axes_locatable



def show_one_image(image_path):
    image = imageio.imread(image_path)
    plt.imshow(image)


class NucleiDataset(Dataset):
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


def show_random_dataset_image(dataset):
    idx = np.random.randint(0, len(dataset))  # take a random sample
    img, mask = dataset[idx]  # get the image and the nuclei masks
    f, axarr = plt.subplots(1, 2)  # make two plots on one figure
    axarr[0].imshow(img[0])  # show the image
    axarr[1].imshow(mask[0], interpolation=None)  # show the masks
    _ = [ax.axis("off") for ax in axarr]  # remove the axes
    print("Image size is %s" % {img[0].shape})
    plt.show()


def apply_and_show_random_image(f, path="nuclei_train_data"):
    ds = NucleiDataset(path)

    # pick random raw image from dataset
    img_tensor = ds[np.random.randint(len(ds))][0]

    batch_tensor = torch.unsqueeze(img_tensor, 0) # add batch dimension that some torch modules expect
    out_tensor = f(batch_tensor) # apply torch module
    out_tensor = out_tensor.squeeze(0) # remove batch dimension
    img_arr = img_tensor.numpy()[0] # turn into numpy array, look at first channel
    out_arr = out_tensor.detach().numpy()[0] # turn into numpy array, look at first channel
    
    # intialilze figure
    fig, axs = plt.subplots(1,2, figsize=(10,20))
    
    # Show input image, add info and colorbar
    img_min, img_max = (img_arr.min(), img_arr.max())  # get value range
    inim = axs[0].imshow(img_arr, vmin = img_min, vmax = img_max)
    axs[0].set_title("Input Image")
    axs[0].set_xlabel(f"min: {img_min:.2f}, max: {img_max:.2f}, shape: {img_arr.shape}")
    div = make_axes_locatable(axs[0])
    cb = fig.colorbar(inim, cax=div.append_axes("right", size="5%", pad=0.05))
    cb.outline.set_visible(False)
    
    # Show ouput image, add info and colorbar
    out_min, out_max = (out_arr.min(), out_arr.max()) # get value range
    outim = axs[1].imshow(out_arr, vmin=out_min,vmax=out_max)
    axs[1].set_title("First Channel of Output")
    axs[1].set_xlabel(f"min: {out_min:.2f}, max: {out_max:.2f}, shape: {out_arr.shape}")
    div = make_axes_locatable(axs[1])
    cb = fig.colorbar(outim, cax=div.append_axes("right", size="5%", pad=0.05))
    cb.outline.set_visible(False)

    # center images and remove ticks
    max_bounds = [max(ax.get_ybound()[1] for ax in axs), max(ax.get_xbound()[1] for ax in axs)]
    for ax in axs:
        diffy = abs(ax.get_ybound()[1] - max_bounds[0])
        diffx = abs(ax.get_xbound()[1] - max_bounds[1])
        ax.set_ylim([ax.get_ybound()[0]-diffy/2.,max_bounds[0]-diffy/2.])
        ax.set_xlim([ax.get_xbound()[0] -diffx/2., max_bounds[1]-diffx/2.])
        ax.set_xticks([])
        ax.set_yticks([])
        
        # for spine in ["bottom", "top", "left", "right"]: # get rid of box
        #     ax.spines[spine].set_visible(False)
    


def train(
    model,
    loader,
    optimizer,
    loss_function,
    epoch,
    log_interval=100,
    log_image_interval=20,
    tb_logger=None,
    device=None,
    early_stop=False,
):
    if device is None:
        # You can pass in a device or we will default to using
        # the gpu. Feel free to try training on the cpu to see
        # what sort of performance difference there is
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # set the model to train mode
    model.train()

    # move model to device
    model = model.to(device)

    # iterate over the batches of this epoch
    for batch_id, (x, y) in enumerate(loader):
        # move input and target to the active device (either cpu or gpu)
        x, y = x.to(device), y.to(device)

        # zero the gradients for this iteration
        optimizer.zero_grad()

        # apply model and calculate loss
        prediction = model(x)
        if prediction.shape != y.shape:
            y = crop(y, prediction)
        loss = loss_function(prediction, y)

        # backpropagate the loss and adjust the parameters
        loss.backward()
        optimizer.step()

        # log to console
        if batch_id % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_id * len(x),
                    len(loader.dataset),
                    100.0 * batch_id / len(loader),
                    loss.item(),
                )
            )

        # log to tensorboard
        if tb_logger is not None:
            step = epoch * len(loader) + batch_id
            tb_logger.add_scalar(
                tag="train_loss", scalar_value=loss.item(), global_step=step
            )
            # check if we log images in this iteration
            if step % log_image_interval == 0:
                tb_logger.add_images(
                    tag="input", img_tensor=x.to("cpu"), global_step=step
                )
                tb_logger.add_images(
                    tag="target", img_tensor=y.to("cpu"), global_step=step
                )
                tb_logger.add_images(
                    tag="prediction",
                    img_tensor=prediction.to("cpu").detach(),
                    global_step=step,
                )

        if early_stop and batch_id > 5:
            print("Stopping test early!")
            break

def compute_receptive_field(depth, kernel_size, downsample_factor):
    fov = 1
    downsample_factor_prod = 1
    # encoder
    for layer in range(depth - 1):
        # two convolutions, each adds (kernel size - 1 ) * current downsampling level
        fov = fov + 2 * (kernel_size - 1) * downsample_factor_prod
        # downsampling multiplies by downsample factor
        fov = fov * downsample_factor
        downsample_factor_prod *= downsample_factor
    # bottom layer just two convs
    fov = fov + 2 * (kernel_size - 1) * downsample_factor_prod

    # decoder
    for layer in range(0, depth - 1)[::-1]:
        # upsample
        downsample_factor_prod /= downsample_factor
        # two convolutions, each adds (kernel size - 1) * current downsampling level
        fov = fov + 2 * (kernel_size - 1) * downsample_factor_prod

    return fov


def plot_receptive_field(unet, npseed=10, path="nuclei_train_data"):
    ds = NucleiDataset(path)
    np.random.seed(npseed)
    img_tensor = ds[np.random.randint(len(ds))][0]
    
    img_arr = np.squeeze(img_tensor.numpy())
    print(img_arr.shape)
    fov = compute_receptive_field(unet.depth, unet.kernel_size, unet.downsample_factor)

    fig=plt.figure(figsize=(5, 5))
    plt.imshow(img_arr)#, cmap='gray')
    
    # visualize receptive field
    xmin = img_arr.shape[1]/2 - fov/2
    xmax = img_arr.shape[1]/2 + fov/2
    ymin = img_arr.shape[0]/2 - fov/2
    ymax = img_arr.shape[0]/2 + fov/2
    color = "red"
    plt.hlines(ymin, xmin, xmax, color=color, lw=3)
    plt.hlines(ymax, xmin, xmax, color=color, lw=3)
    plt.vlines(xmin, ymin, ymax, color=color, lw=3)
    plt.vlines(xmax, ymin, ymax, color=color, lw=3)
    plt.show()