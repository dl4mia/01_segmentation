# %% [markdown]
# # Build Your Own U-Net
#
# <hr style="height:2px;">
#
# In this notebook, we will build and train our own U-Net, with the goal of understanding the architecture and being able to use it for many different tasks in the rest of the course.
#
# Written by Larissa Heinrich, Caroline Malin-Mayor, and Morgan Schwartz, with inspiration from William Patton.

# %% [markdown]
# <div class="alert alert-danger">
# Please use kernel <code>segmentation</code> for this exercise.
# </div>

# %% [markdown]
# Proposed outline of exercise:
# 1. Go through each component (convolution block, max pooling, transposed convolution, sigmoid, skip connection and concatenation), "implement", and verify on a small example.
# 2. Put these together into a U-Net model with arguments controlling the presence or number of each. Test on a small example (data provided).
# 3. Train with various configurations, similar to Will's exercise below, but with actually training multiple configurations (e.g. with and without skip connections), and visually inspect training samples in tensorboard. Use semantic segmentation on kaggle dataset with provided training and no validation, no augmentation, and no quantiative metrics (leave these for actual semantic segmentation exercise.

# %% [markdown]
# <hr style="height:2px;">
#
# ## The libraries

# %%
# %matplotlib inline
# %load_ext tensorboard
import os
from pathlib import Path
import imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from typing import Optional
from local import NucleiDataset, show_random_dataset_image, train

# %%
# make sure gpu is available. Please call a TA if this cell fails
assert torch.cuda.is_available()


# %% [markdown]
# <hr style="height:2px;">
#
# ## The Components of a U-Net

# %% [markdown]
# The [U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) architecture has proven to outperform the other architectures in segmenting biological and medical images. It is also commonly used for other tasks that require the output to be the same resolution as the input, such as style transfer and denoising. Below is an overview figure of the U-Net architecture ([source](https://pythonawesome.com/u-net-architecture-for-multimodal-biomedical-image-segmentation/)). We will go through each of the components first (hint: all of them can be found in the list of PyTorch modules [here](https://pytorch.org/docs/stable/nn.html#convolution-layers)), and then fit them all together to make our very own U-Net.
# ![image](static/unet-image.png)

# %% [markdown]
# ### Convolution Block

# %% [markdown]
# #### Convolution
# A U-Net is a convolutional neural network, which means that the main type of operation is a convolution. Convolutions with defined kernels were covered briefly in the pre-course materials.
#
# <img src="./static/2D_Convolution_Animation.gif" width="400" height="300">


# %% [markdown]
# Shown here is a 3x3 kernel being convolved with an input array to get an output array of the same size. For each pixel of the input, the value at that same pixel of the output is computed by multiplying the kernel element-wise with the surrounding 3x3 neighborhood around the input pixel, and then summing the result.


# %% [markdown]
# #### Padding
#
# You will notice that at the edges of the input, this animation shows greyed out values that extend past the input. This is known as padding the input. This example uses "same" padding, which means the values at the edges are repeated. The other option we will use in this exercise is "valid" padding, which essentially means no padding. In the case of valid padding, the output will be smaller than the input, as values at the edges of the output will not be computed. "Same" padding can introduce edge artifacts, but "valid" padding loses output size at every convolution. Note that the amount of padding (for same) and the amount of size lost (for valid) depends on the size of the kernel - a 3x3 convolution would require padding of 1, a 5x5 convolution would require a padding of 2, and so on.


# %% [markdown]
# #### ReLU Activation
# The Rectified Linear Unit (ReLU) is a common activation function, which takes the max of a value and 0, shown below. It introduces a non-linearity into the neural network - without a non-linear activation function, a neural network could not learn non-linear functions.
#
# <img src="./static/ReLU.png" width="400" height="300">

# %% [markdown]
# #### Convolution block
# The convolution block (ConvBlock) of a standard U-Net has two 3x3 convolutions, each of which is followed by a ReLU activation. Our implementation will handle other sizes of convolutions as well. The first convolution in the block will handle changing the input number of feature maps/channels into the output, and the second convolution will have the same number of feature maps in and out. You will use [torch.nn.Conv2D](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d) and [torch.nn.ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU) to implement the ConvBlock below.


# %%
class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: str = "same",
    ):
        """ A convolution block for a U-Net. Contains two convolutions, each followed by a ReLU.

        Args:
            in_channels (int): The number of input channels for this conv block. Depends on
                the layer and side of the U-Net and the hyperparameters.
            out_channels (int): The number of output channels for this conv block. Depends on
                the layer and side of the U-Net and the hyperparameters.
            kernel_size (int): The size of the kernel. A kernel size of N signifies an
                NxN square kernel.
        """
        super(ConvPass, self).__init__()

        # determine padding size based on method
        if padding in ("VALID", "valid"):
            pad = 0  # compute this
        elif padding in ("SAME", "same"):
            pad = kernel_size // 2  # compute this
        else:
            raise RuntimeError("invalid string value for padding")

        # define layers in conv pass
        self.conv_pass = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, padding=pad
            ),  # leave out
            torch.nn.ReLU(),  # leave out
            torch.nn.Conv2d(
                out_channels, out_channels, kernel_size=kernel_size, padding=pad
            ),  # leave out
            torch.nn.ReLU(),  # leave out
        )

    def forward(self, x):
        return self.conv_pass(x)


# %% [markdown]
# ## Visualize Output of ConvBlock


# %%
def apply_and_show_random_image(f):
    ds = NucleiDataset("nuclei_train_data")
    img_tensor = ds[np.random.randint(len(ds))][0]
    batch_tensor = torch.unsqueeze(img_tensor, 0)
    result_tensor = f(batch_tensor)
    result_tensor = result_tensor.squeeze(0)
    img_arr = img_tensor.numpy()[0]
    img_min, img_max = (img_arr.min(), img_arr.max())
    result_arr = result_tensor.detach().numpy()[0]
    result_min, result_max = (result_arr.min(), result_arr.max())
    fig, axs = plt.subplots(1,2, figsize=(10,20), sharey=True, sharex=True)
    axs[1].imshow(result_arr, vmin=result_min,vmax=result_max, aspect="equal")
    axs[1].set_title("First Channel of Output")
    axs[1].set_xlabel(f"min: {result_min:.2f}, max: {result_max:.2f}, shape: {result_arr.shape}")
    
    axs[0].imshow(img_arr, vmin = img_min, vmax = img_max, aspect="equal")
    axs[0].set_title("Input Image")
    axs[0].set_xlabel(f"min: {img_min:.2f}, max: {img_max:.2f}, shape: {img_arr.shape}")
    for ax in axs:
        for spine in ["bottom", "top", "left", "right"]:
            ax.spines[spine].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
    


# %%
torch.manual_seed(63)
conv = ConvPass(1,2,3,"valid")
apply_and_show_random_image(conv)


# %% [markdown]
# ### Downsampling / Max Pooling

# %% [markdown]
# Between each layer of the U-Net on the left side, there is a downsample step. Traditionally, this is done with a 2x2 max pooling operation : each 2x2 square of input values is replaced with the maximum value in the output. This results in an output that is half the size in each dimension. There are other ways to downsample, for example with average pooling, but we will stick with max pooling for this exercise.
#
# Below, you will implement a Downsample model using [torch.nn.MaxPool2D](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html). The downsample factor will be provided as an argument to the module, allowing you to customize how much you downsample at each layer. However, not every value can be used to downsample - if the downsample factor does not evenly divide the dimensions of the input to the layer, the Downsample module should throw an error.


# %%
class Downsample(torch.nn.Module):
    def __init__(self, downsample_factor: int):
        """TODO: Docstring"""

        super(Downsample, self).__init__()

        self.downsample_factor = downsample_factor

        self.down = torch.nn.MaxPool2d(
            downsample_factor
        )  # leave out

    def check_valid(self, image_size: tuple[int, int]) -> bool:
        """Check if the downsample factor evenly divides each image dimension
        Note: there are multiple ways to do this!
        """
        for dim in image_size:
            if dim % self.downsample_factor != 0:  # Leave out whole implementation
                return False
        return True

    def forward(self, x):
        if not self.check_valid(tuple(x.size()[-2:])):
            raise RuntimeError(
                "Can not downsample shape %s with factor %s"
                % (x.size(), self.downsample_factor)
            )

        return self.down(x)


# %%
down = Downsample(16)
apply_and_show_random_image(down)

# %% [markdown]
# ### Upsampling

# %% [markdown]
# The right side of the U-Net contains upsampling between the layers. There are many ways to upsample: we will examine those implemented in [torch.nn.Upsample](https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html#torch.nn.Upsample).


# %%
# First, we make fake input to illustrate the upsampling techniques
# Pytorch expects a batch and channel dimension before the actual data,
# So this simulates a 1D input
sample_1d_input = torch.tensor([[[1,2,3,4]]], dtype=torch.float64)
# And this simulates a 2D input
sample_2d_input = torch.tensor([[[[1,2,],[3,4]]]], dtype=torch.float64)
sample_2d_input

# %%
# See what happens if you vary the scale_factor and modes - you will need different dimensional
# inputs for certain modes
torch.nn.functional.upsample(sample_2d_input, scale_factor=2, mode="bicubic")
# Note: To test a pytorch function outside of a module, you can use torch.nn.functional.upsample
# However, when you use it inside a module, just use torch.nn.Upsample

# %% [markdown]
# Now we will implement our Upsample module. The scale factor and mode will the passed as arguments to the module, and you should pass these along to [torch.nn.Upsample](https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html#torch.nn.Upsample).

# %%
class Upsample(torch.nn.Module):
    def __init__(
        self,
        scale_factor: int,  # passed to torch.nn.Upsample
        mode: str = "nearest",  # passed to torch.nn.Upsample
    ):
        """TODO: docstring"""
        super(Upsample, self).__init__()
        self.up = torch.nn.Upsample(scale_factor=scale_factor, mode=mode)  # leave out

    def forward(self, x):
        return self.up(x)  # leave out


# %%
up = Upsample(2)
apply_and_show_random_image(up)

# %% [markdown]
# ### Skip Connections and Concatenation

# %% [markdown]
# TODO: Add explanation
#
# TODO: Add test


# %%
class CropAndConcat(torch.nn.Module):
    def crop(self, x, y):
        """Center-crop x to match spatial dimensions given by y."""

        x_target_size = x.size()[:-2] + y.size()[-2:]

        offset = tuple((a - b) // 2 for a, b in zip(x.size(), x_target_size))

        slices = tuple(slice(o, o + s) for o, s in zip(offset, x_target_size))

        return x[slices]

    def forward(self, f_left, g_out):
        """TODO: Docstring"""
        f_cropped = self.crop(f_left, g_out)  # leave this out

        return torch.cat([f_cropped, g_out], dim=1)  # leave this out


# %% [markdown]
# ### Output Block

# %% [markdown]
# Create a final block for outputting what you want for your class. Different than normal conv block
#
# TODO: explanation, test


# %%
class OutputConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Optional[
            str
        ] = None,  # Accepts the name of any torch activation function (e.g., ``ReLU`` for ``torch.nn.ReLU``).
    ):
        """
        Use a convolution with kernel size 1 to obtain the appropriate number of output channels. Then apply final activation.

        """
        super().__init__()
        conv = torch.nn.Conv2d(in_channels, out_channels, 1, padding=0)
        if activation is not None:
            activation = getattr(torch.nn, activation)
            self.final_conv = torch.nn.Sequential[conv, activation]
        else:
            self.final_conv = conv

    def forward(self, x):
        return self.final_conv(x)


# %% [markdown]
# <div class="alert alert-block alert-success">
#     <h2>Checkpoint 1</h2>
#
# Congratulations! You have implemented most of a U-Net!
#
# Here are some questions for you to consider.
# TODO:
#
#
#
# We will go over this portion together and answer any questions soon, but feel free to start on the next section where we put it all together.
#
# </div>
#
# <hr style="height:2px;">

# %% [markdown]
# ## Putting the UNet together
#
# Now we will make a UNet class that combines all of these components as shown in the image. Because our UNet class will inherit from pytorch.nn.Module, we get a lot of functionality for free - we just need to initialize the model and write a forward function.
# ![image](static/unet-image.png)


# %% [markdown]
# TODO: Finish cleaning up this UNet to match our defined solutions above. Then, decide which parts they should implement.

# %%


class UNet(torch.nn.Module):
    def __init__(
        self,
        depth,
        in_channels,
        num_fmaps,
        fmap_inc_factor,
        downsample_factor,
        kernel_size=3,
        padding="VALID",
        upsample_mode="nearest",
        final_activation=None,
        out_channels=1,
    ):
        """Create a U-Net::
            f_in --> f_left ------------------>> f_right + f_up--> f_out
                        |                                   ^
                        v                                   |
                     g_in --> g_left ------->> g_right --> g_out
                                 |               ^
                                 v               |
                                       ...
        where each ``-->`` is a convolution pass, each `-->>` a crop, and down
        and up arrows are max-pooling and transposed convolutions,
        respectively.
        The U-Net expects 2D tensors shaped like::
            ``(batch=1, channels, height, width)``.
        Args:
            depth:
                The number of levels in the UNet. 2 is the smallest that really
                makes sense for the UNet architecture, as a one layer UNet is
                basically just 2 conv blocks.
            in_channels:
                The number of input channels.
            num_fmaps:
                The number of feature maps in the first layer. This is also the
                number of output feature maps. Stored in the ``channels``
                dimension.
            fmap_inc_factor:
                By how much to multiply the number of feature maps between
                layers. If layer 0 has ``k`` feature maps, layer ``l`` will
                have ``k*fmap_inc_factor**l``.
            downsample_factor:
                Factor to use for down- and up-sampling the
                feature maps between layers.
            kernel_size (optional):
                Kernel size to use in convolutions on both sides of the UNet.
                Defaults to 3.
            padding (optional):
                How to pad convolutions. Either 'same' or 'valid' (default).
            final_activation (optional):
                What activation to use in your final output block. Depends on your task.
                Defaults to sigmoid
            out_channels (optional):
                How many output channels you want. Depends on your task. Defaults to 1.
        """

        super(UNet, self).__init__()

        self.depth = depth
        self.in_channels = in_channels
        self.num_fmaps = num_fmaps
        self.fmap_inc_factor = fmap_inc_factor
        self.downsample_factor = downsample_factor
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.upsample_mode = upsample_mode
        self.final_activation = final_activation
        self.out_channels = out_channels

        # left convolutional passes
        self.l_conv = torch.nn.ModuleList()
        for level in range(self.depth):
            fmaps_in, fmaps_out = self.compute_fmaps_encoder(level)
            self.l_conv.append(
                ConvBlock(fmaps_in, fmaps_out, self.kernel_size, self.padding)
            )

        self.l_down = torch.nn.ModuleList()
        for level in range(self.depth - 1):
            self.l_down.append(Downsample(self.downsample_factor))

        # right up/crop/concatenate layers
        self.r_up = torch.nn.ModuleList()
        for level in range(self.depth - 1):
            self.r_up.append(
                Upsample(
                    downsample_factor,
                    mode=self.upsample_mode,
                )
            )
        self.crop_up = torch.nn.ModuleList()
        for level in range(self.depth - 1):
            self.crop_up.append(CropAndConcat())

        # right convolutional passes
        self.r_conv = torch.nn.ModuleList()
        for level in range(self.depth - 1):
            fmaps_in, fmaps_out = self.compute_fmaps_decode(level)
            self.r_conv.append(
                ConvPass(
                    fmaps_in,
                    fmaps_out,
                    self.kernel_size,
                    self.padding,
                )
            )
        self.final_conv = OutputConv(
            self.compute_fmaps_decode(0)[1], self.out_channels, self.final_activation
        )

    def compute_fmaps_encoder(self, level: int) -> tuple[int, int]:
        """Compute the number of input and output feature maps for a conv block at a given level
        of the UNet encoder. TODO: add args, output
        """
        if level == 0:
            fmaps_in = self.in_channels
        else:
            fmaps_in = self.num_fmaps * self.fmap_inc_factor ** (level - 1)

        fmaps_out = self.num_fmaps * self.fmap_inc_factor**level
        return fmaps_in, fmaps_out

    def compute_fmaps_decode(self, level: int) -> tuple[int, int]:
        """Compute the number of input and output feature maps for a conv block at a given level
        of the UNet decoder. TODO: add args, output
        """
        fmaps_out = self.num_fmaps * self.fmap_inc_factor ** (level)
        concat_fmaps = self.compute_fmaps_encoder(level)[
            1
        ]  # The channels that come from the skip connection
        fmaps_in = concat_fmaps + self.num_fmaps * self.fmap_inc_factor ** (level + 1)

        return fmaps_in, fmaps_out

    def rec_forward(self, level, f_in):

        # index of level in layer arrays
        i = self.depth - level - 1

        # convolve
        f_left = self.l_conv[i](f_in)

        # end of recursion
        if level == 0:
            fs_out = f_left
        else:
            # down
            g_in = self.l_down[i](f_left)
            # nested levels
            gs_out = self.rec_forward(level - 1, g_in)
            # up, concat, and crop
            f_up = self.r_up[i](gs_out)
            fs_right = self.crop_up[i](f_left, f_up)

            # convolve
            fs_out = self.r_conv[i](fs_right)

        return fs_out

    def forward(self, x):

        y = self.rec_forward(self.depth - 1, x)

        return self.final_conv(y)


# %%
simple_net = UNet(
        depth=2,
        in_channels=1,
        num_fmaps=12,
        fmap_inc_factor=3,
        downsample_factor=2,
        kernel_size=3,
        padding="same",
        upsample_mode="nearest",)
# TODO: fix valid padding error
# TODO: Apply to one image to test that no errors happen

# %% [markdown]
# <div class="alert alert-block alert-success">
#     <h2>Checkpoint 2</h2>
#
# We are about ready to start training! But before we do, we will stop and discuss your guesses.
#
# Questions to consider:
# TODO
#
# </div>
#
# <hr style="height:2px;">

# %% [markdown]
# ## Let's try the UNet!
# We will get more into the details of training and evaluating semantic segmentation models in the next exercise. For now, we will provide an example pipeline that will train a UNet to classify each pixel in an image of cells as foreground or background.

# %%
dataset = NucleiDataset("nuclei_train_data")
for i in range(5):
    show_random_dataset_image(dataset)

# %%
train_loader = DataLoader(dataset)

# %% tags=["solution"]
loss_function: torch.nn.Module = torch.nn.BCELoss()


# %% tags=["solution"]
# apply training for one epoch
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


# %% [markdown]
# <div class="alert alert-block alert-warning">
#     Quick sanity check for your train function to make sure no errors are thrown:
#     Good place to test unetA, unetB, unetC, unetD to see if you can eliminate some
# </div>


# %%

train(
    simple_net,
    train_loader,
    optimizer=torch.optim.Adam(simple_net.parameters()),
    loss_function=torch.nn.MSELoss(),
    epoch=0,
    log_interval=1,
    early_stop=True,
)


# %%
# start a tensorboard writer
logger = SummaryWriter("runs/Unet")
# %tensorboard --logdir runs

# %%
# Here is where students can define their own unet with whatever parameters they want to try,
# or use one of the examples from the thought exercise
model = UNet(...)

# %%
# use adam optimizer
optimizer = torch.optim.Adam(model.parameters())

# train for $25$ epochs
# during the training you can inspect the
# predictions in the tensorboard
n_epochs = 25
for epoch in range(n_epochs):
    # train
    train(
        model,
        train_loader,
        optimizer=optimizer,
        loss_function=loss_function,
        epoch=epoch,
        log_interval=5,
        tb_logger=logger,
    )


# %% [markdown]
# TODO: Make them try a drastic variation to understand the importance of some aspect of the network (e.g. no skip connections, one layer, etc.). Might make sense to explain activation functions if we want them to use non-relu ones later.

# %% [markdown]
# <div class="alert alert-block alert-success">
#     <h2>Checkpoint 3</h2>
#
# This is the end of the guided exercise. We will go over all of the code up until this point shortly. While you wait you are encouraged to try alternative loss functions, evaluation metrics, augmentations, and networks.
# After this come additional exercises if you are interested and have the time.
#
# </div>
# <hr style="height:2px;">

# %% [markdown]
# ## Additional Exercises
#
# 1. Modify and evaluate the following architecture variants of the U-Net:
#     * use [GroupNorm](https://pytorch.org/docs/stable/nn.html#torch.nn.GroupNorm) to normalize convolutional group inputs
#     * use more layers in your U-Net.
#
# 2. Use the Dice Coefficient as loss function. Before we only used it for validation, but it is differentiable and can thus also be used as loss. Compare to the results from exercise 2.
# Hint: The optimizer we use finds minima of the loss, but the minimal value for the Dice coefficient corresponds to a bad segmentation. How do we need to change the Dice Coefficient to use it as loss nonetheless?
#
# 3. Compare the results of these trainings to the first one. If any of the modifications you've implemented show better results, combine them (e.g. add both GroupNorm and one more layer) and run trainings again.
# What is the best result you could get?

# %% [markdown]
#
# <div class="alert alert-block alert-info">
#     <b>Task BONUS 4.1</b>: Group Norm, update the U-Net to use a GroupNorm layer
# </div>


# %%
class UNetGN(UNet):
    """
    A subclass of UNet that implements GroupNorm in each convolutional block
    """

    # Convolutional block for single layer of the decoder / encoder
    # we apply two 2d convolutions with relu activation
    def _conv_block(self, in_channels, out_channels):
        # See the original U-Net for an example of how to build the convolutional block
        # We want operation -> activation -> normalization (2x)
        # Hint: Group norm takes a "num_groups" argument. Use 8 to match the solution
        return ...


# %% tags=["solution"]
class UNetGN(UNet):
    """
    A subclass of UNet that implements GroupNorm in each convolutional block
    """

    # Convolutional block for single layer of the decoder / encoder
    # we apply two 2d convolutions with relu activation
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, out_channels),
        )


# %%
model = UNetGN(1, 1, final_activation=nn.Sigmoid())

optimizer = torch.optim.Adam(model.parameters())

metric = DiceCoefficient()

logger = SummaryWriter("runs/UNetGN")


# train for 40 epochs
# during the training you can inspect the
# predictions in the tensorboard
n_epochs = 40
for epoch in range(n_epochs):
    train(
        model,
        train_loader,
        optimizer=optimizer,
        loss_function=loss_function,
        epoch=epoch,
        log_interval=5,
        tb_logger=logger,
    )
    step = epoch * num_train_pairs
    validate(model, val_loader, loss_function, metric, step=step, tb_logger=logger)


# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Task BONUS 4.2</b>: More Layers
# </div>

# %%
# Experiment with more layers. For example UNet with depth 5

model = ...

optimizer = torch.optim.Adam(model.parameters())

metric = DiceCoefficient()

loss = torch.nn.BCELoss()

logger = SummaryWriter("runs/UNet5layers")

# %% tags=["solution"]
# Experiment with more layers. For example UNet with depth 5

model = UNet(1, 1, depth=5, final_activation=nn.Sigmoid())

optimizer = torch.optim.Adam(model.parameters())

metric = DiceCoefficient()

loss = torch.nn.BCELoss()

logger = SummaryWriter("runs/UNet5layers")

# %%
# train for 25 epochs
# during the training you can inspect the
# predictions in the tensorboard
n_epochs = 25
for epoch in range(n_epochs):
    train(
        model,
        train_loader,
        optimizer=optimizer,
        loss_function=loss,
        epoch=epoch,
        log_interval=5,
        tb_logger=logger,
    )
    step = epoch * num_train_pairs
    validate(model, val_loader, loss, metric, step=step, tb_logger=logger)


# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Task BONUS 4.3</b>: Dice Loss
#     Dice Loss is a simple inversion of the Dice Coefficient.
#     We already have a Dice Coefficient implementation, so now we just
#     need a layer that can invert it.
# </div>


# %%
class DiceLoss(nn.Module):
    """ """

    def __init__(self, offset: float = 1):
        super().__init__()
        self.dice_coefficient = DiceCoefficient()

    def forward(self, x, y): ...


# %% tags=["solution"]
class DiceLoss(nn.Module):
    """
    This layer will simply compute the dice coefficient and then negate
    it with an optional offset.
    We support an optional offset because it is common to have 0 as
    the optimal loss. Since the optimal dice coefficient is 1, it is
    convenient to get 1 - dice_coefficient as our loss.

    You could leave off the offset and simply have -1 as your optimal loss.
    """

    def __init__(self, offset: float = 1):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(offset), requires_grad=False)
        self.dice_coefficient = DiceCoefficient()

    def forward(self, x, y):
        coefficient = self.dice_coefficient(x, y)
        return self.offset - coefficient


# %%
# Now combine the Dice Coefficient layer with the Invert layer to make a Dice Loss
dice_loss = ...


# %% tags=["solution"]
# Now combine the Dice Coefficient layer with the Invert layer to make a Dice Loss
dice_loss = DiceLoss()

# %%
# Experiment with Dice Loss
net = ...
optimizer = ...
metric = ...
loss_func = ...

# %% tags=["solution"]
# Experiment with Dice Loss
net = UNet(1, 1, final_activation=nn.Sigmoid())
optimizer = torch.optim.Adam(net.parameters())
metric = DiceCoefficient()
loss_func = dice_loss

# %%
logger = SummaryWriter("runs/UNet_diceloss")

n_epochs = 40
for epoch in range(n_epochs):
    train(
        net,
        train_loader,
        optimizer=optimizer,
        loss_function=loss_func,
        epoch=epoch,
        log_interval=5,
        tb_logger=logger,
    )
    step = epoch * num_train_pairs
    validate(net, val_loader, loss_func, metric, step=step, tb_logger=logger)
