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

# %%
# make sure gpu is available. Please call a TA if this cell fails
assert torch.cuda.is_available()


# %% [markdown]
# <hr style="height:2px;">
#
# ## The Components of a U-Net

# %% [markdown]
# The[U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) architecture has proven to outperform the other architectures in segmenting biological and medical images. It is also commonly used for other tasks that require the output to be the same resolution as the input, such as style transfer and denoising. Below is an overview figure of the U-Net architecture ([source](https://pythonawesome.com/u-net-architecture-for-multimodal-biomedical-image-segmentation/)). We will go through each of the components first (hint: all of them can be found in the list of PyTorch modules [here](https://pytorch.org/docs/stable/nn.html#convolution-layers)), and then fit them all together to make our very own U-Net.
# ![image](static/unet-image.png)

# %% [markdown]
# ### Convolution Block

# %% [markdown]
# TODO: Explanation for pad computation (what is padding? what is valid/same? How to compute?)
#
# TODO: Explanation for conv_pass layers (links to docs)
#
# TODO: Write a test case with defined (blur? Random?) kernel and visualize the input/output of the conv block.


# %%
# target output can be somewhere between the unet.py implementation - but I suggest sticking with two convs, relu
class ConvPass(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: str = "same",
    ):
        """TODO: Docstring"""
        super(ConvPass, self).__init__()

        # determine padding size based on method
        if padding in ("VALID", "valid"):
            pad = 0  # compute this
        elif padding in ("SAME", "same"):
            pad = tuple(np.array(kernel_size) // 2)  # compute this
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
# ### Downsampling / Max Pooling

# %% [markdown]
# TODO: Explain downsampling (link to maxPool2d, mention other ways are possible but this is normal) (talk about invalid downsampling factor if input is bad size)
#
# TOOD: test case afterward


# %%
class Downsample(torch.nn.Module):
    def __init__(self, downsample_factor: int):
        """TODO: Docstring"""

        super(Downsample, self).__init__()

        self.downsample_factor = downsample_factor

        self.down = torch.nn.MaxPool2d(
            downsample_factor, stride=downsample_factor
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
        if not check_valid(tuple(x.size()[-2:])):
            raise RuntimeError(
                "Can not downsample shape %s with factor %s"
                % (x.size(), self.downsample_factor)
            )

        return self.down(x)


# %% [markdown]
# ### Upsampling

# %% [markdown]
# TODO: explanation (link torch.nn.Upsample) (hint that they pass factor not size)
#
# TODO: Test


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

        return torch.cat([f_cropped, g_cropped], dim=1)  # leave this out


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

        conv = torch.nn.Conv2D(in_channels, out_channels, 1, padding=0)
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
            f_in --> f_left --------------------------->> f_right--> f_out
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
        self.final_activation = final_activation
        self.out_channels = out_channels

        # left convolutional passes
        self.l_conv = nn.ModuleList()
        for level in range(self.depth):
            fmaps_in, fmaps_out = self.compute_fmaps_encoder(level)
            self.l_conv.append(
                ConvPass(fmaps_in, fmaps_out, self.kernel_size, self.padding)
            )

        self.l_down = nn.ModuleList()
        for level in range(self.depth - 1):
            self.l_down.append(Downsample(self.downsample_factor))

        # right up/crop/concatenate layers
        self.r_up = nn.ModuleList()
        for level in range(self.depth - 1):
            self.r_up.append(
                Upsample(
                    downsample_factor,
                    mode=self.upsample_mode,
                )
            )
        self.crop_up = nn.ModuleList()
        for level in range(self.depth - 1):
            self.crop_up.append(CropAndConcat())

        # right convolutional passes
        self.r_conv = nn.ModuleList()
        for level in range(self.depth - 1):
            fmaps_in, fmaps_out = self.cnmpute_fmaps_decode(level)
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
            fs_right = self.r_up[i](f_left, gs_out)

            # convolve
            fs_out = self.r_conv[i](fs_right)

        return fs_out

    def forward(self, x):

        y = self.rec_forward(self.depth - 1, x)

        return self.final_conv(y)


# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Task 2.1</b>: Spot the best U-Net
#
# In the next cell you fill find a series of U-Net definitions. Most of them won't work. Some of them will work but not well. One will do well. Can you identify which model is the winner? Unfortunately you can't yet test your hypotheses yet since we have not covered loss functions, optimizers, and train/validation loops.
#
# </div>

# %%
unetA = UNet(
    in_channels=1, out_channels=1, depth=4, final_activation=torch.nn.Sigmoid()
)
unetB = UNet(in_channels=1, out_channels=1, depth=9, final_activation=None)
unetC = torch.nn.Sequential(
    UNet(in_channels=1, out_channels=1, depth=4, final_activation=torch.nn.ReLU()),
    torch.nn.Sigmoid(),
)
unetD = torch.nn.Sequential(
    UNet(in_channels=1, out_channels=1, depth=1, final_activation=None),
    torch.nn.Sigmoid(),
)


# %%
# Provide your guesses as to what, if anything, might go wrong with each of these models:
#
# unetA:
#
# unetB:
#
# unetC:
#
# unetD:

favorite_unet: UNet = ...

# %% tags=["solution"]
# Provide your guesses as to what, if anything, might go wrong with each of these models:
#
# unetA: The correct unet.
#
# unetB: Too deep. You won't be able to train with input size 256 since the lowest level will get zero sized tensors.
#
# unetC: A classic mistake putting a Sigmoid after a Relu activation. You will never predict anything < 0.5
#
# unetD: barely any depth to this unet. It should train and give you what you want, I just wouldn't expect good performance

favorite_unet: UNet = unetA

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

# %% [markdown]
# TODO: add back in the data loaders (without augmentation) and

# %% tags=["solution"]
loss_function: torch.nn.Module = nn.BCELoss()


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
simple_net = UNet(1, 1, depth=1, final_activation=nn.Sigmoid())

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
