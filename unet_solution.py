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
# TODOS:
# - Test for crop and concat
# - Test shape of outputs for different blocks
# - Better explain unet implementation where each element of list is a layer (good scaffolding)
# - Create solution tags and empty scaffolding
# - Introduce dataset and segmentation in one cell
# - Better training tasks

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
from local import NucleiDataset, show_random_dataset_image, train, apply_and_show_random_image
import unet_tests

# %%
# make sure gpu is available. Please call a TA if this cell fails
# assert torch.cuda.is_available()


# %% [markdown]
# <hr style="height:2px;">
#
# ## The Components of a U-Net

# %% [markdown]
# The [U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) architecture has proven to outperform the other architectures in segmenting biological and medical images. It is also commonly used for other tasks that require the output to be the same resolution as the input, such as style transfer and denoising. Below is an overview figure of the U-Net architecture from the original [paper](https://arxiv.org/pdf/1505.04597.pdf). We will go through each of the components first (hint: all of them can be found in the list of PyTorch modules [here](https://pytorch.org/docs/stable/nn.html#convolution-layers)), and then fit them all together to make our very own U-Net.
# ![image](static/UNet_figure.png)

# %% [markdown]
# ### Component 1: Upsampling

# %% [markdown]
# We will start with the Upsample module we will use in our U-Net. The right side of the U-Net contains upsampling between the layers. There are many ways to upsample: in the example above, they use a 2x2 transposed convolution, but we will use the PyTorch Upsample Module [torch.nn.Upsample](https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html#torch.nn.Upsample).


# %% [markdown]
# #### Pytorch Modules
# Modules are the building blocks of PyTorch models, and contain lots of magic that makes training models easy. If you aren't familiar with PyTorch modules, take a look at the official documentation [here](https://pytorch.org/docs/stable/notes/modules.html). For our purposes, it is crucial to note how Modules can have submodules defined in the `__init__` function, and how the `forward` function is defined and called.

# %%
# Here we make fake input to illustrate the upsampling techniques
# Pytorch expects a batch and channel dimension before the actual data,
# So this simulates a 1D input
sample_1d_input = torch.tensor([[[1,2,3,4]]], dtype=torch.float64)
# And this simulates a 2D input
sample_2d_input = torch.tensor([[[[1,2,],[3,4]]]], dtype=torch.float64)
sample_2d_input

# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Task 1:</b> Try out different upsampling techniques
#     <p>For our U-net, we will use the built-in PyTorch Upsample Module. Here we will practice declaring and calling an Upsample module with different parameters.</p>
#     <ol>
#         <li>Declare an instance of the pytorch Upsample module with scale_factor 2 and mode <code>"nearest"</code>. the Modules you want to use (in this case, <code>torch.nn.Upsample</code> with the correct arguments) in the <code>__init__</code> function.</li>
#         <li>Call the module's forward function on the <code>sample_2d_input</code> to see what the nearest mode does.</li> 
#         <li>Vary the scale factor and mode to see what changes. Check the documentation for possible modes and required input dimensions.</li>
#     </ol>
# </div>

# %%
up = torch.nn.Upsample(scale_factor=2, mode="nearest")   # need to keep scaffolding for up here
up(sample_2d_input)

# %% [markdown]
# Here is an additional example on image data.

# %%
apply_and_show_random_image(up)

# %% [markdown]
# ### Component 2: Downsampling

# %% [markdown]
# Between each layer of the U-Net on the left side, there is a downsample step. Traditionally, this is done with a 2x2 max pooling operation. There are other ways to downsample, for example with average pooling, but we will stick with max pooling for this exercise.


# %%
sample_2d_input = torch.tensor(np.arange(16, dtype=np.float64).reshape((1,1,4,4)))
sample_2d_input

# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Task 2.1:</b> Try out max pooling
#         <p>Using the docs for <a href=https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html>torch.nn.MaxPool2d</a>,
#         try it out in function form in the cell below. Try varying the stride and the padding, to see how the output changes.
#         </p>

# %%
max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
max_pool(sample_2d_input)


# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Task 2.2:</b> Implement a Downsample Module
#     <p>This is very similar to the built in MaxPool2D, but additionally has to check if the downsample factor matches in the input size. Note that we provide the forward function for you - in future Modules, you will implement the forward yourself.</p>
#     <ol>
#         <li>Declare the submodules you want to use (in this case, <code>torch.nn.MaxPool2D</code> with the correct arguments) in the <code>__init__</code> function. In our Downsample Module, we do not want to allow padding or strides other than the input kernel size.</li>
#         <li>Write a function to check if the downsample factor is valid. If the downsample factor does not evenly divide the dimensions of the input to the layer, this function should return False.</li>
#     </ol>
# </div>

# %%
class Downsample(torch.nn.Module):
    def __init__(self, downsample_factor: int):
        """TODO: Docstring"""

        super().__init__()

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
down = Downsample(4)
apply_and_show_random_image(down)

# %%
unet_tests.TestDown(Downsample).run()

# %% [markdown]
# ### Component 3: Convolution Block

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
# <div class="alert alert-block alert-info">
#     <b>Task 3:</b> Implement a ConvBlock module
#     <p>The convolution block (ConvBlock) of a standard U-Net has two 3x3 convolutions, each of which is followed by a ReLU activation. Our implementation will handle other sizes of convolutions as well. The first convolution in the block will handle changing the input number of feature maps/channels into the output, and the second convolution will have the same number of feature maps in and out.</p>
#     <ol>
#         <li>Declare the submodules you want to use in the <code>__init__</code> function. Because you will always be calling four submodules in sequence (<a href=https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d>torch.nn.Conv2D</a>, <a href=https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU>torch.nn.ReLU</a>, Conv2D, ReLU), you can use <a href=https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html>torch.nn.Sequential</a> to hold the convolutions and ReLUs.</li>
#         <li>Call the modules in the forward function. If you used <code>torch.nn.Sequential</code> in step 1, you only need to call the Sequential module, but if not, you can call the Conv2D and ReLU Modules explicitly.</li>
#     </ol>
# </div>
#
# If you get stuck, refer back to the <a href=https://pytorch.org/docs/stable/notes/modules.html>Module</a> documentation for hints and examples of how to define a PyTorch Module. 


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
        super().__init__()

        if kernel_size % 2 == 0:
            msg = "Only allowing odd kernel sizes."
            raise ValueError(msg)
        # determine padding size based on method
        if padding.upper() == "VALID":
            pad = 0  # compute this
        elif padding.upper() == "SAME":
            pad = kernel_size // 2  # compute this
        else:
            msg = "invalid string value for padding. Choose SAME or VALID."
            raise ValueError(msg)

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

        for _name, layer in self.named_modules():
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    def forward(self, x):
        return self.conv_pass(x) # leave out


# %% [markdown]
# #### Visualize Output of ConvBlock


# %%
torch.manual_seed(26)
conv = ConvBlock(1,2,5,"same")
apply_and_show_random_image(conv)


# %%
unet_tests.TestConvBlock(ConvBlock).run()

# %% [markdown]
# ### Component 4: Skip Connections and Concatenation

# %% [markdown]
# The skip connections between the left and right side of the U-Net are central to successfully obtaining high-resolution output. At each layer, the output of the left conv block is concatenated to the output of the upsample block on the right side from the last layer below. Since upsampling, especially with the "nearest" algorithm, does not actually add high resolution information, the concatenation of the right side conv block output is crucial to generate high resolution segmentations.
#
# If the convolutions in the U-Net are valid, the right side will be smaller than the left side, so the right side output must be cropped before concatenation. We provide a helper function to do this cropping. 
#
# TODO: Add test


# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Task 4:</b> Implement a CropAndConcat module
#     <p>Below, you must implement the forward algorithm, including the cropping (using the provided helper function <code>self.crop</code>) and the concatenation (using <a href=https://pytorch.org/docs/stable/generated/torch.cat.html#torch.cat>torch.cat</a>).
# </p>
# </div>

# %%
class CropAndConcat(torch.nn.Module):
    def crop(self, x, y):
        """Center-crop x to match spatial dimensions given by y."""

        x_target_size = x.size()[:-2] + y.size()[-2:]

        offset = tuple((a - b) // 2 for a, b in zip(x.size(), x_target_size))

        slices = tuple(slice(o, o + s) for o, s in zip(offset, x_target_size))

        return x[slices]

    def forward(self, encoder_output, upsample_output):
        encoder_cropped = self.crop(encoder_output, upsample_output)  # leave this out

        return torch.cat([encoder_cropped, upsample_output], dim=1)  # leave this out


# %%
unet_tests.TestCropAndConcat(CropAndConcat).run()

# %% [markdown]
# ### Component 5: Output Block

# %% [markdown]
# The final block we need to write for our U-Net is the output convolution block. The exact format of output you want depends on your task, so our U-Net must be flexible enough to handle different numbers of out channels and different final activation functions.


# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Task 5:</b> Implement an OutputConv Module
#     <ol>
#         <li>Define the convolution and final activation module in the <code>__init__</code> function. You can use a convolution with kernel size 1 to get the appropriate number of output channels.</li>
#         <li>Call the final convolution and activation modules in the <code>forward</code> function</li>
#     </ol>
# </div>

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
        Use a convolution with kernel size 1 to obtain the appropriate number of output channels. 
        Then apply final activation.
        """
        super().__init__()
        self.final_conv = torch.nn.Conv2d(in_channels, out_channels, 1, padding=0) # leave this out
        if activation is None:
            self.activation = None
        else:
            self.activation = getattr(torch.nn, activation)()

    def forward(self, x):
        x = self.final_conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


# %%
out_conv = OutputConv(in_channels=1, out_channels=1, activation="ReLU")
apply_and_show_random_image(out_conv)

# %%
unet_tests.TestOutputConv(OutputConv).run()

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
# Now we will make a U-Net class that combines all of these components as shown in the image. This image shows a U-Net of depth 5 with specific input channels, feature maps, upsampling, and final activation. Ours will be configurable with regards to depth and other features.
# ![image](static/UNet_figure.png)


# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Task 6:</b> U-Net Implementation
#     <ol>
#         <li>TODO: list what they need to implement specifically</li>
#         </ol>
# </div>

# %%
class UNet(torch.nn.Module):
    def __init__(
        self,
        depth: int,
        in_channels: int,
        out_channels: int = 1, 
        final_activation: str | None = None,
        num_fmaps: int = 64,
        fmap_inc_factor: int = 2,
        downsample_factor: int = 2,
        kernel_size: int = 3,
        padding: str = "same",
        upsample_mode: str = "nearest",
    ):
        """A U-Net for 2D input that expects tensors shaped like::
            ``(batch, channels, height, width)``.
        Args:
            depth:
                The number of levels in the U-Net. 2 is the smallest that really
                makes sense for the U-Net architecture, as a one layer U-Net is
                basically just 2 conv blocks.
            in_channels:
                The number of input channels in your dataset.
            out_channels (optional):
                How many output channels you want. Depends on your task. Defaults to 1.
            final_activation (optional):
                What activation to use in your final output block. Depends on your task.
                Defaults to None.
            num_fmaps (optional):
                The number of feature maps in the first layer. Defaults to 64.
            fmap_inc_factor (optional):
                By how much to multiply the number of feature maps between
                layers. Layer ``l`` will have ``num_fmaps*fmap_inc_factor**l`` 
                feature maps. Defaults to 2.
            downsample_factor (optional):
                Factor to use for down- and up-sampling the feature maps between layers.
                Defaults to 2.
            kernel_size (optional):
                Kernel size to use in convolutions on both sides of the UNet.
                Defaults to 3.
            padding (optional):
                How to pad convolutions. Either 'same' or 'valid'. Defaults to "same."
            upsample_mode (optional):
                The upsampling mode to pass to torch.nn.Upsample. Usually "nearest" 
                or "bilinear." Defaults to "nearest."
        """

        super().__init__()

        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_activation = final_activation
        self.num_fmaps = num_fmaps
        self.fmap_inc_factor = fmap_inc_factor
        self.downsample_factor = downsample_factor
        self.kernel_size = kernel_size
        self.padding = padding
        self.upsample_mode = upsample_mode

        # left convolutional passes
        self.left_convs = torch.nn.ModuleList()
        for level in range(self.depth):
            fmaps_in, fmaps_out = self.compute_fmaps_encoder(level)
            self.left_convs.append(
                ConvBlock(
                    fmaps_in,
                    fmaps_out,
                    self.kernel_size,
                    self.padding
                )
            )

        # right convolutional passes
        self.right_convs = torch.nn.ModuleList()
        for level in range(self.depth - 1):
            fmaps_in, fmaps_out = self.compute_fmaps_decoder(level)
            self.right_convs.append(
                ConvBlock(
                    fmaps_in,
                    fmaps_out,
                    self.kernel_size,
                    self.padding,
                )
            )
        
        self.downsample = Downsample(self.downsample_factor)
        self.upsample = torch.nn.Upsample(
                    scale_factor=self.downsample_factor,
                    mode=self.upsample_mode,
                )
        self.crop_and_concat = CropAndConcat()
        self.final_conv = OutputConv(
            self.compute_fmaps_decoder(0)[1], self.out_channels, self.final_activation
        )

    def compute_fmaps_encoder(self, level: int) -> tuple[int, int]:
        """Compute the number of input and output feature maps for 
        a conv block at a given level of the UNet encoder (left side). 

        Args:
            level (int): The level of the U-Net which we are computing
            the feature maps for. Level 0 is the input level, level 1 is
            the first downsampled layer, and level=depth - 1 is the bottom layer.

        Output (tuple[int, int]): The number of input and output feature maps
            of the encoder convolutional pass in the given level.
        """
        if level == 0:  # Leave out function
            fmaps_in = self.in_channels
        else:
            fmaps_in = self.num_fmaps * self.fmap_inc_factor ** (level - 1)

        fmaps_out = self.num_fmaps * self.fmap_inc_factor**level
        return fmaps_in, fmaps_out

    def compute_fmaps_decoder(self, level: int) -> tuple[int, int]:
        """Compute the number of input and output feature maps for a conv block
        at a given level of the UNet decoder (right side). Note:
        The bottom layer (depth - 1) is considered an "encoder" conv pass, 
        so this function is only valid up to depth - 2.
        
        Args:
            level (int): The level of the U-Net which we are computing
            the feature maps for. Level 0 is the input level, level 1 is
            the first downsampled layer, and level=depth - 1 is the bottom layer.

        Output (tuple[int, int]): The number of input and output feature maps
            of the encoder convolutional pass in the given level.
        """
        fmaps_out = self.num_fmaps * self.fmap_inc_factor ** (level)  # Leave out function
        concat_fmaps = self.compute_fmaps_encoder(level)[
            1
        ]  # The channels that come from the skip connection
        fmaps_in = concat_fmaps + self.num_fmaps * self.fmap_inc_factor ** (level + 1)

        return fmaps_in, fmaps_out

    def forward(self, x):
        # left side
        convolution_outputs = []
        layer_input = x
        for i in range(self.depth - 1):  # leave out center of for loop
            conv_out = self.left_convs[i](layer_input)
            convolution_outputs.append(conv_out)
            downsampled = self.downsample(conv_out)
            layer_input = downsampled

        # bottom
        conv_out = self.left_convs[-1](layer_input)
        layer_input = conv_out

        # right
        for i in range(0, self.depth-1)[::-1]:  # leave out center of for loop
            upsampled = self.upsample(layer_input)
            concat = self.crop_and_concat(convolution_outputs[i], upsampled)
            conv_output = self.right_convs[i](concat)
            layer_input = conv_output

        return self.final_conv(layer_input)


# %%
unet_tests.TestUNet(UNet).run()

# %%
simple_net = UNet(
        depth=2,
        in_channels=1,
        num_fmaps=12,
        fmap_inc_factor=3,
        downsample_factor=2,
        kernel_size=3,
        padding="valid",
        upsample_mode="nearest",)

# %%
apply_and_show_random_image(simple_net)

# %% [markdown]
# ### Receptive Field
#
# The receptive field of a U-Net is the set of input pixels that contribute to a specific output pixel.
#
# The receptive field of a single 3x3 convolution is simply the 3x3 grid of inputs. Each subsequent convolution adds the (kernel size - 1) to the receptive field, so two 3x3 convolutions have a 5x5 receptive field for each output pixel.
#
# Downsampling increases the receptive field as well. After 2x2 max pooling, a 3x3 convolution has a receptive field of 6x6 pre-downsampled pixels. Every operation further increases the receptive field, so the final receptive field of a U-Net output depends on the depth, kernel size, and downsample factor.
#
# The `plot_receptive_field` function visualizes the receptive field of a given U-Net - the square shows how many input pixels contribute to the output at the center pixel. Try it out with different U-Nets to get a sense of how varying the depth, kernel size, and downsample factor affect the receptive field of a U-Net.

# %%
from local import plot_receptive_field

new_net = UNet(
        depth=2,
        in_channels=1,
        downsample_factor=2,
        kernel_size=3,)
plot_receptive_field(new_net)

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
# ## Train a U-Net!
# We will get more into the details of evaluating semantic segmentation models in the next exercise. For now, we will provide an example pipeline that will train a U-Net to classify each pixel in an image of cells as foreground or background.

# %% [markdown]
# ### Dataset
# For our segmentation exercises, we will be using a nucleus segmentation dataset from [Kaggle 2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018/data). We have pre-downloaded the dataset to the shared drive and provded a pytorch Dataset called `NucleiDataset` to use for training. In this exercise, we will do semantic segmentation and train a model to classify foreground and background, but you will learn more about this task in the next exercise.
# Below, we visualize five examples of input data and foreground mask.

# %%
from torchvision import transforms

dataset = NucleiDataset("nuclei_train_data", transforms.RandomCrop(256))
for i in range(5):
    show_random_dataset_image(dataset)

# %%
train_loader = DataLoader(dataset)

# %% tags=["solution"]
loss_function: torch.nn.Module = torch.nn.MSELoss()


# %% tags=["solution"]
def crop(x, target):
    """Center-crop x to match spatial dimensions given by target."""

    x_target_size = x.size()[:-2] + target.size()[-2:]

    offset = tuple((a - b) // 2 for a, b in zip(x.size(), x_target_size))

    slices = tuple(slice(o, o + s) for o, s in zip(offset, x_target_size))

    return x[slices]
    
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


# %%
# start a tensorboard writer
logger = SummaryWriter("runs/Unet")
# %tensorboard --logdir runs

# %%
# Here is where students can define their own unet with whatever parameters they want to try,
# or use one of the examples from the thought exercise
model = UNet(depth=4,
        in_channels=1)

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
