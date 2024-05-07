# %% [markdown]
# # Build Your Own U-Net
#
# <hr style="height:2px;">
#
# In this notebook, we will build and train our own U-Net, with the goal of understanding the architecture and being able to use it for many different tasks in the rest of the course.
#
# Written by Larissa Heinrich, Caroline Malin-Mayor, and Morgan Schwartz, with inspiration from William Patton.

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

from local import (
    NucleiDataset,
    show_random_dataset_image,
    train,
    apply_and_show_random_image,
)
import unet_tests

# %%
# make sure gpu is available. Please call a TA if this cell fails
assert torch.cuda.is_available()


# %% [markdown]
# ## The Dataset
# For our segmentation exercises, we will be using a nucleus segmentation dataset from [Kaggle 2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018/data). We have pre-downloaded the dataset to the shared drive and provded a pytorch Dataset called `NucleiDataset` to use for training. In this exercise, we will create a U-Net and train it to classify foreground and background, but focus on the U-Net itself - you will learn more about semantic segmentation in the next exercise.
# Below, we create a dataset and then visualize a random image.

# %%
dataset = NucleiDataset("nuclei_train_data")

# %%
show_random_dataset_image(dataset)

# %% [markdown]
# Rerun the cell above a few times to see different images.

# %% [markdown] tags=[]
# <hr style="height:2px;">
#
# ## The Components of a U-Net

# %% [markdown] tags=[]
# The [U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) architecture has proven to outperform the other architectures in segmenting biological and medical images. It is also commonly used for other tasks that require the output to be the same resolution as the input, such as style transfer and denoising. Below is an overview figure of the U-Net architecture from the original [paper](https://arxiv.org/pdf/1505.04597.pdf). We will go through each of the components first (hint: all of them can be found in the list of PyTorch modules [here](https://pytorch.org/docs/stable/nn.html#convolution-layers)), and then fit them all together to make our very own U-Net.
# ![image](static/UNet_figure.png)

# %% [markdown] tags=[]
# ### Component 1: Upsampling

# %% [markdown] tags=[]
# We will start with the Upsample module that we will use in our U-Net. The right side of the U-Net contains upsampling between the layers. There are many ways to upsample: in the example above, they use a 2x2 transposed convolution, but we will use the PyTorch Upsample Module [torch.nn.Upsample](https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html#torch.nn.Upsample).


# %% [markdown] tags=[]
# #### Pytorch Modules
# Modules are the building blocks of PyTorch models, and contain lots of magic that makes training models easy. If you aren't familiar with PyTorch modules, take a look at the official documentation [here](https://pytorch.org/docs/stable/notes/modules.html). For our purposes, it is crucial to note how Modules can have submodules defined in the `__init__` function, and how the `forward` function is defined and called.

# %% tags=[]
# Here we make fake input to illustrate the upsampling techniques
# Pytorch expects a batch and channel dimension before the actual data,
# So this simulates a 1D input
sample_1d_input = torch.tensor([[[1, 2, 3, 4]]], dtype=torch.float64)
# And this simulates a 2D input
sample_2d_input = torch.tensor(
    [[[[1, 2], [3, 4]]]],
    dtype=torch.float64,
)
sample_2d_input.shape, sample_2d_input

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info">
#     <h4>Task 1: Try out different upsampling techniques</h4>
#     <p>For our U-net, we will use the built-in PyTorch Upsample Module. Here we will practice declaring and calling an Upsample module with different parameters.</p>
#     <ol>
#         <li>Declare an instance of the pytorch Upsample module with <code>scale_factor</code> 2 and mode <code>"nearest"</code>.</li>
#         <li>Call the instance of Upsample on the <code>sample_2d_input</code> to see what the nearest mode does.</li>
#         <li>Vary the scale factor and mode to see what changes. Check the documentation for possible modes and required input dimensions.</li>
#     </ol>
# </div>

# %% tags=[]
# TASK 1.1: initialize  an upsample module
up = ...  # YOUR CODE HERE

# TASK 1.2: apply your upsample module to `sample_2d_input`

# YOUR CODE HERE

# %% tags=[]
# TASK 1.3: vary scale factor and mode
# YOUR CODE HERE

# %% tags=["solution"]
# SOLUTION 1.1: initialize  an upsample module
up = torch.nn.Upsample(scale_factor=2, mode="nearest")

# SOLUTION 1.2: apply your upsample module to `sample_2d_input`
up(sample_2d_input)

# %% [markdown] tags=[]
# Here is an additional example on image data.

# %% tags=[]
apply_and_show_random_image(up, dataset)

# %% [markdown] tags=[]
# ### Component 2: Downsampling

# %% [markdown] tags=[]
# Between each layer of the U-Net on the left side, there is a downsample step. Traditionally, this is done with a 2x2 max pooling operation. There are other ways to downsample, for example with average pooling, but we will stick with max pooling for this exercise.


# %% tags=[]
sample_2d_input = torch.tensor(np.arange(16, dtype=np.float64).reshape((1, 1, 4, 4)))
sample_2d_input

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info">
#     <h4>Task 2A: Try out max pooling</h4>
#         <p>Using the docs for <a href=https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html>torch.nn.MaxPool2d</a>,
#         try initializing the module and applying it to the sample input. Try varying the parameters to see how the output changes.
#         </p>

# %% tags=[]
# TASK 2A: Initialize max pooling and apply to sample input
# YOUR CODE HERE

# %% tags=["solution"]
# SOLUTION 2A: Initialize max pooling and apply to sample input
max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
max_pool(sample_2d_input)


# %% [markdown] tags=[]
# <div class="alert alert-block alert-info">
#     <h4>Task 2B: Implement a Downsample Module</h4>
#     <p>This is very similar to the built in MaxPool2d, but additionally has to check if the downsample factor matches in the input size. Note that we provide the forward function for you - in future Modules, you will implement the forward yourself.</p>
#     <ol>
#         <li>Declare the submodules you want to use (in this case, <code>torch.nn.MaxPool2d</code> with the correct arguments) in the <code>__init__</code> function. In our Downsample Module, we do not want to use padding and the stride should match the input kernel size.</li>
#         <li>Write a function to check if the downsample factor is valid. If the downsample factor does not evenly divide the dimensions of the input to the layer, this function should return False.</li>
#     </ol>
# </div>


# %% tags=[]
class Downsample(torch.nn.Module):
    def __init__(self, downsample_factor: int):
        """Initialize a MaxPool2d module with the input downsample fator"""

        super().__init__()

        self.downsample_factor = downsample_factor
        # TASK 2B1: Initialize the downsample module
        self.down = ...  # YOUR CODE HERE

    def check_valid(self, image_size: tuple[int, int]) -> bool:
        """Check if the downsample factor evenly divides each image dimension
        Note: there are multiple ways to do this!
        """
        # TASK 2B2: Check that the image_size is valid to use with the downsample factor
        # YOUR CODE HERE

    def forward(self, x):
        if not self.check_valid(tuple(x.size()[-2:])):
            raise RuntimeError(
                "Can not downsample shape %s with factor %s"
                % (x.size(), self.downsample_factor)
            )

        return self.down(x)


# %% tags=["solution"]
class Downsample(torch.nn.Module):
    def __init__(self, downsample_factor: int):
        """Initialize a MaxPool2d module with the input downsample fator"""

        super().__init__()

        self.downsample_factor = downsample_factor
        # SOLUTION 2B1: Initialize the downsample module
        self.down = torch.nn.MaxPool2d(downsample_factor)

    def check_valid(self, image_size: tuple[int, int]) -> bool:
        """Check if the downsample factor evenly divides each image dimension
        Note: there are multiple ways to do this!
        """
        # SOLUTION 2B2: Check that the image_size is valid to use with the downsample factor
        for dim in image_size:
            if dim % self.downsample_factor != 0:
                return False
        return True

    def forward(self, x):
        if not self.check_valid(tuple(x.size()[-2:])):
            raise RuntimeError(
                "Can not downsample shape %s with factor %s"
                % (x.size(), self.downsample_factor)
            )

        return self.down(x)


# %% tags=[]
down = Downsample(4)
apply_and_show_random_image(down, dataset)

# %% [markdown] tags=[]
# We wrote some rudimentary tests for each of the torch modules you are writing. If you get an error from your code or an AssertionError from the test, you should probably have another look ath your implementation.

# %% tags=[]
unet_tests.TestDown(Downsample).run()

# %% [markdown] tags=[]
# ### Component 3: Convolution Block

# %% [markdown]
# #### Convolution
# A U-Net is a convolutional neural network, which means that the main type of operation is a convolution. Convolutions with defined kernels were covered briefly in the pre-course materials.
#
# <img src="./static/2D_Convolution_Animation.gif" width="400" height="300">


# %% [markdown] tags=[]
# Shown here is a 3x3 kernel being convolved with an input array to get an output array of the same size. For each pixel of the input, the value at that same pixel of the output is computed by multiplying the kernel element-wise with the surrounding 3x3 neighborhood around the input pixel, and then summing the result.


# %% [markdown] tags=[]
# #### Padding
#
# You will notice that at the edges of the input, this animation shows greyed out values that extend past the input. This is known as padding the input. This example uses "same" padding, which means the values at the edges are repeated. The other option we will use in this exercise is "valid" padding, which essentially means no padding. In the case of valid padding, the output will be smaller than the input, as values at the edges of the output will not be computed. "Same" padding can introduce edge artifacts, but "valid" padding reduces the output size at every convolution. Note that the amount of padding (for same) and the amount of size lost (for valid) depends on the size of the kernel - a 3x3 convolution would require padding of 1, a 5x5 convolution would require a padding of 2, and so on.


# %% [markdown] tags=[]
# #### ReLU Activation
# The Rectified Linear Unit (ReLU) is a common activation function, which takes the max of a value and 0, shown below. It introduces a non-linearity into the neural network - without a non-linear activation function, a neural network could not learn non-linear functions.
#
# <img src="./static/ReLU.png" width="400" height="300">

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info">
#     <h4>Task 3: Implement a ConvBlock module</h4>
#     <p>The convolution block (ConvBlock) of a standard U-Net has two 3x3 convolutions, each of which is followed by a ReLU activation. Our implementation will handle other sizes of convolutions as well. The first convolution in the block will handle changing the input number of feature maps/channels into the output, and the second convolution will have the same number of feature maps in and out.</p>
#     <ol>
#         <li>Calculate the amount of padding required for same and valid convolutions</li>
#         <li>Declare the submodules you want to use in the <code>__init__</code> function. Because you will always be calling four submodules in sequence (<a href=https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d>torch.nn.Conv2d</a>, <a href=https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU>torch.nn.ReLU</a>, Conv2d, ReLU), you can use <a href=https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html>torch.nn.Sequential</a> to hold the convolutions and ReLUs.</li>
#         <li>Call the modules in the forward function. If you used <code>torch.nn.Sequential</code> in step 1, you only need to call the Sequential module, but if not, you can call the Conv2d and ReLU Modules explicitly.</li>
#     </ol>
# </div>
#
# If you get stuck, refer back to the <a href=https://pytorch.org/docs/stable/notes/modules.html>Module</a> documentation for hints and examples of how to define a PyTorch Module.


# %% tags=[]
class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: str = "same",
    ):
        """A convolution block for a U-Net. Contains two convolutions, each followed by a ReLU.

        Args:
            in_channels (int): The number of input channels for this conv block. Depends on
                the layer and side of the U-Net and the hyperparameters.
            out_channels (int): The number of output channels for this conv block. Depends on
                the layer and side of the U-Net and the hyperparameters.
            kernel_size (int): The size of the kernel. A kernel size of N signifies an
                NxN square kernel.
            padding (str): The type of convolution padding to use. Either "same" or "valid"
        """
        super().__init__()

        if kernel_size % 2 == 0:
            msg = "Only allowing odd kernel sizes."
            raise ValueError(msg)

        # TASK 3.1: Determine padding size based on method
        if padding.upper() == "VALID":
            pad = ...  # YOUR CODE HERE
        elif padding.upper() == "SAME":
            pad = ...  # YOUR CODE HERE
        else:
            msg = "invalid string value for padding. Choose SAME or VALID."
            raise ValueError(msg)

        # TASK 3.2: Initialize your modules and define layers in conv pass
        # YOUR CODE HERE

        for _name, layer in self.named_modules():
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    def forward(self, x):
        # TASK 3.3: Apply the modules you defined to the input x
        ...  # YOUR CODE HERE


# %% tags=["solution"]
class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: str = "same",
    ):
        """A convolution block for a U-Net. Contains two convolutions, each followed by a ReLU.

        Args:
            in_channels (int): The number of input channels for this conv block. Depends on
                the layer and side of the U-Net and the hyperparameters.
            out_channels (int): The number of output channels for this conv block. Depends on
                the layer and side of the U-Net and the hyperparameters.
            kernel_size (int): The size of the kernel. A kernel size of N signifies an
                NxN square kernel.
            padding (str): The type of convolution padding to use. Either "same" or "valid"
        """
        super().__init__()

        if kernel_size % 2 == 0:
            msg = "Only allowing odd kernel sizes."
            raise ValueError(msg)

        # SOLUTION 3.1: Determine padding size based on method
        if padding.upper() == "VALID":
            pad = 0  # compute this
        elif padding.upper() == "SAME":
            pad = kernel_size // 2  # compute this
        else:
            msg = "invalid string value for padding. Choose SAME or VALID."
            raise ValueError(msg)

        # SOLUTION 3.2: Initialize your modules and define layers in conv pass
        self.conv_pass = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, padding=pad
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                out_channels, out_channels, kernel_size=kernel_size, padding=pad
            ),
            torch.nn.ReLU(),
        )

        for _name, layer in self.named_modules():
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    def forward(self, x):
        # SOLUTION 3.3: Apply the modules you defined to the input x
        return self.conv_pass(x)


# %% [markdown] tags=[]
# #### Test and Visualize Output of ConvBlock


# %% tags=[]
unet_tests.TestConvBlock(ConvBlock).run()

# %% tags=[]
torch.manual_seed(26)
conv = ConvBlock(1, 2, 5, "same")
apply_and_show_random_image(conv, dataset)


# %% [markdown] tags=[]
# ### Component 4: Skip Connections and Concatenation

# %% [markdown] tags=[]
# The skip connections between the left and right side of the U-Net are central to successfully obtaining high-resolution output. At each layer, the output of the left conv block is concatenated to the output of the upsample block on the right side from the last layer below. Since upsampling, especially with the "nearest" algorithm, does not actually add high resolution information, the concatenation of the right side conv block output is crucial to generate high resolution segmentations.
#
# If the convolutions in the U-Net are valid, the right side will be smaller than the left side, so the right side output must be cropped before concatenation. We provide a helper function to do this cropping.


# %% [markdown] tags=[]
# <div class="alert alert-block alert-info">
#     <h4>Task 4: Implement a CropAndConcat module</h4>
#     <p>Below, you must implement the <code>forward</code> method, including the cropping (using the provided helper function <code>self.crop</code>) and the concatenation (using <a href=https://pytorch.org/docs/stable/generated/torch.cat.html#torch.cat>torch.cat</a>).
# </p>
# Hint: Use the <code>dim</code> keyword argument of <a href=https://pytorch.org/docs/stable/generated/torch.cat.html#torch.cat>torch.cat</a> to choose along which axis to concatenate the tensors.
# </p>
# Hint: The tensors have the layout (batch, channel, x, y)
# </div>


# %% tags=[]
class CropAndConcat(torch.nn.Module):
    def crop(self, x, y):
        """Center-crop x to match spatial dimensions given by y."""

        x_target_size = x.size()[:-2] + y.size()[-2:]

        offset = tuple((a - b) // 2 for a, b in zip(x.size(), x_target_size))

        slices = tuple(slice(o, o + s) for o, s in zip(offset, x_target_size))

        return x[slices]

    def forward(self, encoder_output, upsample_output):
        # TASK 4: Implement the forward function
        ...


# %% tags=["solution"]
class CropAndConcat(torch.nn.Module):
    def crop(self, x, y):
        """Center-crop x to match spatial dimensions given by y."""

        x_target_size = x.size()[:-2] + y.size()[-2:]

        offset = tuple((a - b) // 2 for a, b in zip(x.size(), x_target_size))

        slices = tuple(slice(o, o + s) for o, s in zip(offset, x_target_size))

        return x[slices]

    def forward(self, encoder_output, upsample_output):
        # SOLUTION 4: Implement the forward function
        encoder_cropped = self.crop(encoder_output, upsample_output)

        return torch.cat([encoder_cropped, upsample_output], dim=1)


# %% tags=[]
unet_tests.TestCropAndConcat(CropAndConcat).run()

# %% [markdown] tags=[]
# ### Component 5: Output Block

# %% [markdown] tags=[]
# The final block we need to write for our U-Net is the output convolution block. The exact format of output you want depends on your task, so our U-Net must be flexible enough to handle different numbers of out channels and different final activation functions.


# %% [markdown] tags=[]
# <div class="alert alert-block alert-info">
#     <h4>Task 5: Implement an OutputConv Module</h4>
#     <ol>
#         <li>Define the convolution module in the <code>__init__</code> function. You can use a convolution with kernel size 1 to get the appropriate number of output channels. The activation submodule is provided for you.</li>
#         <li>Call the final convolution and activation modules in the <code>forward</code> function</li>
#     </ol>
# </div>


# %% tags=[]
class OutputConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str | None = None,
    ):
        """
        A module that uses a convolution with kernel size 1 to get the appropriate
        number of output channels, and then optionally applies a final activation.

        Args:
            in_channels (int): The number of feature maps that will be input to the
                OutputConv block.
            out_channels (int): The number of channels that you want in the output
            activation (str | None, optional): Accepts the name of any torch activation
                function  (e.g., ``ReLU`` for ``torch.nn.ReLU``) or None for no final
                activation. Defaults to None.
        """
        super().__init__()

        # TASK 5.1: Define the convolution submodule
        # YOUR CODE HERE

        if activation is None:
            self.activation = None
        else:
            self.activation = getattr(torch.nn, activation)()

    def forward(self, x):
        # TASK 5.2: Implement the forward function
        # YOUR CODE HERE
        ...


# %% tags=["solution"]
class OutputConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str | None = None,
    ):
        """
        A module that uses a convolution with kernel size 1 to get the appropriate
        number of output channels, and then optionally applies a final activation.

        Args:
            in_channels (int): The number of feature maps that will be input to the
                OutputConv block.
            out_channels (int): The number of channels that you want in the output
            activation (str | None, optional): Accepts the name of any torch activation
                function  (e.g., ``ReLU`` for ``torch.nn.ReLU``) or None for no final
                activation. Defaults to None.
        """
        super().__init__()

        # SOLUTION 5.1: Define the convolution submodule
        self.final_conv = torch.nn.Conv2d(in_channels, out_channels, 1, padding=0)

        if activation is None:
            self.activation = None
        else:
            self.activation = getattr(torch.nn, activation)()

    def forward(self, x):
        # SOLUTION 5.2: Implement the forward function
        x = self.final_conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


# %% tags=[]
unet_tests.TestOutputConv(OutputConv).run()

# %% tags=[]
out_conv = OutputConv(in_channels=1, out_channels=1, activation="ReLU")
apply_and_show_random_image(out_conv, dataset)

# %% [markdown] tags=[]
# <div class="alert alert-block alert-success">
#     <h2>Checkpoint 1</h2>
#
# Congratulations! You have implemented most of a U-Net!
# We will go over this portion together and answer any questions soon, but feel free to start on the next section where we put it all together.
#
# </div>
#
# <hr style="height:2px;">

# %% [markdown] tags=[]
# ## Putting the U-Net together
#
# Now we will make a U-Net class that combines all of these components as shown in the image. This image shows a U-Net of depth 5 with specific input channels, feature maps, upsampling, and final activation. Ours will be configurable with regards to depth and other features.
# ![image](static/UNet_figure.png)


# %% [markdown] tags=[]
# <div class="alert alert-block alert-info">
#     <h4>Task 6: U-Net Implementation</h4>
#     <p>Now we will implement our U-Net! We have written some of it for you - follow the steps below to fill in the missing parts.</p>
#     <ol>
#         <li>Write the helper functions <code>compute_fmaps_encoder</code> and <code>compute_fmaps_decoder</code> that compute the number of input and output feature maps at each level of the U-Net.</li>
#         <li>Declare a list of encoder (left) and decoder (right) ConvPasses depending on your depth using the helper functions you wrote above. Consider the special case at the bottom of the U-Net carefully!</li>
#         <li>Declare an Upsample, Downsample, CropAndConcat, and OutputConv block.</li>
#         <li>Implement the <code>forward</code> function, applying the modules you declared above in the proper order.</li>
#         </ol>
# </div>


# %% tags=[]
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
                layers. Encoder layer ``l`` will have ``num_fmaps*fmap_inc_factor**l``
                output feature maps. Defaults to 2.
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
        # TASK 6.2A: Initialize list here

        # right convolutional passes
        self.right_convs = torch.nn.ModuleList()
        # TASK 6.2B: Initialize list here

        # TASK 6.3: Initialize other modules here

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
        # TASK 6.1A: Implement this function
        pass

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
        # TASK 6.1B: Implement this function
        pass

    def forward(self, x):
        # left side
        # Hint - you will need the outputs of each encoder layer later
        convolution_outputs = []
        layer_input = x
        for i in range(self.depth - 1):
            # TASK 6.4A: Implement encoder here
            ...

        # bottom
        # TASK 6.4B: Implement bottom of U-Net here

        # right
        for i in range(0, self.depth - 1)[::-1]:
            # TASK 6.4C: Implement decoder here
            ...

        return  # TASK 6.4D: Return the final output


# %% tags=["solution"]
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
                layers. Encoder layer ``l`` will have ``num_fmaps*fmap_inc_factor**l``
                output feature maps. Defaults to 2.
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
        # SOLUTION 6.2A: Initialize list here
        for level in range(self.depth):
            fmaps_in, fmaps_out = self.compute_fmaps_encoder(level)
            self.left_convs.append(
                ConvBlock(fmaps_in, fmaps_out, self.kernel_size, self.padding)
            )

        # right convolutional passes
        self.right_convs = torch.nn.ModuleList()
        # SOLUTION 6.2B: Initialize list here
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

        # SOLUTION 6.3: Initialize other modules here
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
        # SOLUTION 6.1A: Implement this function
        if level == 0:
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
            of the decoder convolutional pass in the given level.
        """
        # SOLUTION 6.1B: Implement this function
        fmaps_out = self.num_fmaps * self.fmap_inc_factor ** (level)
        concat_fmaps = self.compute_fmaps_encoder(level)[
            1
        ]  # The channels that come from the skip connection
        fmaps_in = concat_fmaps + self.num_fmaps * self.fmap_inc_factor ** (level + 1)

        return fmaps_in, fmaps_out

    def forward(self, x):
        # left side
        convolution_outputs = []
        layer_input = x
        for i in range(self.depth - 1):
            # SOLUTION 6.4A: Implement encoder here
            conv_out = self.left_convs[i](layer_input)
            convolution_outputs.append(conv_out)
            downsampled = self.downsample(conv_out)
            layer_input = downsampled

        # bottom
        # SOLUTION 6.4B: Implement bottom of U-Net here
        conv_out = self.left_convs[-1](layer_input)
        layer_input = conv_out

        # right
        for i in range(0, self.depth - 1)[::-1]:
            # SOLUTION 6.4C: Implement decoder here
            upsampled = self.upsample(layer_input)
            concat = self.crop_and_concat(convolution_outputs[i], upsampled)
            conv_output = self.right_convs[i](concat)
            layer_input = conv_output

        # SOLUTION 6.4D: Return the final output
        return self.final_conv(layer_input)


# %% [markdown] tags=[]
# Below we declare a very simple U-Net and then apply it to a random image. Because we have not trained the U-Net the output should look similar to the output of random convolutions. If you get errors here, go back and fix your U-Net implementation!

# %% tags=[]
unet_tests.TestUNet(UNet).run()

# %% tags=[]
simple_net = UNet(depth=2, in_channels=1)

# %% tags=[]
apply_and_show_random_image(simple_net, dataset)

# %% [markdown] tags=[]
# ### Receptive Field
#
# The receptive field of a U-Net is the set of input pixels that contribute to a specific output pixel.
#
# The receptive field of a single 3x3 convolution is simply the 3x3 grid of inputs. Each subsequent convolution adds the kernel size - 1 to the receptive field, so two 3x3 convolutions have a 5x5 receptive field for each output pixel.
#
# Downsampling increases the receptive field as well. After 2x2 max pooling, a 3x3 convolution has a receptive field of 6x6 pre-downsampled pixels. Every operation further increases the receptive field, so the final receptive field of a U-Net output depends on the depth, kernel size, and downsample factor.

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info">
#     <h4>Task 7: Receptive Field</h4>
#     <p>The <code>plot_receptive_field</code> function visualizes the receptive field of a given U-Net - the square shows how many input pixels contribute to the output at the center pixel. Try it out with different U-Nets to get a sense of how varying the depth, kernel size, and downsample factor affect the receptive field of a U-Net.</p>
# </div>

# %% tags=[]
from local import plot_receptive_field

new_net = ...  # TASK 7: declare your U-Net here
if isinstance(new_net, UNet):
    plot_receptive_field(new_net)

# %% tags=["solution"]
from local import plot_receptive_field

# SOLUTION 7: declare your U-Net here
new_net = UNet(
    depth=2,
    in_channels=1,
    downsample_factor=2,
    kernel_size=3,
)
if isinstance(new_net, UNet):
    plot_receptive_field(new_net)

# %% [markdown] tags=[]
# <div class="alert alert-block alert-success">
#     <h2>Checkpoint 2</h2>
#
# We are about ready to start training! But before we do, we will stop and discuss your implementation.
#
# Questions to consider:
# <ol>
#     <li>Which feature of the U-Net has most effect on the receptive field?</li>
#     <li>For each hyperparameter: Can you think of scenarios in which you would consider changing this parameter? Why? </li>
# </ol>
#
# </div>
#
# <hr style="height:2px;">

# %% [markdown] tags=[]
# ## Train a U-Net!
# We will get more into the details of evaluating semantic segmentation models in the next exercise. For now, we will provide an example pipeline that will train a U-Net to classify each pixel in an image of cells as foreground or background. You should have seen training loops like this in the pre-course exercises, so there is no implementation task here.
#
# We will use Tensorboard to log our training runs, so there is code in the train loop to log aspects of the training to Tensorboard.

# %% tags=[]
train_dataset = NucleiDataset("nuclei_train_data", transforms.RandomCrop(256))
train_loader = DataLoader(train_dataset)

# %% tags=[]
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
        # if necessary, crop the masks to match the model output shape
        if prediction.shape != y.shape:
            y = crop(y, prediction)
        if y.dtype != prediction.dtype:
            y = y.type(prediction.dtype)
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


# %% [markdown] tags=[]
# Now we start our Tensorboard server and show it inside the notebook. We haven't logged anything yet so you'll see "No dashboards are active for the current data set." After you start training, you may need to press the refresh button in the top right to see the logs. When looking at the loss, we recommend smoothing (slider in the settings toolbar) because the raw numbers are very noisy.

# %% tags=[]
# start a tensorboard writer
# %tensorboard --logdir unet_runs

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info">
#     <h3>Task 8: Declare and train a U-Net</h3>
#     <ol>
#         <li>Declare a U-Net with your favorite parameters!</li>
#         <li>Train the U-Net for 5 epochs, and inspect the training loss and the output predictions in Tensorboard.</li>
#         </ol>
# </div>

# %% tags=[]
model = ...  # TASK 8.1: Declare your U-Net here and name it below
model_name = "my_fav_unet"  # This name will be used in the tensorboard logs
logger = SummaryWriter(f"unet_runs/{model_name}")

# %% tags=["solution"]
model = UNet(
    depth=4, in_channels=1
)  # SOLUTION 8.1: Declare your U-Net here and name it below
model_name = "my_fav_unet"  # This name will be used in the tensorboard logs
logger = SummaryWriter(f"unet_runs/{model_name}")

# %% tags=[]
# use adam optimizer
optimizer = torch.optim.Adam(model.parameters())

# train for $5$ epochs
# TASK 8.2: Run this cell and examine outputs in Tensorboard above!
n_epochs = 5
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


# %% [markdown] tags=[]
# <div class="alert alert-block alert-info">
#     <h3>Task 9: Train more U-Nets! </h3>
#     <ol>
#         <li>Declare a new U-Net with different parameters. If you use a different name in the Tensorboard logger, you will be able to know which logs are from which run.</li>
#         <li>Train your new U-Net - does it perform better or worse? How can you tell?</li>
#         <li>Keep tinkering with different architectures until the end of the exercise!</li>
#         </ol>
# </div>

# %% [markdown] tags=[]
# <div class="alert alert-block alert-success">
#     <h2>Checkpoint 3</h2>
#
# This is the end of the guided exercise. We will go over all of the code up until this point shortly. While you wait you are encouraged to try different U-Nets, training epochs, etc.
# </div>
