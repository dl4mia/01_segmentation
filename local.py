import os
import imageio
import matplotlib.pyplot as plt
from matplotlib import gridspec, ticker
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from mpl_toolkits.axes_grid1 import make_axes_locatable

from skimage.segmentation import relabel_sequential
from scipy.optimize import linear_sum_assignment


def show_one_image(image_path):
    image = imageio.imread(image_path)
    plt.imshow(image)


class NucleiDataset(Dataset):
    """A PyTorch dataset to load cell images and nuclei masks"""

    def __init__(self, root_dir, transform=None, img_transform=None):
        self.root_dir = (
            "/group/dl4miacourse/segmentation/" + root_dir
        )  # the directory with all the training samples
        self.samples = os.listdir(self.root_dir)  # list the samples
        self.transform = (
            transform  # transformations to apply to both inputs and targets
        )

        self.img_transform = img_transform  # transformations to apply to raw image only
        #  transformations to apply just to inputs
        inp_transforms = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),  # 0.5 = mean and 0.5 = variance
            ]
        )

        self.loaded_imgs = [None] * len(self.samples)
        self.loaded_masks = [None] * len(self.samples)
        for sample_ind in range(len(self.samples)):
            img_path = os.path.join(
                self.root_dir, self.samples[sample_ind], "image.tif"
            )
            image = Image.open(img_path)
            image.load()
            self.loaded_imgs[sample_ind] = inp_transforms(image)
            mask_path = os.path.join(
                self.root_dir, self.samples[sample_ind], "mask.tif"
            )
            mask = Image.open(mask_path)
            mask.load()
            self.loaded_masks[sample_ind] = transforms.ToTensor()(mask)

    # get the total number of samples
    def __len__(self):
        return len(self.samples)

    # fetch the training sample given its index
    def __getitem__(self, idx):
        # we'll be using Pillow library for reading files
        # since many torchvision transforms operate on PIL images
        image = self.loaded_imgs[idx]
        mask = self.loaded_masks[idx]
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
    axarr[0].set_title("Image")
    axarr[1].imshow(mask[0], interpolation=None)  # show the masks
    axarr[1].set_title("Mask")
    _ = [ax.axis("off") for ax in axarr]  # remove the axes
    print("Image size is %s" % {img[0].shape})
    plt.show()


def show_random_dataset_image_with_prediction(dataset, model, device="cpu"):
    idx = np.random.randint(0, len(dataset))  # take a random sample
    img, mask = dataset[idx]  # get the image and the nuclei masks
    x = img.to(device).unsqueeze(0)
    y = model(x)[0].detach().cpu().numpy()
    print("MSE loss:", np.mean((mask[0].numpy() - y[0]) ** 2))
    f, axarr = plt.subplots(1, 3)  # make two plots on one figure
    axarr[0].imshow(img[0])  # show the image
    axarr[0].set_title("Image")
    axarr[1].imshow(mask[0], interpolation=None)  # show the masks
    axarr[1].set_title("Mask")
    axarr[2].imshow(y[0], interpolation=None)  # show the prediction
    axarr[2].set_title("Prediction")
    _ = [ax.axis("off") for ax in axarr]  # remove the axes
    print("Image size is %s" % {img[0].shape})
    plt.show()


def show_random_augmentation_comparison(dataset_a, dataset_b):
    assert len(dataset_a) == len(dataset_b)
    idx = np.random.randint(0, len(dataset_a))  # take a random sample
    img_a, mask_a = dataset_a[idx]  # get the image and the nuclei masks
    img_b, mask_b = dataset_b[idx]  # get the image and the nuclei masks
    f, axarr = plt.subplots(2, 2)  # make two plots on one figure
    axarr[0, 0].imshow(img_a[0])  # show the image
    axarr[0, 0].set_title("Image")
    axarr[0, 1].imshow(mask_a[0], interpolation=None)  # show the masks
    axarr[0, 1].set_title("Mask")
    axarr[1, 0].imshow(img_b[0])  # show the image
    axarr[1, 0].set_title("Augmented Image")
    axarr[1, 1].imshow(mask_b[0], interpolation=None)  # show the prediction
    axarr[1, 1].set_title("Augmented Mask")
    _ = [ax.axis("off") for ax in axarr.flatten()]  # remove the axes
    plt.show()


def apply_and_show_random_image(f, ds):

    # pick random raw image from dataset
    img_tensor = ds[np.random.randint(len(ds))][0]

    batch_tensor = torch.unsqueeze(
        img_tensor, 0
    )  # add batch dimension that some torch modules expect
    out_tensor = f(batch_tensor)  # apply torch module
    out_tensor = out_tensor.squeeze(0)  # remove batch dimension
    img_arr = img_tensor.numpy()[0]  # turn into numpy array, look at first channel
    out_arr = out_tensor.detach().numpy()[
        0
    ]  # turn into numpy array, look at first channel

    # intialilze figure
    fig, axs = plt.subplots(1, 2, figsize=(10, 20))

    # Show input image, add info and colorbar
    img_min, img_max = (img_arr.min(), img_arr.max())  # get value range
    inim = axs[0].imshow(img_arr, vmin=img_min, vmax=img_max)
    axs[0].set_title("Input Image")
    axs[0].set_xlabel(f"min: {img_min:.2f}, max: {img_max:.2f}, shape: {img_arr.shape}")
    div = make_axes_locatable(axs[0])
    cb = fig.colorbar(inim, cax=div.append_axes("right", size="5%", pad=0.05))
    cb.outline.set_visible(False)

    # Show ouput image, add info and colorbar
    out_min, out_max = (out_arr.min(), out_arr.max())  # get value range
    outim = axs[1].imshow(out_arr, vmin=out_min, vmax=out_max)
    axs[1].set_title("First Channel of Output")
    axs[1].set_xlabel(f"min: {out_min:.2f}, max: {out_max:.2f}, shape: {out_arr.shape}")
    div = make_axes_locatable(axs[1])
    cb = fig.colorbar(outim, cax=div.append_axes("right", size="5%", pad=0.05))
    cb.outline.set_visible(False)

    # center images and remove ticks
    max_bounds = [
        max(ax.get_ybound()[1] for ax in axs),
        max(ax.get_xbound()[1] for ax in axs),
    ]
    for ax in axs:
        diffy = abs(ax.get_ybound()[1] - max_bounds[0])
        diffx = abs(ax.get_xbound()[1] - max_bounds[1])
        ax.set_ylim([ax.get_ybound()[0] - diffy / 2.0, max_bounds[0] - diffy / 2.0])
        ax.set_xlim([ax.get_xbound()[0] - diffx / 2.0, max_bounds[1] - diffx / 2.0])
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

    fig = plt.figure(figsize=(5, 5))
    plt.imshow(img_arr)  # , cmap='gray')

    # visualize receptive field
    xmin = img_arr.shape[1] / 2 - fov / 2
    xmax = img_arr.shape[1] / 2 + fov / 2
    ymin = img_arr.shape[0] / 2 - fov / 2
    ymax = img_arr.shape[0] / 2 + fov / 2
    color = "red"
    plt.hlines(ymin, xmin, xmax, color=color, lw=3)
    plt.hlines(ymax, xmin, xmax, color=color, lw=3)
    plt.vlines(xmin, ymin, ymax, color=color, lw=3)
    plt.vlines(xmax, ymin, ymax, color=color, lw=3)
    plt.show()


def compute_affinities(seg: np.ndarray, nhood: list):

    nhood = np.array(nhood)

    shape = seg.shape
    n_edges = nhood.shape[0]
    dims = nhood.shape[1]
    affinity = np.zeros((n_edges,) + shape, dtype=np.int32)

    for e in range(n_edges):
        affinity[
            e,
            max(0, -nhood[e, 0]) : min(shape[0], shape[0] - nhood[e, 0]),
            max(0, -nhood[e, 1]) : min(shape[1], shape[1] - nhood[e, 1]),
        ] = (
            (
                seg[
                    max(0, -nhood[e, 0]) : min(shape[0], shape[0] - nhood[e, 0]),
                    max(0, -nhood[e, 1]) : min(shape[1], shape[1] - nhood[e, 1]),
                ]
                == seg[
                    max(0, nhood[e, 0]) : min(shape[0], shape[0] + nhood[e, 0]),
                    max(0, nhood[e, 1]) : min(shape[1], shape[1] + nhood[e, 1]),
                ]
            )
            * (
                seg[
                    max(0, -nhood[e, 0]) : min(shape[0], shape[0] - nhood[e, 0]),
                    max(0, -nhood[e, 1]) : min(shape[1], shape[1] - nhood[e, 1]),
                ]
                > 0
            )
            * (
                seg[
                    max(0, nhood[e, 0]) : min(shape[0], shape[0] + nhood[e, 0]),
                    max(0, nhood[e, 1]) : min(shape[1], shape[1] + nhood[e, 1]),
                ]
                > 0
            )
        )

    return affinity


def evaluate(gt_labels: np.ndarray, pred_labels: np.ndarray, th: float = 0.5):
    """Function to evaluate a segmentation."""

    pred_labels_rel, _, _ = relabel_sequential(pred_labels)
    gt_labels_rel, _, _ = relabel_sequential(gt_labels)

    overlay = np.array([pred_labels_rel.flatten(), gt_labels_rel.flatten()])

    # get overlaying cells and the size of the overlap
    overlay_labels, overlay_labels_counts = np.unique(
        overlay, return_counts=True, axis=1
    )
    overlay_labels = np.transpose(overlay_labels)

    # get gt cell ids and the size of the corresponding cell
    gt_labels_list, gt_counts = np.unique(gt_labels_rel, return_counts=True)
    gt_labels_count_dict = {}

    for l, c in zip(gt_labels_list, gt_counts):
        gt_labels_count_dict[l] = c

    # get pred cell ids
    pred_labels_list, pred_counts = np.unique(pred_labels_rel, return_counts=True)

    pred_labels_count_dict = {}
    for l, c in zip(pred_labels_list, pred_counts):
        pred_labels_count_dict[l] = c

    num_pred_labels = int(np.max(pred_labels_rel))
    num_gt_labels = int(np.max(gt_labels_rel))
    num_matches = min(num_gt_labels, num_pred_labels)

    # create iou table
    iouMat = np.zeros((num_gt_labels + 1, num_pred_labels + 1), dtype=np.float32)

    for (u, v), c in zip(overlay_labels, overlay_labels_counts):
        iou = c / (gt_labels_count_dict[v] + pred_labels_count_dict[u] - c)
        iouMat[int(v), int(u)] = iou

    # remove background
    iouMat = iouMat[1:, 1:]

    # use IoU threshold th
    if num_matches > 0 and np.max(iouMat) > th:
        costs = -(iouMat > th).astype(float) - iouMat / (2 * num_matches)
        gt_ind, pred_ind = linear_sum_assignment(costs)
        assert num_matches == len(gt_ind) == len(pred_ind)
        match_ok = iouMat[gt_ind, pred_ind] > th
        tp = np.count_nonzero(match_ok)
    else:
        tp = 0
    fp = num_pred_labels - tp
    fn = num_gt_labels - tp
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    accuracy = tp / (tp + fp + fn)

    return precision, recall, accuracy


def plot_two(img: np.ndarray, sdt: np.ndarray, label: str):
    """
    Helper function to plot an image and the auxiliary (intermediate)
    representation of the target.
    """
    fig = plt.figure(constrained_layout=False, figsize=(10, 3))
    spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
    ax1 = fig.add_subplot(spec[0, 0])
    ax1.set_xlabel("Image", fontsize=20)
    plt.imshow(img, cmap="magma")
    ax2 = fig.add_subplot(spec[0, 1])
    ax2.set_xlabel(label, fontsize=20)
    t = plt.imshow(sdt, cmap="magma")
    cbar = fig.colorbar(t, fraction=0.046, pad=0.04)
    tick_locator = ticker.MaxNLocator(nbins=3)
    cbar.locator = tick_locator
    cbar.update_ticks()
    _ = [ax.set_xticks([]) for ax in [ax1, ax2]]
    _ = [ax.set_yticks([]) for ax in [ax1, ax2]]
    plt.tight_layout()
    plt.show()


def plot_three(
    image: np.ndarray, intermediate: np.ndarray, pred: np.ndarray, label: str = "Target"
):
    """
    Helper function to plot an image, the auxiliary (intermediate)
    representation of the target and the model prediction.
    """
    fig = plt.figure(constrained_layout=False, figsize=(10, 3))
    spec = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)
    ax1 = fig.add_subplot(spec[0, 0])
    ax1.set_xlabel("Image", fontsize=20)
    plt.imshow(image, cmap="magma")
    ax2 = fig.add_subplot(spec[0, 1])
    ax2.set_xlabel(label, fontsize=20)
    plt.imshow(intermediate, cmap="magma")
    ax3 = fig.add_subplot(spec[0, 2])
    ax3.set_xlabel("Prediction", fontsize=20)
    t = plt.imshow(pred, cmap="magma")
    cbar = fig.colorbar(t, fraction=0.046, pad=0.04)
    tick_locator = ticker.MaxNLocator(nbins=3)
    cbar.locator = tick_locator
    cbar.update_ticks()
    _ = [ax.set_xticks([]) for ax in [ax1, ax2, ax3]]  # remove the xticks
    _ = [ax.set_yticks([]) for ax in [ax1, ax2, ax3]]  # remove the yticks
    plt.tight_layout()
    plt.show()


def plot_four(
    image: np.ndarray,
    intermediate: np.ndarray,
    pred: np.ndarray,
    seg: np.ndarray,
    label: str = "Target",
    cmap: str = "nipy_spectral",
):
    """
    Helper function to plot an image, the auxiliary (intermediate)
    representation of the target, the model prediction and the predicted segmentation mask.
    """

    fig = plt.figure(constrained_layout=False, figsize=(10, 3))
    spec = gridspec.GridSpec(ncols=4, nrows=1, figure=fig)
    ax1 = fig.add_subplot(spec[0, 0])
    ax1.imshow(image)  # show the image
    ax1.set_xlabel("Image", fontsize=20)
    ax2 = fig.add_subplot(spec[0, 1])
    ax2.imshow(intermediate)  # show the masks
    ax2.set_xlabel(label, fontsize=20)
    ax3 = fig.add_subplot(spec[0, 2])
    t = ax3.imshow(pred)
    ax3.set_xlabel("Pred.", fontsize=20)
    tick_locator = ticker.MaxNLocator(nbins=3)
    cbar = fig.colorbar(t, fraction=0.046, pad=0.04)
    cbar.locator = tick_locator
    cbar.update_ticks()
    ax4 = fig.add_subplot(spec[0, 3])
    ax4.imshow(seg, cmap=cmap, interpolation="none")
    ax4.set_xlabel("Seg.", fontsize=20)
    _ = [ax.set_xticks([]) for ax in [ax1, ax2, ax3, ax4]]  # remove the xticks
    _ = [ax.set_yticks([]) for ax in [ax1, ax2, ax3, ax4]]  # remove the yticks
    plt.tight_layout()
    plt.show()


def test_maximum(find_local_maxima):
    true_array = np.zeros((28, 28))
    locs_x = np.random.randint(0, 28, size=(3))
    locs_y = np.random.randint(0, 28, size=(3))
    true_array[locs_x, locs_y] = 1
    test_array = find_local_maxima(true_array, 3)[0] > 1

    fig = plt.figure(constrained_layout=False, figsize=(10, 3))
    spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
    ax1 = fig.add_subplot(spec[0, 0])
    plt.imshow(true_array)
    plt.title("TRUE MAXIMA")
    ax1 = fig.add_subplot(spec[0, 1])
    plt.imshow(test_array)
    plt.title("FOUND MAXIMA")
    return
