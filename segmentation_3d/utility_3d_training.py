import torch
import copy
import cc3d
import numpy as np
import torch.nn as nn
import json
import torch.nn.functional as F
from monai.utils import MetricReduction, convert_to_dst_type, optional_import, set_determinism
from typing import Dict, Hashable, Mapping, List, Optional
from monai.losses import DiceCELoss
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from monai.networks.utils import one_hot
from monai.metrics import CumulativeAverage, compute_dice, do_metric_reduction




def logits2pred(logits, sigmoid=False, dim=1):
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    return torch.softmax(logits, dim=dim) if not sigmoid else torch.sigmoid(logits)

def con_comp(seg_array):
    # input: a binary segmentation array output: an array with seperated (indexed) connected components of the segmentation array
    connectivity = 26  # 18 or 26
    conn_comp = cc3d.connected_components(seg_array, connectivity=connectivity)
    return conn_comp


def false_pos_pix(gt_array, pred_array, pred_array_baseline=None):
    # compute number of voxels of false positive connected components in prediction mask
    pred_conn_comp = con_comp(pred_array)

    false_pos = 0
    false_pos_num = 0
    for idx in range(1, min(pred_conn_comp.max() + 1, 50)):
        comp_mask = np.isin(pred_conn_comp, idx)
        if comp_mask.sum() <= 8:  # ignore small connected components (0.64 ml)
            continue
        if (comp_mask * gt_array).sum() == 0:
            false_pos = false_pos + comp_mask.sum()
            false_pos_num = false_pos_num + 1

    return false_pos_num


def false_neg_pix(gt_array, pred_array):
    # compute number of voxels of false negative connected components (of the ground truth mask) in the prediction mask
    gt_conn_comp = con_comp(gt_array)
    #print(gt_conn_comp)
    false_neg = 0
    true_pos = 0
    false_neg_num = 0
    true_pos_num = 0
    for idx in range(1, min(gt_conn_comp.max() + 1, 50)):
        comp_mask = np.isin(gt_conn_comp, idx)
        if (comp_mask * pred_array).sum() == 0:
            false_neg = false_neg + comp_mask.sum()
            false_neg_num = false_neg_num + 1
        else:
            true_pos = true_pos + comp_mask.sum()
            true_pos_num = true_pos_num + 1

    return true_pos_num, false_neg_num

class TPFPFNHelper:
    def __init__(self):
        super().__init__()
        pass

    def __call__(self, y_pred, y):
        n_pred_ch = y_pred.shape[1]
        if n_pred_ch > 1:
            y_pred = torch.argmax(y_pred, dim=1, keepdim=True)
        else:
            raise ValueError("y_pred must have more than 1 channel, use softmax instead")

        n_gt_ch = y.shape[1]
        if n_gt_ch > 1:
            y = torch.argmax(y, dim=1, keepdim=True)

        # reducing only spatial dimensions (not batch nor channels)
        TP_sum = 0
        FP_sum = 0
        FN_sum = 0
        y_copy = copy.deepcopy(y).detach().cpu().numpy().squeeze()
        y_pred_copy = copy.deepcopy(y_pred).detach().cpu().numpy().squeeze()
        if y_copy.ndim == 3:  # if batch dim is reduced
            y_copy = y_copy[np.newaxis, ...]
            y_pred_copy = y_pred_copy[np.newaxis, ...]

        for ii in range(y_copy.shape[0]):
            y_ = y_copy[ii]
            y_pred_ = y_pred_copy[ii]

            FP = false_pos_pix(y_, y_pred_)
            TP, FN = false_neg_pix(y_, y_pred_)

            TP_sum += TP
            FP_sum += FP
            FN_sum += FN

        return TP_sum, FP_sum, FN_sum  # all are volumes



# Function to load JSON file
def load_json(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        return data
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# PyTorch
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class DiceHelper:
    def __init__(
        self,
        sigmoid: bool = False,
        include_background: Optional[bool] = None,
        to_onehot_y: Optional[bool] = None,
        softmax: Optional[bool] = None,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN_BATCH,
        get_not_nans: bool = True,
        ignore_empty: bool = True,
        activate: bool = False,
    ) -> None:
        super().__init__()

        self.sigmoid = sigmoid

        self.reduction = reduction
        self.get_not_nans = get_not_nans
        self.ignore_empty = ignore_empty

        self.include_background = sigmoid if include_background is None else include_background
        self.to_onehot_y = not sigmoid if to_onehot_y is None else to_onehot_y
        self.softmax = not sigmoid if softmax is None else softmax
        self.activate = activate
        self.loss = DiceCELoss(include_background=True)

    def __call__(self, y_pred: Union[torch.Tensor, list], y: torch.Tensor):

        n_pred_ch = y_pred.shape[1]

        if self.softmax:
            if n_pred_ch > 1:
                y_pred = torch.argmax(y_pred, dim=1, keepdim=True)
                y_pred = one_hot(y_pred, num_classes=n_pred_ch, dim=1)
        elif self.sigmoid:
            if self.activate:
                y_pred = torch.sigmoid(y_pred)
            y_pred = (y_pred > 0.5).float()

        if self.to_onehot_y and n_pred_ch > 1 and y.shape[1] == 1:
            y = one_hot(y, num_classes=n_pred_ch, dim=1)

        #data = self.loss(input = y_pred, target = y)
        data = compute_dice(
            y_pred=y_pred, y=y, include_background=self.include_background, ignore_empty=self.ignore_empty
        )

        f, not_nans = do_metric_reduction(data, self.reduction)
        return (f, not_nans) if self.get_not_nans else f



def get_greater_channel_mask(volume):
    """
    Given a 5D volume of shape (1, 2, width, height, depth), returns a binary mask of shape
    (width, height, depth) where the mask has value 1 where the second channel is greater than the first.

    Parameters:
    volume (np.ndarray): A 5D NumPy array of shape (1, 2, width, height, depth)

    Returns:
    np.ndarray: A 3D binary mask of shape (width, height, depth) with values 0 and 1
    """
    # Input validation
    if not isinstance(volume, np.ndarray):
        raise TypeError("Input volume must be a NumPy array.")
    if volume.ndim != 5:
        raise ValueError("Input volume must be a 5D NumPy array with shape (1, 2, width, height, depth).")
    if volume.shape[0] != 1:
        raise ValueError("The first dimension of the input volume must be of size 1.")
    if volume.shape[1] != 2:
        raise ValueError("The second dimension of the input volume must be of size 2 (channels).")

    # Remove the singleton batch dimension
    volume = volume[0]  # Now shape is (2, width, height, depth)

    # Extract the two channels
    channel0 = volume[0]  # First channel (width, height, depth)
    channel1 = volume[1]  # Second channel (width, height, depth)

    # Compute the binary mask where the second channel is greater than the first
    mask = channel1 > channel0  # Boolean array (width, height, depth)

    # Convert boolean mask to binary (0 and 1)
    binary_mask = mask.astype(np.uint8)

    return binary_mask
def get_max_pixel_value_3d(images, targets, outputs):

    #print(f"type: {type(targets)}")
    #print(f"image type: {type(images)}")
    #print(f"image size: {images.shape}")
    #print(f"outputs size: {outputs.shape}")

    mask_targets = get_greater_channel_mask(targets)
    mask_outputs = get_greater_channel_mask(outputs)

    #print(f"outputs size: {outputs.shape}")
    #mask_outputs = outputs.unsqueeze(1)
    #mask_targets = targets.unsqueeze(1)

    segmented_pixels = images * mask_outputs  # apply mask to original image to get segmented pixels
    target_pixels = images * mask_targets  # apply target to original image

    #print(f"segmented_pixels size: {segmented_pixels.shape}")

    max_target = np.max(target_pixels, axis=2)
    max_target = np.max(max_target, axis=2)
    max_target = np.max(max_target, axis=2)
    #print(f"max_target size: {max_target.shape}")

    max_target = max_target[0, 1]

    max_output = np.max(segmented_pixels, axis=2)
    max_output = np.max(max_output, axis=2)
    max_output = np.max(max_output, axis=2)
    max_output = max_output[0, 1]

    return max_target, max_output


