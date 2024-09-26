import numpy as np
import torch
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def mask2rle(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode_modified(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    # loops through and increments the start by the sum of all the points before it
    sum = 0
    for i in range(0, len(starts)):
        starts[i] = sum + starts[i]
        sum = starts[i] + lengths[i]

    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    img = img.reshape(shape)
    return np.transpose(img)
    # return img


def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def get_max_pixel_value(images, targets, outputs):
    mask_outputs = outputs.unsqueeze(1)
    mask_targets = targets.unsqueeze(1)

    segmented_pixels = images * mask_outputs  # apply mask to original image to get segmented pixels
    target_pixels = images * mask_targets  # apply target to original image

    max_target, _ = torch.max(target_pixels, dim=2)
    max_target, _ = torch.max(max_target, dim=2)
    max_target, _ = torch.max(max_target, dim=1)

    max_output, _ = torch.max(segmented_pixels, dim=2)
    max_output, _ = torch.max(max_output, dim=2)
    max_output, _ = torch.max(max_output, dim=1)

    return max_target, max_output


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

    print(f"type: {type(targets)}")
    print(f"image type: {type(images)}")
    print(f"outputs size: {outputs.shape}")

    mask_targets = get_greater_channel_mask(targets)
    mask_outputs = get_greater_channel_mask(outputs)

    print(f"outputs size: {outputs.shape}")
    #mask_outputs = outputs.unsqueeze(1)
    #mask_targets = targets.unsqueeze(1)

    segmented_pixels = images * mask_outputs  # apply mask to original image to get segmented pixels
    target_pixels = images * mask_targets  # apply target to original image

    print(f"segmented_pixels size: {segmented_pixels.shape}")

    max_target, _ = torch.max(target_pixels, dim=2)
    max_target, _ = torch.max(max_target, dim=2)
    max_target, _ = torch.max(max_target, dim=1)

    max_output, _ = torch.max(segmented_pixels, dim=2)
    max_output, _ = torch.max(max_output, dim=2)
    max_output, _ = torch.max(max_output, dim=1)

    return max_target, max_output

