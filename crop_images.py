import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from PIL import Image


# Used cropping calucation
def center_crop_and_pad_nifti_image(nifti_image, target_shape=(128, 128, 256), center_point=None):
    # Load the image data
    img_data = nifti_image.get_fdata()

    # Use the provided center point or calculate the center of the image
    if center_point is None:
        center_point = tuple(s // 2 for s in img_data.shape)

    # Initialize start and end indices for cropping
    start_idx = [0, 0, 0]
    end_idx = list(img_data.shape)

    # Adjust the start and end indices to include as many pixels as possible
    for i in range(3):
        start_idx[i] = max(center_point[i] - target_shape[i] // 2, 0)
        end_idx[i] = start_idx[i] + target_shape[i]
        if end_idx[i] > img_data.shape[i]:
            end_idx[i] = img_data.shape[i]
            start_idx[i] = max(end_idx[i] - target_shape[i], 0)

    # Crop the image
    cropped_img = img_data[start_idx[0]:end_idx[0], start_idx[1]:end_idx[1], start_idx[2]:end_idx[2]]

    # Calculate padding to make each side equal
    pad_width = [(max(0, (target_shape[i] - cropped_img.shape[i]) // 2),
                  max(0, (target_shape[i] - cropped_img.shape[i]) // 2)) for i in range(3)]

    # Adjust for any off-by-one errors in padding
    for i in range(3):
        if sum(pad_width[i]) + cropped_img.shape[i] < target_shape[i]:
            pad_width[i] = (pad_width[i][0], pad_width[i][1] + 1)

    # Pad the image
    cropped_padded_img = np.pad(cropped_img, pad_width, mode='constant', constant_values=0)
    print(f"inside crop sum: {np.sum(cropped_padded_img)}")

    return cropped_padded_img
    # Create a new NIfTI image
    #new_img = nib.Nifti1Image(cropped_padded_img, nifti_image.affine, header=nifti_image.header)

    #return new_img


def center_of_furthest_pixels(volume):
    """
    Compute the center of the furthest apart non-zero pixels in each dimension of a 3D volume.

    Args:
    - volume (numpy.ndarray): A 3D numpy array representing the volume.

    Returns:
    - tuple: The (x, y, z) coordinates of the center.
    """
    # Ensure the volume is a numpy array
    volume = np.asarray(volume)

    # Find the indices of all non-zero pixels
    non_zero_indices = np.argwhere(volume > 0)

    # Check if there are any non-zero pixels
    if len(non_zero_indices) == 0:
        return None  # Return None if there are no non-zero pixels

    # Find the min and max indices in each dimension
    min_indices = non_zero_indices.min(axis=0)

    max_indices = non_zero_indices.max(axis=0)

    # Compute the center in each dimension
    center = np.round((min_indices + max_indices) / 2)

    return int(center[0]), int(center[1]), int(center[2])

def resample_nifti_image(nifti_image, new_resolution=(4, 4, 4)):
    # Load the NIfTI file
    img = nifti_image

    # Get the current voxel sizes from the header
    original_voxel_sizes = img.header.get_zooms()

    # Calculate the zoom factors for resampling
    zoom_factors = [o / n for o, n in zip(original_voxel_sizes, new_resolution)]

    # Resample the image data
    resampled_data = zoom(img.get_fdata(), zoom_factors, order=3)

    # Update the header with the new resolution
    new_header = img.header.copy()
    new_header.set_zooms(new_resolution)

    # Create a new NIfTI image with the resampled data and updated header
    resampled_img = nib.Nifti1Image(resampled_data, img.affine, header=new_header)

    return resampled_img

def save_2d_image_lossless(image, file_name):
    """
    Saves a 2D image to a file in a lossless format (PNG).

    Args:
    image (ndarray): A 2D image array.
    file_name (str): The name of the file to save the image.
    """
    # Convert the image to PIL Image format and ensure it's in 'L' mode for grayscale
    pil_image = Image.fromarray(image).convert('L')

    # Save the image in a lossless format (PNG)
    pil_image.save(file_name, format='PNG')


def crop_images_to_mips():

    image_path = "/UserData/Xin/Monai_Auto3dSeg/COG_lymph_seg/Auto3dseg/data/COG/imagesTr"
    label_path = "/UserData/Xin/Monai_Auto3dSeg/COG_lymph_seg/Auto3dseg/data/COG/labelsTr"
    files = []
    try:
        files = os.listdir(label_path)
    except FileNotFoundError:
        print("folder not found")
    # print(files)

    """
    for file in files:
        print(file[-11:-7])
        if file[-11:-7] == "0000":
            #print("pet")
            #print(file[:-12])
            file_name = file[:-12] + "_pet"
        if file[-11:-7] == "0001":
            #print("ct")
            #print(file[:-12])
            file_name = file[:-12] + "_ct"
        print(file_name)
    """
    i = 0
    for file in files:
        print(i)

        pet_path = os.path.join(image_path, str(file[:-7]) + "_0000.nii.gz")
        pet_img = nib.load(pet_path)
        pet_img = resample_nifti_image(pet_img)


        file_path = os.path.join(label_path, file)
        img = nib.load(file_path)
        print(f"untouched labeled sum: {np.sum(img.get_fdata())}")

        resampled_img = resample_nifti_image(img)
        file_name = file[:-7]
        volume_data = resampled_img.get_fdata()
        print(f"resampled labeled sum: {np.sum(volume_data)}")

        #volume_data = img.get_fdata()

        #initial_sum = np.sum(volume_data)

        center = center_of_furthest_pixels(volume_data)
        cropped_img = center_crop_and_pad_nifti_image(img, target_shape=(128, 128, 256), center_point=center)
        #print(f"after cropping sum: {np.sum(cropped_img)}")

        cropped_pet = center_crop_and_pad_nifti_image(pet_img, target_shape=(128, 128, 256), center_point=center)

        #volume_data = cropped_img.get_fdata()
        #pet_volume = cropped_pet.get_fdata()
        print(f"full 3d label sum: {np.sum(volume_data)}")

        volume_data = cropped_img
        pet_volume = cropped_pet
        #if initial_sum != cropped_sum:
        #    print(initial_sum)
        #    print(cropped_sum)
        #    print(file)
        #save off label mips
        mip_sagittal = np.max(volume_data, axis=0) # sagittal
        print(f"saved sum: {np.sum(mip_sagittal)}")
        #save_2d_image_lossless(mip_sagittal, "/UserData/Zach_Analysis/cog_data_splits/mips/cropped_mips/sagittal/label/" + file_name + "_label_sagittal.png") # sagittal

        mip_coronal = np.max(volume_data, axis=1) # coronial
        #save_2d_image_lossless(mip_coronal, "/UserData/Zach_Analysis/cog_data_splits/mips/cropped_mips/coronal/label" + file_name + "_label_coronal.png") #

        mip_axial = np.max(volume_data, axis=2) # axial
        #save_2d_image_lossless(mip_axial, "/UserData/Zach_Analysis/cog_data_splits/mips/cropped_mips/axial/label/" + file_name +  "_label_axial.png") #

        # save off pet mips
        mip_sagittal_pet = np.max(pet_volume, axis=0)  # sagittal
        #save_2d_image_lossless(mip_sagittal_pet, "/UserData/Zach_Analysis/cog_data_splits/mips/cropped_mips/sagittal/image/" + file_name + "_pet_sagittal.png")  # sagittal

        mip_coronal_pet = np.max(pet_volume, axis=1)  # coronial
        #save_2d_image_lossless(mip_coronal_pet, "/UserData/Zach_Analysis/cog_data_splits/mips/cropped_mips/coronal/image/" + file_name + "_pet_coronal.png")  #

        mip_axial_pet = np.max(pet_volume, axis=2)  # axial
        #save_2d_image_lossless(mip_axial_pet, "/UserData/Zach_Analysis/cog_data_splits/mips/cropped_mips/axial/image/" + file_name + "_pet_axial.png")  #




        i += 1