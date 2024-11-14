import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import os
import nibabel as nib


def insert_newlines(text, word_limit):
    """
    Insert newlines into the text every 'word_limit' words for better display in the plot title.

    Parameters:
    text (str): The text to process.
    word_limit (int): Number of words after which to insert a newline.

    Returns:
    str: The text with newlines inserted.
    """
    words = text.split()
    lines = [' '.join(words[i:i+word_limit]) for i in range(0, len(words), word_limit)]
    return '\n'.join(lines)

def plot_volumes(pet_volume, ct_volume, label_volume, label_name, slice_num, voxel_dims, ct_dims, sentence, suv_value, output_path):
    """
    Plot PET, CT, and label volumes with specified settings.

    Parameters:
    pet_volume (numpy.ndarray): The PET image volume.
    ct_volume (numpy.ndarray): The CT image volume.
    label_volume (numpy.ndarray): The label volume.
    label_name (str): The name of the label, used for saving the figure.
    slice_num (int): The slice number to be used for axial slices.
    voxel_dims (tuple): The voxel dimensions (dx, dy, dz) of the PET volume.
    ct_dims (tuple): The voxel dimensions (dx, dy, dz) of the CT volume.
    sentence (str): The sentence to be displayed in the title.
    suv_value (float): The SUV value to be displayed in the title.
    output_path (str): The directory where the figure will be saved.
    """

    # Rotate and flip the CT volume for proper orientation
    rotated_volume = np.transpose(ct_volume, (1, 0, 2))
    rotated_volume = np.flip(rotated_volume, axis=1)
    rotated_volume = np.rot90(rotated_volume, axes=(0, 1))
    ct_volume = rotated_volume

    # Flip the label and PET volumes along axis=0 for correct orientation
    label_volume = np.flip(label_volume, axis=0)
    ct_volume = np.flip(ct_volume, axis=0)
    pet_volume = np.flip(pet_volume, axis=0)

    # Find the slice index with the maximum label sum in the axial plane
    sums_axial = np.sum(label_volume, axis=(0, 1))
    k_num = np.argmax(sums_axial)

    # Find the slice index with the maximum label sum in the coronal plane
    sum_coronal = np.sum(label_volume, axis=(0, 2))
    coronal_slice = int(np.argmax(sum_coronal) * voxel_dims[1] / ct_dims[1])

    # Find the slice index with the maximum label sum in the sagittal plane
    sums_sagittal = np.sum(label_volume, axis=(1, 2))
    sagittal_slice = int(np.argmax(sums_sagittal) * voxel_dims[0] / ct_dims[0])

    # Extract the axial slice from the CT volume
    transaxial_slice = ct_volume[:, :, k_num]

    # Generate Maximum Intensity Projections (MIPs) for PET volume
    mip_coronal = np.max(pet_volume, axis=1)
    mip_sagittal = np.max(pet_volume, axis=0)
    mip_axial = transaxial_slice  # Using the CT axial slice for consistency

    # Extract corresponding CT slices
    ct_mip_coronal = ct_volume[:, coronal_slice, :]
    ct_mip_sagittal = ct_volume[sagittal_slice, :, :]

    # Generate label projections
    label_coronal = np.max(label_volume, axis=1)
    label_sagittal = np.max(label_volume, axis=0)
    label_axial = np.max(label_volume, axis=2)

    # Extract the PET transaxial slice and rotate for display
    pet_transaxial_slice = pet_volume[:, :, k_num]
    pet_transaxial_slice = np.rot90(pet_transaxial_slice, k=-1)

    # Rotate images for proper display
    mip_coronal = np.rot90(mip_coronal)
    label_coronal = np.rot90(label_coronal)
    mip_sagittal = np.rot90(mip_sagittal)
    label_sagittal = np.rot90(label_sagittal)
    mip_axial = np.rot90(mip_axial, k=-1)
    label_axial = np.rot90(label_axial, k=-1)
    ct_mip_coronal = np.rot90(ct_mip_coronal)
    ct_mip_sagittal = np.rot90(ct_mip_sagittal)

    # Flip images horizontally for correct orientation
    mip_coronal = np.fliplr(mip_coronal)
    label_coronal = np.fliplr(label_coronal)
    ct_mip_coronal = np.fliplr(ct_mip_coronal)

    # Start plotting
    plt.figure(figsize=(24, 12))

    # First subplot: Original MIP Coronal
    ax1 = plt.subplot(2, 4, 1)
    ax1.imshow(mip_coronal, cmap='gray_r', vmax=10, aspect=voxel_dims[2] / voxel_dims[1])
    ax1.set_title('Original Image')

    # Second subplot: MIP Coronal with label overlay
    ax2 = plt.subplot(2, 4, 2)
    ax2.imshow(mip_coronal, cmap='gray_r', vmax=10, aspect=voxel_dims[2] / voxel_dims[1])
    ax2.imshow(np.where(label_coronal == 1, 250, np.nan), cmap='spring', alpha=0.9,
               aspect=voxel_dims[2] / voxel_dims[1])
    ax2.set_title('MIP Coronal')

    # Third subplot: MIP Sagittal with label overlay
    ax3 = plt.subplot(2, 4, 3)
    ax3.imshow(mip_sagittal, cmap='gray_r', vmax=10, aspect=voxel_dims[2] / voxel_dims[1])
    ax3.imshow(np.where(label_sagittal == 1, 250, np.nan), cmap='spring', alpha=0.9,
               aspect=voxel_dims[2] / voxel_dims[1])
    ax3.set_title('MIP Sagittal')

    # Fourth subplot: PET Axial slice with label overlay
    ax4 = plt.subplot(2, 4, 4)
    ax4.imshow(pet_transaxial_slice, cmap='gray_r', vmax=10, aspect=voxel_dims[0] / voxel_dims[1])
    ax4.imshow(np.where(label_axial == 1, 250, np.nan), cmap='spring', alpha=0.9,
               aspect=voxel_dims[0] / voxel_dims[1])
    ax4.set_title(f'PET Axial slice: {k_num}')

    # Rescale label_axial to match mip_axial size
    scale_x = mip_axial.shape[0] / label_axial.shape[0]
    scale_y = mip_axial.shape[1] / label_axial.shape[1]
    label_axial_resized = zoom(label_axial, (scale_x, scale_y), order=0)
    label_axial = label_axial_resized

    # Fifth subplot: Original CT Axial Slice
    ax5 = plt.subplot(2, 4, 5)
    ax5.imshow(mip_axial, cmap='gray', vmax=600, vmin=-300)
    ax5.set_title(f'Original Axial Slice: {slice_num}')

    # Sixth subplot: CT Axial Slice with label overlay
    ax6 = plt.subplot(2, 4, 8)
    ax6.imshow(mip_axial, cmap='gray', vmax=600, vmin=-300)
    ax6.imshow(np.where(label_axial == 1, 250, np.nan), cmap='spring', alpha=0.9,
               aspect=ct_dims[0] / ct_dims[1])
    ax6.set_title(f'Axial Slice: {slice_num} With Label')

    # Rescale label_coronal and label_sagittal to match CT slices
    rescale_factor_coronal = [ct_mip_coronal.shape[0] / label_coronal.shape[0],
                              ct_mip_coronal.shape[1] / label_coronal.shape[1]]
    rescale_factor_sagittal = [ct_mip_sagittal.shape[0] / label_sagittal.shape[0],
                               ct_mip_sagittal.shape[1] / label_sagittal.shape[1]]

    label_coronal_rescaled = zoom(label_coronal, rescale_factor_coronal, order=1)
    label_sagittal_rescaled = zoom(label_sagittal, rescale_factor_sagittal, order=1)

    # Seventh subplot: CT Coronal with label overlay
    ax7 = plt.subplot(2, 4, 6)
    ax7.imshow(ct_mip_coronal, cmap='gray', vmax=600, vmin=-300, aspect=ct_dims[2] / ct_dims[1])
    ax7.imshow(np.where(label_coronal_rescaled == 1, 250, np.nan), cmap='spring', alpha=0.9,
               aspect=ct_dims[2] / ct_dims[1])
    ax7.set_title(f'CT Coronal: {coronal_slice} With Label')

    # Eighth subplot: CT Sagittal with label overlay
    ax8 = plt.subplot(2, 4, 7)
    ax8.imshow(ct_mip_sagittal, cmap='gray', vmax=600, vmin=-300, aspect=ct_dims[2] / ct_dims[1])
    ax8.imshow(np.where(label_sagittal_rescaled == 1, 250, np.nan), cmap='spring', alpha=0.9,
               aspect=ct_dims[2] / ct_dims[1])
    ax8.set_title(f'Sagittal Slice: {512 - sagittal_slice} With Label')

    # Insert newlines into the sentence for display
    sentence = insert_newlines(sentence, word_limit=25)

    # Add the title with the sentence, number of pixels, and SUV value
    plt.suptitle(f"{sentence} Pixels: {np.sum(label_coronal)} SUV: {suv_value}", fontsize=12, color='black')

    # Save the figure
    plt.savefig(os.path.join(output_path, label_name + '.png'), dpi=300)
    plt.close()

def testing_ploting_external_cog_data():

    pet_location = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/images6/PETWB_013536_01_suv_cropped.nii.gz"
    # Load the NIfTI file using nibabel
    pet_nii_image = nib.load(pet_location)
    # Extract the image data as a NumPy array
    pet_volume = pet_nii_image.get_fdata()
    # Optionally, retrieve the voxel dimensions (e.g., for use in plotting)
    voxel_dims = pet_nii_image.header.get_zooms()

    # Load CT volume
    ct_location = '/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/images6/PETWB_013536_01_ct_cropped.nii.gz'
    ct_nii = nib.load(ct_location)
    ct_volume = ct_nii.get_fdata()
    ct_dims = ct_nii.header.get_zooms()

    # Load Label volume
    label_location = '/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/labels6/PETWB_013536_01_label_1.nii.gz'
    label_nii = nib.load(label_location)
    label_volume = label_nii.get_fdata()

    # Assuming you have the pet_volume, ct_volume, label_volume, and other parameters ready
    plot_volumes(
        pet_volume=pet_volume,
        ct_volume=ct_volume,
        label_volume=label_volume,
        label_name='example_label',
        slice_num=50,
        voxel_dims=(3.0, 3.0, 3.0), # or voxel_dims
        ct_dims=(1.0, 1.0, 1.0),
        sentence='This is an example sentence for the plot title.',
        suv_value=2.5,
        output_path='/path/to/save/figure'
    )
