

import pandas as pd
import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.ndimage import binary_erosion

def insert_newlines(text, word_limit=15):
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        current_line.append(word)
        if len(current_line) == word_limit:
            lines.append(' '.join(current_line))
            current_line = []

    # Add the last line if there are any remaining words
    if current_line:
        lines.append(' '.join(current_line))

    return '\n'.join(lines)

def normalize_mip(mip):
    # Step 2: Clip values above 11 to 11
    clipped = np.clip(mip, None, 11)

    # Step 3: Normalize the image so that 11 maps to 255
    # First, ensure no division by zero errors by setting the max value to at least 1
    max_value = np.max(clipped.max())
    normalized = (clipped / max_value) * 255

    # Convert to uint8 type for image representation
    normalized_uint8 = normalized.astype(np.uint8)
    return normalized_uint8




def resample_image(ct_image, target_shape):
    """
    Resample the 3D image to the target shape using interpolation.

    :param ct_image: numpy.ndarray, the original CT image with shape (512, 512, 299)
    :param target_shape: tuple, the target dimensions (192, 192, 299)
    :return: numpy.ndarray, the resampled image
    """
    # Calculate the zoom factors for each dimension
    zoom_factors = [n / o for n, o in zip(target_shape, ct_image.shape)]

    # Resample using spline interpolation
    resampled_image = zoom(ct_image, zoom_factors, order=3)  # order=3 uses cubic interpolation
    return resampled_image


def create_label_outline(label_array):
    """
    Create an outline of the label by eroding and subtracting from the original label.

    :param label_array: numpy.ndarray, binary label array where 1s represent the label
    :return: numpy.ndarray, binary array where 1s represent the outline of the label
    """
    # Perform binary erosion on the label
    eroded_label = binary_erosion(label_array, structure=np.ones((3, 3)))

    # Subtract the eroded label from the original to get the outline
    outline = label_array - eroded_label
    return outline

from skimage import measure

def find_contours_skimage(binary_image, level=0.5):
    # Find contours at a constant value level
    contours = measure.find_contours(binary_image, level=level)
    return contours

def plot_for_label_accuracy_assessment(df):


    image_path_base = "/mnt/Bradshaw/UW_PET_Data/SUV_images/"
    label_path_base = "/mnt/Bradshaw/UW_PET_Data/raw_nifti_uw_pet/uw_labels_v2_nifti/"
    labels_to_skip = ["PETWB_006370_04_label_2", "PETWB_011355_01_label_5", "PETWB_002466_01_label_1",
                      "PETWB_012579_01_label_2", "PETWB_003190_01_label_3",
                      "PETWB_011401_02_label_3"]

    i = 0
    for index, row in df.iterrows():

        print(f"index: {index}")

        if i > 25:
            break
        i += 1

        petlymph = row["Petlymph"]
        if row["Label_Name"] in labels_to_skip:
            continue

        # gets the location of the suv converted image if it exists
        folder_name = str(petlymph) #+ "_" + str(petlymph)
        image_path = os.path.join(image_path_base, folder_name)
        file_names = os.listdir(image_path)
        index_of_suv = [index for index, element in enumerate(file_names) if "suv" in element.lower()]
        image_path = os.path.join(image_path, file_names[index_of_suv[0]])

        # gets location of label nifti
        label_name = row["Label_Name"]
        label_path = os.path.join(label_path_base, label_name + ".nii.gz")

        index_of_ct = [index for index, element in enumerate(file_names) if "ct" in element.lower()]
        # Check if any file was found that contains "CT"
        if index_of_ct:
            # Update image_path to include the file name of the CT image
            ct_image_path = os.path.join(image_path_base, folder_name, file_names[index_of_ct[0]])
        else:
            # Handle the case where no CT file is found
            ct_image_path = None
            print("No CT file found in the directory.")
            continue

        ct_image = nib.load(ct_image_path)
        ct_volume = ct_image.get_fdata()
        print(f"ct dimensions: {ct_volume.shape}")
        slice_num = row["Slice"]
        k_num = row["k"]
        #transaxial_slice = ct_volume[:, :, slice_num]

        # loads in the image as a numpy array
        nii_image = nib.load(image_path)
        img = nii_image.get_fdata()
        #print(f"pet image dimensions: {img.shape}")

        # loads in the label as a numpy array
        nii_label = nib.load(label_path)
        label = nii_label.get_fdata()

        ct_label = resample_image(label, ct_volume.shape)
        nib.save(ct_label,
                 '/mnt/Bradshaw/UW_PET_Data/raw_nifti_uw_pet/resampled_labels/' + row["Label_Name"] + '.nii.gz')
        ct_label = np.round(ct_label).astype(int)
        print(f"ct label dimensions: {ct_label.shape} sum: {np.sum(ct_label)}")
        transaxial_slice = ct_volume[:, :, k_num]

        mip_coronal = np.max(img, axis=1)
        mip_sagittal = np.max(img, axis=0) # I think
        #mip_axial = np.max(img, axis=2) # I think this axis is right
        mip_axial = transaxial_slice
        #mip_coronal = normalize_mip(mip_coronal)

        label_coronal = np.max(label, axis=1)
        label_sagittal = np.max(label, axis=0)
        #label_axial = np.max(label, axis=2)
        #label_axial = np.max(ct_label[:, :, slice_num])
        label_axial = np.max(ct_label, axis=2)

        #mip_coronal = np.rot90(mip_coronal)
        #label_coronal = np.rot90(label_coronal) #label_coronal.T

        # Rotate the images for better display
        mip_coronal = np.rot90(mip_coronal)
        label_coronal = np.rot90(label_coronal)
        mip_sagittal = np.rot90(mip_sagittal)
        label_sagittal = np.rot90(label_sagittal)
        mip_axial = np.rot90(mip_axial, k=-1)
        label_axial = np.rot90(label_axial, k=-1)
        #outline = create_label_outline(label_axial)

        plt.figure(figsize=(24, 10))
        plt.subplot(1, 5, 1)  # 1 row, 2 columns, first subplot
        plt.imshow(mip_coronal, cmap='gray', vmax = 10)  # 'viridis' is a colormap, you can choose others like 'gray', 'plasma', etc.

        # switch the axis plotting of the y axis
        locs, _ = plt.yticks()
        y_min, y_max = plt.ylim()
        filtered_locs = [loc for loc in locs if loc > -1 and loc < mip_coronal.shape[0]]
        filtered_labels = [f"{int(y_max - loc)*-1}" for loc in filtered_locs]
        plt.yticks(filtered_locs, filtered_labels)

        label = label_coronal
        # Set zeros in the second array to NaN for transparency
        label = np.where(label == 1, 250, label)
        array_label_nan = np.where(label == 0, np.nan, label)

        # Plotting
        plt.figure(figsize=(24, 10))

        ax1 = plt.subplot(1, 5, 1)
        ax1.imshow(mip_coronal, cmap='gray_r', vmax=10)
        ax1.set_title('MIP')

        ax2 = plt.subplot(1, 5, 2)
        ax2.imshow(mip_coronal, cmap='gray_r', vmax=10)
        ax2.imshow(np.where(label_coronal == 1, 250, np.nan), cmap='spring', alpha=0.9)
        ax2.set_title('Coronal')

        ax3 = plt.subplot(1, 5, 3)
        ax3.imshow(mip_sagittal, cmap='gray_r', vmax=10)
        ax3.imshow(np.where(label_sagittal == 1, 250, np.nan), cmap='spring', alpha=0.9)
        ax3.set_title('Sagittal')

        ax4 = plt.subplot(1, 5, 4)
        ax4.imshow(mip_axial, cmap='gray_r', vmax=500, vmin=-200)
        ax4.set_title(f'Axial Slice: {slice_num}')

        ax5 = plt.subplot(1, 5, 5)
        ax5.imshow(mip_axial, cmap='gray_r', vmax=500, vmin = -200)
        ax5.imshow(np.where(label_axial == 1, 250, np.nan), cmap='spring', alpha=0.9)
        #plt.imshow(np.where(outline == 1, 250, np.nan) , cmap='spring', alpha=0.9) # Overlay the outline in 'spring' colormap
        ax5.set_title(f'Axial Slice: {slice_num}')



        """
        # Plotting the fourth subplot for the axial view with contour overlay
        ax5 = plt.subplot(1, 5, 5)
        ax5.imshow(mip_axial, cmap='gray', vmax=500, vmin=-200)
        
        # Generate and plot contours from the label
        contours = measure.find_contours(label_axial, level=0.5)  # Use your predefined function or this direct call
        for contour in contours:
            ax5.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')  # Plotting contours

        # Plot contours
        for contour in contours:
            ax5.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')  # contour[:, 1] is x, contour[:, 0] is y
        """

        plt.suptitle(row["sentence"] + " pixels: " + str(np.sum(label_coronal)), fontsize=12, color='black')

        plt.savefig("/UserData/Zach_Analysis/petlymph_image_data/prediction_mips_for_presentations/mips_accuracy_assessment/" + label_name)

        plt.close()