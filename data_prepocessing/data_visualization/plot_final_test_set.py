
import pandas as pd
import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.ndimage import binary_erosion

def get_slice_thickness(folder_name):
    image_path_base = "/mnt/Bradshaw/UW_PET_Data/SUV_images/"
    image_path = os.path.join(image_path_base, folder_name)
    file_names = os.listdir(image_path)
    index_of_suv = [index for index, element in enumerate(file_names) if "suv" in element.lower()]
    image_path = os.path.join(image_path, file_names[index_of_suv[0]])
    nii_image = nib.load(image_path)
    header = nii_image.header
    voxel_dims = header.get_zooms()
    return voxel_dims
def extract_image_id(path):
    # Extract the part of the filename before '_suv_cropped.nii.gz'
    return path.split('/')[-1].split('_suv_cropped')[0]

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

def plot_final_testset(df):


    #image_path_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/images4_v2/"
    #label_path_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/labels4_v2/"
    image_path_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/images6/"
    label_path_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/labels6/"

    original_df = pd.read_excel("/UserData/Zach_Analysis/suv_slice_text/uw_all_pet_preprocess_chain_v4/removed_wrong_suv_max_and_slices_13.xlsx")
    crop_df = pd.read_excel("/UserData/Zach_Analysis/suv_slice_text/uw_all_pet_preprocess_chain_v4/crop_offset_lookup.xlsx")
    i = 0

    for index, row in df.iterrows():
        print(f"index: {index}")
        i += 1

        petlymph = extract_image_id(row["image"])

        if "017530" not in petlymph:
            continue
        dims = get_slice_thickness(petlymph)
        print(f"dims: {dims}")
        image_path = os.path.join(image_path_base, petlymph + "_suv_cropped.nii.gz")
        ct_image_path = os.path.join(image_path_base,  petlymph + "_ct_cropped.nii.gz")
        print(f"image name: {petlymph}")
        # gets location of label nifti
        label_name = row["label_name"]
        label_path = os.path.join(label_path_base, label_name + ".nii.gz")

        ct_image = nib.load(ct_image_path)
        ct_volume = ct_image.get_fdata()
        print(f"ct dimensions: {ct_volume.shape}")
        slice_num = row["slice_num"]
        #k_num = row["k"]
        original_row = original_df.loc[original_df['Label_Name'] == label_name]
        crop_row = crop_df.loc[crop_df['id'] == petlymph]
        crop_offset = crop_row['crop_offset'].iloc[0]
        #print(f"crop offset: {crop_offset}")
        #k_num = original_row["k"]
        k_num = -1*slice_num + crop_offset - 1
        #crop_offset = 1
        slice_estimation = (slice_num - (crop_offset - 1))*(dims[2]/3)
        slice_estimation = int(np.round(slice_estimation))
        #print(f"k num:{k_num}")
        # transaxial_slice = ct_volume[:, :, slice_num]

        # loads in the image as a numpy array
        nii_image = nib.load(image_path)
        img = nii_image.get_fdata()
        # print(f"pet image dimensions: {img.shape}")

        """
        # loads in the label as a numpy array
        nii_label = nib.load(label_path)
        label = nii_label.get_fdata()

        if os.path.exists('/mnt/Bradshaw/UW_PET_Data/raw_nifti_uw_pet/resampled_labels/' + row["Label_Name"] + '.nii.gz'):
            ct_label = nib.load('/mnt/Bradshaw/UW_PET_Data/raw_nifti_uw_pet/resampled_labels/' + row["Label_Name"] + '.nii.gz')
        else:
            ct_label = nib.Nifti1Image(label, affine=ct_volume.affine, header=ct_volume.header)
            #ct_label = resample_image(label, ct_volume.shape)
            nib.save(ct_label, '/mnt/Bradshaw/UW_PET_Data/raw_nifti_uw_pet/resampled_labels/' + row["Label_Name"] + '.nii.gz')
        """

        if os.path.exists(label_path):
            # loads in the label as a numpy array
            nii_label = nib.load(label_path)
            label = nii_label.get_fdata()  # the label data
        else:
            continue

        # flip data
        label = np.flip(label, axis=0)
        ct_volume = np.flip(ct_volume, axis=0)
        img = np.flip(img, axis=0)
        """
        # Checking if the resampled label file already exists
        resampled_path = '/mnt/Bradshaw/UW_PET_Data/resampled_labels_final_test_set/' + row["Label_Name"] + '.nii.gz'
        if os.path.exists(resampled_path):
            ct_label = nib.load(resampled_path)
            print("found label")
        else:
            print("resampling label")
            ct_label = resample_image(label, ct_volume.shape)
            # If it doesn't exist, create a new NIfTI image using the label array, affine, and header from the original label image
            ct_label = nib.Nifti1Image(ct_label, affine=nii_label.affine, header=nii_label.header)
            # Optionally, save it if needed
            nib.save(ct_label, resampled_path)
        """
        #k_num = int(-1*np.round(slice_estimation))
        #print(k_num)

        sums = np.sum(label, axis=(0, 1))
        # Find the index of the slice with the maximum sum
        k_num = np.argmax(sums)

        ct_label = label
        #ct_label = ct_label.get_fdata()
        #ct_label = np.round(ct_label).astype(int)
        #print(f"ct label dimensions: {ct_label.shape} sum: {np.sum(ct_label)}")
        transaxial_slice = ct_volume[:, :, k_num]

        mip_coronal = np.max(img, axis=1)
        mip_sagittal = np.max(img, axis=0) # I think
        # mip_axial = np.max(img, axis=2) # I think this axis is right
        mip_axial = transaxial_slice
        # mip_coronal = normalize_mip(mip_coronal)

        label_coronal = np.max(label, axis=1)
        label_sagittal = np.max(label, axis=0)
        # label_axial = np.max(label, axis=2)
        # label_axial = np.max(ct_label[:, :, slice_num])
        label_axial = np.max(ct_label, axis=2)

        # mip_coronal = np.rot90(mip_coronal)
        # label_coronal = np.rot90(label_coronal) #label_coronal.T

        # Rotate the images for better display
        mip_coronal = np.rot90(mip_coronal)
        label_coronal = np.rot90(label_coronal)
        mip_sagittal = np.rot90(mip_sagittal)
        label_sagittal = np.rot90(label_sagittal)
        mip_axial = np.rot90(mip_axial, k=-1)
        label_axial = np.rot90(label_axial, k=-1)
        # outline = create_label_outline(label_axial)

        plt.figure(figsize=(24, 10))
        plt.subplot(1, 5, 1)  # 1 row, 2 columns, first subplot
        plt.imshow(mip_coronal, cmap='gray', vmax = 10)  # 'viridis' is a colormap, you can choose others like 'gray', 'plasma', etc.

        # switch the axis plotting of the y axis
        locs, _ = plt.yticks()
        y_min, y_max = plt.ylim()
        filtered_locs = [loc for loc in locs if loc > -1 and loc < mip_coronal.shape[0]]
        filtered_labels = [f"{int(y_max - loc ) *-1}" for loc in filtered_locs]
        plt.yticks(filtered_locs, filtered_labels)

        label = label_coronal
        # Set zeros in the second array to NaN for transparency
        label = np.where(label == 1, 250, label)
        array_label_nan = np.where(label == 0, np.nan, label)

        # Extract voxel dimensions (in mm)
        voxel_dims = nii_image.header.get_zooms()

        # Plotting
        plt.figure(figsize=(24, 10))

        ax1 = plt.subplot(1, 5, 1)
        ax1.imshow(mip_coronal, cmap='gray_r', vmax=10, aspect=voxel_dims[0 ] /voxel_dims[1])
        ax1.set_title('MIP')

        ax2 = plt.subplot(1, 5, 2)
        ax2.imshow(mip_coronal, cmap='gray_r', vmax=10, aspect=voxel_dims[0 ] /voxel_dims[1])
        ax2.imshow(np.where(label_coronal == 1, 250, np.nan), cmap='spring', alpha=0.9, aspect=voxel_dims[0 ] /voxel_dims[1])
        ax2.set_title('Coronal')

        ax3 = plt.subplot(1, 5, 3)
        ax3.imshow(mip_sagittal, cmap='gray_r', vmax=10, aspect=voxel_dims[0 ] /voxel_dims[1])
        ax3.imshow(np.where(label_sagittal == 1, 250, np.nan), cmap='spring', alpha=0.9, aspect=voxel_dims[0 ] /voxel_dims[1])
        ax3.set_title('Sagittal')

        # Flip the image data horizontally
        mip_axial = np.fliplr(mip_axial)
        label_axial = np.fliplr(label_axial)
        # Flip the image data upside down
        mip_axial = np.flipud(mip_axial)
        label_axial = np.flipud(label_axial)


        ax4 = plt.subplot(1, 5, 4)
        ax4.imshow(mip_axial, cmap='gray', vmax=600, vmin=-300)
        ax4.set_title(f'Axial Slice: {slice_num}')

        ax5 = plt.subplot(1, 5, 5)
        ax5.imshow(mip_axial, cmap='gray', vmax=600, vmin = -300)
        ax5.imshow(np.where(label_axial == 1, 250, np.nan), cmap='spring', alpha=0.9)
        # plt.imshow(np.where(outline == 1, 250, np.nan) , cmap='spring', alpha=0.9) # Overlay the outline in 'spring' colormap
        ax5.set_title(f'Axial Slice: {slice_num} With Label')



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
        #print(original_row)
        sentence = original_row["sentence"].iloc[0]
        #print(sentence)
        #print(type(sentence))
        sentence = insert_newlines(sentence, word_limit=25)
        plt.suptitle(sentence + " Pixels: " + str(np.sum(label_coronal)) +" Slice estimation:" + str(slice_estimation) + " SUV: " + str(
            original_row["SUV"].iloc[0]), fontsize=12, color='black')

        plt.savefig(
            "/UserData/Zach_Analysis/petlymph_image_data/prediction_mips_for_presentations/mips_final_test_set_3_testing_cropped_check_registration/" + label_name,
            dpi=300)

        plt.close()

def plot_final_testset_v2(df):
    # image_path_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/images4_v2/"
    # label_path_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/labels4_v2/"
    #image_path_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/images5/"
    #label_path_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/labels5/"

    image_path_root = "/mnt/Bradshaw/UW_PET_Data/SUV_images/"
    label_path_base = "/mnt/Bradshaw/UW_PET_Data/raw_nifti_uw_pet/uw_labels_v4_nifti/"


    original_df = pd.read_excel(
        "/UserData/Zach_Analysis/suv_slice_text/uw_all_pet_preprocess_chain_v4/removed_wrong_suv_max_and_slices_13.xlsx")
    crop_df = pd.read_excel(
        "/UserData/Zach_Analysis/suv_slice_text/uw_all_pet_preprocess_chain_v4/crop_offset_lookup.xlsx")
    i = 0

    for index, row in df.iterrows():
        print(f"index: {index}")
        i += 1

        petlymph = extract_image_id(row["image"])

        dims = get_slice_thickness(petlymph)
        #print(f"dims: {dims}")
        image_path_base = os.path.join(image_path_root, petlymph)

        file_names = os.listdir(image_path_base)
        index_of_suv = [index for index, element in enumerate(file_names) if "suv" in element.lower()]
        image_path = os.path.join(image_path_base, file_names[index_of_suv[0]])

        index_of_ct = [index for index, element in enumerate(file_names) if "ct" in element.lower()]
        ct_image_path = os.path.join(image_path_base, file_names[index_of_ct[0]])
        #image_path = os.path.join(image_path_base, petlymph, + "_suv_cropped.nii.gz")
        #ct_image_path = os.path.join(image_path_base, petlymph + "_ct_cropped.nii.gz")
        #print(f"image name: {petlymph}")
        # gets location of label nifti
        label_name = row["label_name"]
        label_path = os.path.join(label_path_base, label_name + ".nii.gz")

        ct_image = nib.load(ct_image_path)
        ct_volume = ct_image.get_fdata()
        #print(f"ct dimensions: {ct_volume.shape}")
        slice_num = row["slice_num"]
        # k_num = row["k"]
        original_row = original_df.loc[original_df['Label_Name'] == label_name]
        #crop_row = crop_df.loc[crop_df['id'] == petlymph]
        #crop_offset = crop_row['crop_offset'].iloc[0]
        # print(f"crop offset: {crop_offset}")
        # k_num = original_row["k"]
        #k_num = -1 * slice_num + crop_offset - 1
        # crop_offset = 1
        # slice_estimation = (slice_num - (crop_offset - 1)) * (dims[2] / 3)
        # slice_estimation = int(np.round(slice_estimation))
        # print(f"k num:{k_num}")
        # transaxial_slice = ct_volume[:, :, slice_num]

        # loads in the image as a numpy array
        nii_image = nib.load(image_path)
        img = nii_image.get_fdata()
        # print(f"pet image dimensions: {img.shape}")

        """
        # loads in the label as a numpy array
        nii_label = nib.load(label_path)
        label = nii_label.get_fdata()

        if os.path.exists('/mnt/Bradshaw/UW_PET_Data/raw_nifti_uw_pet/resampled_labels/' + row["Label_Name"] + '.nii.gz'):
            ct_label = nib.load('/mnt/Bradshaw/UW_PET_Data/raw_nifti_uw_pet/resampled_labels/' + row["Label_Name"] + '.nii.gz')
        else:
            ct_label = nib.Nifti1Image(label, affine=ct_volume.affine, header=ct_volume.header)
            #ct_label = resample_image(label, ct_volume.shape)
            nib.save(ct_label, '/mnt/Bradshaw/UW_PET_Data/raw_nifti_uw_pet/resampled_labels/' + row["Label_Name"] + '.nii.gz')
        """

        if os.path.exists(label_path):
            # loads in the label as a numpy array
            nii_label = nib.load(label_path)
            label = nii_label.get_fdata()  # the label data
        else:
            continue

        # flip data
        label = np.flip(label, axis=0)
        ct_volume = np.flip(ct_volume, axis=0)
        img = np.flip(img, axis=0)

        """
        # Checking if the resampled label file already exists
        resampled_path = '/mnt/Bradshaw/UW_PET_Data/resampled_labels_final_test_set/' + row["Label_Name"] + '.nii.gz'
        if os.path.exists(resampled_path):
            ct_label = nib.load(resampled_path)
            print("found label")
        else:
            print("resampling label")
            ct_label = resample_image(label, ct_volume.shape)
            # If it doesn't exist, create a new NIfTI image using the label array, affine, and header from the original label image
            ct_label = nib.Nifti1Image(ct_label, affine=nii_label.affine, header=nii_label.header)
            # Optionally, save it if needed
            nib.save(ct_label, resampled_path)
        """
        # k_num = int(-1*np.round(slice_estimation))
        # print(k_num)

        sums = np.sum(label, axis=(0, 1))
        # Find the index of the slice with the maximum sum
        k_num = np.argmax(sums)

        ct_label = label
        # ct_label = ct_label.get_fdata()
        # ct_label = np.round(ct_label).astype(int)
        # print(f"ct label dimensions: {ct_label.shape} sum: {np.sum(ct_label)}")
        transaxial_slice = ct_volume[:, :, k_num]

        mip_coronal = np.max(img, axis=1)
        mip_sagittal = np.max(img, axis=0)  # I think
        # mip_axial = np.max(img, axis=2) # I think this axis is right
        mip_axial = transaxial_slice
        # mip_coronal = normalize_mip(mip_coronal)

        label_coronal = np.max(label, axis=1)
        label_sagittal = np.max(label, axis=0)
        # label_axial = np.max(label, axis=2)
        # label_axial = np.max(ct_label[:, :, slice_num])
        label_axial = np.max(ct_label, axis=2)

        # mip_coronal = np.rot90(mip_coronal)
        # label_coronal = np.rot90(label_coronal) #label_coronal.T

        # Rotate the images for better display
        mip_coronal = np.rot90(mip_coronal)
        label_coronal = np.rot90(label_coronal)
        mip_sagittal = np.rot90(mip_sagittal)
        label_sagittal = np.rot90(label_sagittal)
        mip_axial = np.rot90(mip_axial, k=-1)
        label_axial = np.rot90(label_axial, k=-1)
        # outline = create_label_outline(label_axial)

        mip_coronal = np.fliplr(mip_coronal)
        label_coronal = np.fliplr(label_coronal)

        plt.figure(figsize=(24, 10))
        plt.subplot(1, 5, 1)  # 1 row, 2 columns, first subplot
        plt.imshow(mip_coronal, cmap='gray',
                   vmax=10)  # 'viridis' is a colormap, you can choose others like 'gray', 'plasma', etc.

        # switch the axis plotting of the y axis
        locs, _ = plt.yticks()
        y_min, y_max = plt.ylim()
        filtered_locs = [loc for loc in locs if loc > -1 and loc < mip_coronal.shape[0]]
        filtered_labels = [f"{int(y_max - loc) * -1}" for loc in filtered_locs]
        plt.yticks(filtered_locs, filtered_labels)

        label = label_coronal
        # Set zeros in the second array to NaN for transparency
        label = np.where(label == 1, 250, label)
        array_label_nan = np.where(label == 0, np.nan, label)

        # Extract voxel dimensions (in mm)
        voxel_dims = nii_image.header.get_zooms()
        print(f"pet voxel dims: {voxel_dims}")

        # Plotting
        plt.figure(figsize=(24, 10))

        ax1 = plt.subplot(1, 5, 1)
        ax1.imshow(mip_coronal, cmap='gray_r', vmax=10, aspect=voxel_dims[0] / voxel_dims[1])
        ax1.set_title('MIP')

        ax2 = plt.subplot(1, 5, 2)
        ax2.imshow(mip_coronal, cmap='gray_r', vmax=10, aspect=voxel_dims[0] / voxel_dims[1])
        ax2.imshow(np.where(label_coronal == 1, 250, np.nan), cmap='spring', alpha=0.9,
                   aspect=voxel_dims[0] / voxel_dims[1])
        ax2.set_title('Coronal')

        ax3 = plt.subplot(1, 5, 3)
        ax3.imshow(mip_sagittal, cmap='gray_r', vmax=10, aspect=voxel_dims[0] / voxel_dims[1])
        ax3.imshow(np.where(label_sagittal == 1, 250, np.nan), cmap='spring', alpha=0.9,
                   aspect=voxel_dims[0] / voxel_dims[1])
        ax3.set_title('Sagittal')

        # Flip the image data horizontally
        #mip_axial = np.fliplr(mip_axial)
        #label_axial = np.fliplr(label_axial)

        # Assuming label_axial needs to be the same size as mip_axial
        scale_x = mip_axial.shape[0] / label_axial.shape[0]
        scale_y = mip_axial.shape[1] / label_axial.shape[1]
        label_axial_resized = zoom(label_axial, (scale_x, scale_y), order=0)  # order=0 for nearest-neighbor interpolation
        label_axial = label_axial_resized
        #print(mip_axial.shape, label_axial.shape)
        #print(mip_axial.dtype, label_axial.dtype)

        ax4 = plt.subplot(1, 5, 4)
        ax4.imshow(mip_axial, cmap='gray', vmax=600, vmin=-300)
        ax4.set_title(f'Axial Slice: {slice_num}')

        ax5 = plt.subplot(1, 5, 5)
        ax5.imshow(mip_axial, cmap='gray', vmax=600, vmin=-300)
        ax5.imshow(np.where(label_axial == 1, 250, np.nan), cmap='spring', alpha=0.9, aspect=voxel_dims[0] / voxel_dims[1])
        # plt.imshow(np.where(outline == 1, 250, np.nan) , cmap='spring', alpha=0.9) # Overlay the outline in 'spring' colormap
        ax5.set_title(f'Axial Slice: {slice_num} With Label')

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
        # print(original_row)
        sentence = original_row["sentence"].iloc[0]
        # print(sentence)
        # print(type(sentence))
        sentence = insert_newlines(sentence, word_limit=25)
        plt.suptitle(sentence + " Pixels: " + str(np.sum(label_coronal)) + " SUV: " + str(
            original_row["SUV"].iloc[0]), fontsize=12, color='black')

        plt.savefig(
            "/UserData/Zach_Analysis/final_testset_evaluation_vg/mips_final_test_set_3_v2/" + label_name,
            dpi=300)

        plt.close()


def plot_final_testset_for_josh_v3(df):
    # image_path_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/images4_v2/"
    # label_path_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/labels4_v2/"
    #image_path_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/images5/"
    #label_path_base = "/mnt/Bradshaw/UW_PET_Data/resampled_cropped_images_and_labels/labels5/"

    image_path_root = "/mnt/Bradshaw/UW_PET_Data/SUV_images/"
    label_path_base = "/mnt/Bradshaw/UW_PET_Data/raw_nifti_uw_pet/uw_labels_v4_nifti/"


    original_df = pd.read_excel(
        "/UserData/Zach_Analysis/suv_slice_text/uw_all_pet_preprocess_chain_v4/removed_wrong_suv_max_and_slices_13.xlsx")
    crop_df = pd.read_excel(
        "/UserData/Zach_Analysis/suv_slice_text/uw_all_pet_preprocess_chain_v4/crop_offset_lookup.xlsx")
    i = 0

    for index, row in df.iterrows():
        print(f"index: {index}")
        i += 1

        petlymph = extract_image_id(row["image"])

        dims = get_slice_thickness(petlymph)
        print(f"dims: {dims}")
        image_path_base = os.path.join(image_path_root, petlymph)

        file_names = os.listdir(image_path_base)
        index_of_suv = [index for index, element in enumerate(file_names) if "suv" in element.lower()]
        image_path = os.path.join(image_path_base, file_names[index_of_suv[0]])

        index_of_ct = [index for index, element in enumerate(file_names) if "ct" in element.lower()]
        ct_image_path = os.path.join(image_path_base, file_names[index_of_ct[0]])
        #image_path = os.path.join(image_path_base, petlymph, + "_suv_cropped.nii.gz")
        #ct_image_path = os.path.join(image_path_base, petlymph + "_ct_cropped.nii.gz")
        print(f"image name: {petlymph}")
        # gets location of label nifti
        label_name = row["label_name"]
        label_path = os.path.join(label_path_base, label_name + ".nii.gz")

        ct_image = nib.load(ct_image_path)
        ct_volume = ct_image.get_fdata()
        print(f"ct dimensions: {ct_volume.shape}")
        slice_num = row["slice_num"]
        # k_num = row["k"]
        original_row = original_df.loc[original_df['Label_Name'] == label_name]
        #crop_row = crop_df.loc[crop_df['id'] == petlymph]
        #crop_offset = crop_row['crop_offset'].iloc[0]
        # print(f"crop offset: {crop_offset}")
        # k_num = original_row["k"]
        #k_num = -1 * slice_num + crop_offset - 1
        # crop_offset = 1
        # slice_estimation = (slice_num - (crop_offset - 1)) * (dims[2] / 3)
        # slice_estimation = int(np.round(slice_estimation))
        # print(f"k num:{k_num}")
        # transaxial_slice = ct_volume[:, :, slice_num]

        # loads in the image as a numpy array
        nii_image = nib.load(image_path)
        img = nii_image.get_fdata()
        # print(f"pet image dimensions: {img.shape}")

        """
        # loads in the label as a numpy array
        nii_label = nib.load(label_path)
        label = nii_label.get_fdata()

        if os.path.exists('/mnt/Bradshaw/UW_PET_Data/raw_nifti_uw_pet/resampled_labels/' + row["Label_Name"] + '.nii.gz'):
            ct_label = nib.load('/mnt/Bradshaw/UW_PET_Data/raw_nifti_uw_pet/resampled_labels/' + row["Label_Name"] + '.nii.gz')
        else:
            ct_label = nib.Nifti1Image(label, affine=ct_volume.affine, header=ct_volume.header)
            #ct_label = resample_image(label, ct_volume.shape)
            nib.save(ct_label, '/mnt/Bradshaw/UW_PET_Data/raw_nifti_uw_pet/resampled_labels/' + row["Label_Name"] + '.nii.gz')
        """

        if os.path.exists(label_path):
            # loads in the label as a numpy array
            nii_label = nib.load(label_path)
            label = nii_label.get_fdata()  # the label data
        else:
            continue

        # flip data
        label = np.flip(label, axis=0)
        ct_volume = np.flip(ct_volume, axis=0)
        img = np.flip(img, axis=0)

        """
        # Checking if the resampled label file already exists
        resampled_path = '/mnt/Bradshaw/UW_PET_Data/resampled_labels_final_test_set/' + row["Label_Name"] + '.nii.gz'
        if os.path.exists(resampled_path):
            ct_label = nib.load(resampled_path)
            print("found label")
        else:
            print("resampling label")
            ct_label = resample_image(label, ct_volume.shape)
            # If it doesn't exist, create a new NIfTI image using the label array, affine, and header from the original label image
            ct_label = nib.Nifti1Image(ct_label, affine=nii_label.affine, header=nii_label.header)
            # Optionally, save it if needed
            nib.save(ct_label, resampled_path)
        """
        # k_num = int(-1*np.round(slice_estimation))
        # print(k_num)

        sums = np.sum(label, axis=(0, 1))
        # Find the index of the slice with the maximum sum
        k_num = np.argmax(sums)

        ct_label = label
        # ct_label = ct_label.get_fdata()
        # ct_label = np.round(ct_label).astype(int)
        # print(f"ct label dimensions: {ct_label.shape} sum: {np.sum(ct_label)}")
        transaxial_slice = ct_volume[:, :, k_num]

        mip_coronal = np.max(img, axis=1)
        mip_sagittal = np.max(img, axis=0)  # I think
        # mip_axial = np.max(img, axis=2) # I think this axis is right
        mip_axial = transaxial_slice
        # mip_coronal = normalize_mip(mip_coronal)

        label_coronal = np.max(label, axis=1)
        label_sagittal = np.max(label, axis=0)
        # label_axial = np.max(label, axis=2)
        # label_axial = np.max(ct_label[:, :, slice_num])
        label_axial = np.max(ct_label, axis=2)

        # mip_coronal = np.rot90(mip_coronal)
        # label_coronal = np.rot90(label_coronal) #label_coronal.T

        # Rotate the images for better display
        mip_coronal = np.rot90(mip_coronal)
        label_coronal = np.rot90(label_coronal)
        mip_sagittal = np.rot90(mip_sagittal)
        label_sagittal = np.rot90(label_sagittal)
        mip_axial = np.rot90(mip_axial, k=-1)
        label_axial = np.rot90(label_axial, k=-1)
        # outline = create_label_outline(label_axial)

        mip_coronal = np.fliplr(mip_coronal)
        label_coronal = np.fliplr(label_coronal)

        plt.figure(figsize=(24, 12))
        plt.subplot(1, 7, 1)  # 1 row, 2 columns, first subplot
        plt.imshow(mip_coronal, cmap='gray',
                   vmax=10)  # 'viridis' is a colormap, you can choose others like 'gray', 'plasma', etc.

        # switch the axis plotting of the y axis
        locs, _ = plt.yticks()
        y_min, y_max = plt.ylim()
        filtered_locs = [loc for loc in locs if loc > -1 and loc < mip_coronal.shape[0]]
        filtered_labels = [f"{int(y_max - loc) * -1}" for loc in filtered_locs]
        plt.yticks(filtered_locs, filtered_labels)

        label = label_coronal
        # Set zeros in the second array to NaN for transparency
        label = np.where(label == 1, 250, label)
        array_label_nan = np.where(label == 0, np.nan, label)

        # Extract voxel dimensions (in mm)
        voxel_dims = nii_image.header.get_zooms()
        print(f'pet voxel dims: {voxel_dims}')
        ct_dims = ct_image.header.get_zooms()
        print(f"ct voxel dims: {ct_dims}")
        # Plotting
        plt.figure(figsize=(24, 12))

        ax1 = plt.subplot(1, 7, 1)
        ax1.imshow(mip_coronal, cmap='gray_r', vmax=10, aspect=voxel_dims[0] / voxel_dims[1])
        ax1.set_title('MIP')

        ax2 = plt.subplot(1, 7, 2)
        ax2.imshow(mip_coronal, cmap='gray_r', vmax=10, aspect=voxel_dims[0] / voxel_dims[1])
        ax2.imshow(np.where(label_coronal == 1, 250, np.nan), cmap='spring', alpha=0.9,
                   aspect=voxel_dims[0] / voxel_dims[1])
        ax2.set_title('Coronal')

        ax3 = plt.subplot(1, 7, 3)
        ax3.imshow(mip_sagittal, cmap='gray_r', vmax=10, aspect=voxel_dims[0] / voxel_dims[1])
        ax3.imshow(np.where(label_sagittal == 1, 250, np.nan), cmap='spring', alpha=0.9,
                   aspect=voxel_dims[0] / voxel_dims[1])
        ax3.set_title('Sagittal')

        # Flip the image data horizontally
        #mip_axial = np.fliplr(mip_axial)
        #label_axial = np.fliplr(label_axial)

        # Assuming label_axial needs to be the same size as mip_axial
        scale_x = mip_axial.shape[0] / label_axial.shape[0]
        scale_y = mip_axial.shape[1] / label_axial.shape[1]
        label_axial_resized = zoom(label_axial, (scale_x, scale_y), order=0)  # order=0 for nearest-neighbor interpolation
        label_axial = label_axial_resized
        #print(mip_axial.shape, label_axial.shape)
        #print(mip_axial.dtype, label_axial.dtype)

        ax4 = plt.subplot(1, 7, 4)
        ax4.imshow(mip_axial, cmap='gray', vmax=600, vmin=-300)
        ax4.set_title(f'Axial Slice: {slice_num}')

        ax5 = plt.subplot(1, 7, 5)
        ax5.imshow(mip_axial, cmap='gray', vmax=600, vmin=-300)
        ax5.imshow(np.where(label_axial == 1, 250, np.nan), cmap='spring', alpha=0.9, aspect=voxel_dims[0] / voxel_dims[1])
        # plt.imshow(np.where(outline == 1, 250, np.nan) , cmap='spring', alpha=0.9) # Overlay the outline in 'spring' colormap
        ax5.set_title(f'Axial Slice: {slice_num} With Label')

        ax6 = plt.subplot(1, 7, 6)
        ax6.imshow(mip_axial, cmap='gray', vmax=600, vmin=-300)
        ax6.imshow(np.where(label_axial == 1, 250, np.nan), cmap='spring', alpha=0.9,
                   aspect=voxel_dims[0] / voxel_dims[1])
        # plt.imshow(np.where(outline == 1, 250, np.nan) , cmap='spring', alpha=0.9) # Overlay the outline in 'spring' colormap
        ax6.set_title(f'Axial Slice: {slice_num} With Label')

        ax7 = plt.subplot(1, 7, 7)
        ax7.imshow(mip_axial, cmap='gray', vmax=600, vmin=-300)
        ax7.imshow(np.where(label_axial == 1, 250, np.nan), cmap='spring', alpha=0.9,
                   aspect=voxel_dims[0] / voxel_dims[1])
        # plt.imshow(np.where(outline == 1, 250, np.nan) , cmap='spring', alpha=0.9) # Overlay the outline in 'spring' colormap
        ax7.set_title(f'Axial Slice: {slice_num} With Label')

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
        # print(original_row)
        sentence = original_row["sentence"].iloc[0]
        # print(sentence)
        # print(type(sentence))
        sentence = insert_newlines(sentence, word_limit=25)
        plt.suptitle(sentence + " Pixels: " + str(np.sum(label_coronal)) + " SUV: " + str(
            original_row["SUV"].iloc[0]), fontsize=12, color='black')

        plt.savefig(
            "/UserData/Zach_Analysis/final_testset_evaluation_vg/mips_final_test_set_3_v3/" + label_name,
            dpi=300)

        plt.close()