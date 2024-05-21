

import pandas as pd
import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt


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


def plot_for_label_accuracy_assessment_v2(df):


    image_path_base = "/mnt/Bradshaw/UW_PET_Data/SUV_images/"
    label_path_base = "/mnt/Bradshaw/UW_PET_Data/raw_nifti_uw_pet/uw_labels_v2_nifti/"
    labels_to_skip = ["PETWB_006370_04_label_2", "PETWB_011355_01_label_5", "PETWB_002466_01_label_1",
                      "PETWB_012579_01_label_2", "PETWB_003190_01_label_3",
                      "PETWB_011401_02_label_3"]

    for index, row in df.iterrows():

        print(f"index: {index}")

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

        # loads in the image as a numpy array
        nii_image = nib.load(image_path)
        img = nii_image.get_fdata()

        # loads in the label as a numpy array
        nii_label = nib.load(label_path)
        label = nii_label.get_fdata()



        mip_coronal = np.max(img, axis=1)
        mip_sagital = np.max(img, axis=0) # I think
        mip_axial = np.max(img, axis=2) # I think this axis is right
        #mip_coronal = normalize_mip(mip_coronal)

        label_coronal = np.max(label, axis=1)
        label_sagittal = np.max(label, axis=0)
        label_axial = np.max(label, axis=2)

        #mip_coronal = np.rot90(mip_coronal)
        #label_coronal = np.rot90(label_coronal) #label_coronal.T

        # Rotate the images for better display
        mip_coronal = np.rot90(mip_coronal)
        label_coronal = np.rot90(label_coronal)
        mip_sagittal = np.rot90(mip_sagittal)
        label_sagittal = np.rot90(label_sagittal)
        mip_axial = np.rot90(mip_axial)
        label_axial = np.rot90(label_axial)

        plt.figure(figsize=(24, 10))
        plt.subplot(1, 4, 1)  # 1 row, 2 columns, first subplot
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

        plt.subplot(1, 4, 2)  # 1 row, 2 columns, second subplo
        # Plot the two numpy arrays overtop of each other
        plt.imshow(mip_coronal, cmap='gray', vmax=10)  # First array with alpha of 0.1
        plt.imshow(array_label_nan, cmap='spring', alpha=0.9)  # Second array over the first, with alpha of 0.1
        #plt.xticks(locs, labels=[f"{int(mip_coronal.shape[1] - loc)}" for loc in locs])

        locs, _ = plt.yticks()
        y_min, y_max = plt.ylim()
        filtered_locs = [loc for loc in locs if loc > -1 and loc < mip_coronal.shape[0]]
        filtered_labels = [f"{int(y_max - loc)*-1}" for loc in filtered_locs]
        plt.yticks(filtered_locs, filtered_labels)


        sentence = row["sentence"] + " pixels: " + str(np.sum(label_coronal))
        #sentence = row["Extracted Sentences"] + " pixels: " + str(np.sum(label_coronal))

        sentence = insert_newlines(sentence, word_limit=17)
        plt.suptitle(sentence, fontsize=12, color='black')

        plt.savefig("/UserData/Zach_Analysis/petlymph_image_data/prediction_mips_for_presentations/mips_accuracy_assessment/" + label_name)

        plt.close()




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
        slice_num = row["Slice"]
        transaxial_slice = ct_volume[:, :, slice_num]

        # loads in the image as a numpy array
        nii_image = nib.load(image_path)
        img = nii_image.get_fdata()

        # loads in the label as a numpy array
        nii_label = nib.load(label_path)
        label = nii_label.get_fdata()

        mip_coronal = np.max(img, axis=1)
        mip_sagittal = np.max(img, axis=0) # I think
        #mip_axial = np.max(img, axis=2) # I think this axis is right
        mip_axial = transaxial_slice
        #mip_coronal = normalize_mip(mip_coronal)

        label_coronal = np.max(label, axis=1)
        label_sagittal = np.max(label, axis=0)
        label_axial = np.max(label, axis=2)

        #mip_coronal = np.rot90(mip_coronal)
        #label_coronal = np.rot90(label_coronal) #label_coronal.T

        # Rotate the images for better display
        mip_coronal = np.rot90(mip_coronal)
        label_coronal = np.rot90(label_coronal)
        mip_sagittal = np.rot90(mip_sagittal)
        label_sagittal = np.rot90(label_sagittal)
        mip_axial = np.rot90(mip_axial)
        label_axial = np.rot90(label_axial)

        plt.figure(figsize=(24, 10))
        plt.subplot(1, 4, 1)  # 1 row, 2 columns, first subplot
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

        ax1 = plt.subplot(1, 4, 1)
        ax1.imshow(mip_coronal, cmap='gray', vmax=10)
        ax1.set_title('MIP')

        ax2 = plt.subplot(1, 4, 2)
        ax2.imshow(mip_coronal, cmap='gray', vmax=10)
        ax2.imshow(np.where(label_coronal == 1, 250, np.nan), cmap='spring', alpha=0.9)
        ax2.set_title('Coronal')

        ax3 = plt.subplot(1, 4, 3)
        ax3.imshow(mip_sagittal, cmap='gray', vmax=10)
        ax3.imshow(np.where(label_sagittal == 1, 250, np.nan), cmap='spring', alpha=0.9)
        ax3.set_title('Sagittal')

        ax4 = plt.subplot(1, 4, 4)
        ax4.imshow(mip_axial, cmap='gray', vmax=10)
        ax4.imshow(np.where(label_axial == 1, 250, np.nan), cmap='spring', alpha=0.9)
        ax4.set_title('Axial')

        plt.suptitle(row["sentence"] + " pixels: " + str(np.sum(label_coronal)), fontsize=12, color='black')

        plt.savefig("/UserData/Zach_Analysis/petlymph_image_data/prediction_mips_for_presentations/mips_accuracy_assessment/" + label_name)

        plt.close()