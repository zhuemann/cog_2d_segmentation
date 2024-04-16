
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

def plot_mips_with_labels(df):


    df_path = "C:/Users/zmh001/PET_visual_grounding/petlymph_visual_grounding_df_drop_non_anatomical_sents.xlsx"
    #df_path = "Z:/Zach_Analysis/petlymph_image_data/unique_labels_uw_lymphoma_anon_4_renumbered_v3.xlsx"
    #df = pd.read_excel(df_path)

    image_path_base = "/UserData/Zach_Analysis/suv_nifti/"
    label_path_base = "/UserData/Zach_Analysis/petlymph_image_data/labelsv2/"
    label_path_base = "/UserData/Zach_Analysis/petlymph_image_data/labels_v6_nifti_test_short/"


    for index, row in df.iterrows():

        print(f"index: {index}")
        if index == 10:
            break
        petlymph = row["Petlymph"]

        # gets the location of the suv converted image if it exists
        folder_name = str(petlymph) + "_" + str(petlymph)
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
        #print(f"sum of pos pixels: {np.sum(label_coronal)}")
        # plt.imshow(mip_coronal, cmap='gray')  # Use an appropriate colormap
        # plt.imshow(label_coronal, cmap="jet", alpha=.2)
        # plt.colorbar()  # Optional, adds a colorbar to show the mapping of values to colors
        # plt.title('2D Maximum Projection')
        # plt.show()
        #label_name = row["Label_Name"]
        # print(img.shape)
        #filename_img = "/UserData/Zach_Analysis/petlymph_image_data/images_coronal_mip_v4/" + str(petlymph) + ".png"
        #filename_label = "/UserData/Zach_Analysis/petlymph_image_data/labels_coronal_mip_v6/" + str(label_name) + ".png"
        # save_as_dicom(mip_coronal, filename)
        #save_2d_image_lossless(mip_coronal, filename_img)
        #save_2d_image_lossless(label_coronal, filename_label)
        mip_coronal = np.rot90(mip_coronal)
        label_coronal = np.rot90(label_coronal) #label_coronal.T
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
        plt.imshow(mip_coronal, cmap='gray', vmax = 10)  # 'viridis' is a colormap, you can choose others like 'gray', 'plasma', etc.

        print(mip_coronal.shape)
        locs, _ = plt.yticks()
        y_min, y_max = plt.ylim()
        plt.yticks(locs, labels=[f"{int(y_max - (loc - y_min))}" for loc in locs])


        label = label_coronal
        # Set zeros in the second array to NaN for transparency
        label = np.where(label == 1, 250, label)
        array_label_nan = np.where(label == 0, np.nan, label)

        plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplo
        # Plot the two numpy arrays overtop of each other
        plt.imshow(mip_coronal, cmap='gray', vmax=10)  # First array with alpha of 0.1
        plt.imshow(array_label_nan, cmap='spring', alpha=0.9)  # Second array over the first, with alpha of 0.1
        #plt.xticks(locs, labels=[f"{int(mip_coronal.shape[1] - loc)}" for loc in locs])

        locs, _ = plt.yticks()
        y_min, y_max = plt.ylim()
        #for loc in locs:
        #    print(loc)
        #plt.yticks(locs, labels=[f"{int(y_max - (loc - y_min))}" for loc in locs])
        #plt.yticks(locs, labels=[f"{int(y_max - loc)}" for loc in locs if loc > -1 and loc < mip_coronal.shape[0]])
        # Filter locations and generate labels simultaneously
        filtered_locs = [loc for loc in locs if loc > -1 and loc < mip_coronal.shape[0]]
        filtered_labels = [f"{int(y_max - loc)*-1}" for loc in filtered_locs]

        # Apply the filtered locations and labels
        plt.yticks(filtered_locs, filtered_labels)



        sentence = row["Extracted Sentences"] + " pixels: " + str(np.sum(label_coronal))
        #print(sentence)
        sentence = insert_newlines(sentence, word_limit=20)
        #print(f"sum of pos pixels: {np.sum(label)}")
        plt.suptitle(sentence, fontsize=12, color='black')

        plt.savefig("/UserData/Zach_Analysis/petlymph_image_data/prediction_mips_for_presentations/mip_plots/" + label_name)
        plt.close()
        #plt.show()