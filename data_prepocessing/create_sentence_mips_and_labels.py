import numpy as np
import os
import pandas
import regex as re
import sys
import pandas as pd
import os
import nibabel as nib
from datetime import datetime
import numpy as np
import glob
import cc3d
import numpy as np
import numpy as np
from PIL import Image

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

def create_mips(df):
    #df_path = "/UserData/Zach_Analysis/suv_slice_text/uw_lymphoma_preprocess_chain_v2/unique_labels_uw_lymphoma_anon_4_renumbered.xlsx"
    #df_path = "/UserData/Zach_Analysis/suv_slice_text/uw_lymphoma_preprocess_chain_v2/unique_labels_uw_lymphoma_anon_4_renumbered_v3.xlsx"
    #df = pd.read_excel(df_path)
    print(len(df))
    print(df)

    df_decode_path = "/UserData/Zach_Analysis/patient_decoding.xlsx"
    df_petlymph = pandas.read_excel(df_decode_path)

    found = 0
    missing_conversion = 0
    data_df = []

    image_path_base = "/UserData/Zach_Analysis/suv_nifti/"
    #label_path_base = "/UserData/Zach_Analysis/petlymph_image_data/labelsv2/"
    label_path_base = "/UserData/Zach_Analysis/petlymph_image_data/labels_v9_nifti/"

    for index, row in df.iterrows():

        # if index > 2:
        #    break
        # get the petlymph number if availible
        print(f"index: {index}")
        #if index < 580:
        #    continue
        """
        petlymph = df_petlymph[df_petlymph["Accession Number"] == row["Accession Number"]]
        if len(petlymph) == 0:
            missing_conversion += 1
            continue
        else:
            petlymph = petlymph["Patient ID"].iloc[0]
            found += 1
        """
        petlymph = row["Petlymph"]
        #if petlymph == "PETLYMPH_3501" or petlymph == "PETLYMPH_2650" or petlymph == "PETLYMPH_3100":
        #    continue
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
        mip_coronal = normalize_mip(mip_coronal)

        label_coronal = np.max(label, axis=1)

        # plt.imshow(mip_coronal, cmap='gray')  # Use an appropriate colormap
        # plt.imshow(label_coronal, cmap="jet", alpha=.2)
        # plt.colorbar()  # Optional, adds a colorbar to show the mapping of values to colors
        # plt.title('2D Maximum Projection')
        # plt.show()
        label_name = row["Label_Name"]
        # print(img.shape)
        filename_img = "/UserData/Zach_Analysis/petlymph_image_data/images_coronal_mip_v10/" + str(petlymph) + ".png"
        filename_label = "/UserData/Zach_Analysis/petlymph_image_data/labels_coronal_mip_v10/" + str(label_name) + ".png"
        # save_as_dicom(mip_coronal, filename)
        save_2d_image_lossless(mip_coronal, filename_img)
        save_2d_image_lossless(label_coronal, filename_label)

    print(missing_conversion)
    print(found)
