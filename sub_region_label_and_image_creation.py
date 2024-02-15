import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import ast
import nibabel as nib
from nilearn.image import resample_img
import re
import cc3d
import math
from PIL import Image
from scipy.ndimage import zoom


def resample_image(image, original_spacing, new_spacing):
    """
    Resample the 3D image to the new spacing.
    """
    # Calculate the zoom factors
    zoom_factors = [o / n for o, n in zip(original_spacing, new_spacing)]
    # Use scipy's zoom function to resample
    resampled_image = zoom(image.get_fdata(), zoom_factors, order=3)
    return resampled_image


def make_mips_from_3d_data():

    # load in all of the refreence points, clavicle, t1 t12
    file_location = "/UserData/Zach_Analysis/cog_sub_region_text/clavicula_locations_v2.xlsx"
    clavicle_df = pd.read_excel(file_location)
    # location of 3d
    image_Tr_path = "/UserData/Xin/Monai_Auto3dSeg/COG_lymph_seg/Auto3dseg/data/COG/imagesTr"
    label_Tr_path = "/UserData/Xin/Monai_Auto3dSeg/COG_lymph_seg/Auto3dseg/data/COG/labelsTr"

    # for each image
    for index, row in clavicle_df.iterrows():
        print(f"index: {index}")

        image_name =  row["FileName"][:25] + "_0001" + row["FileName"][25:]
        print(image_name)
        image_path =  os.path.join(image_Tr_path, image_name)
        label_name = row["FileName"][:25] + row["FileName"][25:]
        label_path = os.path.join(label_Tr_path, label_name)
        #print(file_path)
        # loads in the data in file_path and resizes it to 3x3x3
        try:
            original_spacing = (2, 2, 2)  # Assuming the original spacing is 2mm x 2mm x 2mm
            new_spacing = (3, 3, 3)

            original_image_volume = nib.load(image_path)
            image_volume = resample_image(original_image_volume, original_spacing, new_spacing)

            original_label_volume = nib.load(label_path)
            label_volume = resample_image(original_label_volume, original_spacing, new_spacing)

        except FileNotFoundError:
            print(f"cannot file file: {image_path}")
            continue

        image_2d = np.max(image_volume, axis=1)
        # Apply thresholding: keep values above 0.2 unchanged, set others to 0
        image_2d_threshold = np.where(image_2d > 0.2, image_2d, 0)
        # clip values above 11
        image_2d = np.clip(image_2d_threshold, None, 11)

        # Save the 2D projection as a new NIfTI file
        # Since we are saving a 2D image, we need to adjust the affine accordingly
        affine_2d = original_image_volume.affine.copy()
        affine_2d[2, 2] = 1  # Adjust the z-axis scaling factor to avoid flattening in the saved file

        # Create a new NIfTI image for the 2D projection
        # Note: We add a new axis to make it a 3D array with a single slice, as NIfTI expects 3D data
        nifti_img_2d = nib.Nifti1Image(image_2d[np.newaxis, :, :], affine_2d)

        # Save the 2D NIfTI image
        output_filename_2d = "/UserData/Zach_Analysis/cog_data_splits/mips/head_and_neck_mips/imageTr/" + row["FileName"][:25] + "_baseline" + "_0001" +"_coronal"+ row["FileName"][25:]
        nib.save(nifti_img_2d, output_filename_2d)

    # load in the 3d image and resize it to 3x3x3mm then do the projection down to 2d

    # do the cropping to 11 suv but rescale them back to 1024 after the cropping. # also look into the bit type of the nii file

    # resize the label to 3x3x3 mm and then round .5 and up to 1 and .5 below to 0 then get the connected components
    # get the different mask regions depending on the connected compoents.
    # create the head and neck if it is to the right of the clavicular
    # create the chest labels if it is between the t1 and t12
    # create the abdomen/pelvis if it is below the t12
    # also include the total label


    # save the 2d image, the head and neck label, chest label, pelvis label