import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import ast
import nibabel as nib
from nilearn.image import resample_img
import re

def resample_nii(file_path, new_voxel_size): #, output_path):
    """
    Resample a NIfTI (.nii.gz) file to a new voxel size.

    Parameters:
    file_path (str): Path to the input .nii.gz file.
    new_voxel_size (tuple of float): The desired voxel size, e.g., (3, 3, 3).
    output_path (str): Path to save the resampled image.
    """
    # Load the .nii.gz file
    nii_image = nib.load(file_path)

    # Get the current affine
    affine = nii_image.affine

    # Create a new affine matrix for the desired voxel size
    new_affine = affine.copy()
    for i in range(3):
        new_affine[i, i] = new_voxel_size[i] * affine[i, i] / nii_image.header.get_zooms()[i]

    # Resample the image
    resampled_img = resample_img(nii_image, target_affine=new_affine, interpolation='nearest')

    return resampled_img
"""
def convert_to_list(series_element):
    # Strip leading/trailing spaces and brackets
    formatted_str = series_element.strip('[] ')

    # Replace spaces with commas
    formatted_str = formatted_str.replace(' ', ',')

    # Replace two commas in a row with a single comma
    formatted_str = formatted_str.replace(',,', ',')

    # Convert to a list using ast.literal_eval
    return ast.literal_eval('[' + formatted_str + ']')
"""

def convert_to_list(series_element):
    # Strip leading/trailing spaces and brackets
    formatted_str = series_element.strip('[] ')

    # Replace spaces with commas
    formatted_str = formatted_str.replace(' ', ',')

    # Use regular expression to replace multiple commas with a single comma
    formatted_str = re.sub(r',+', ',', formatted_str)

    # Convert to a list using ast.literal_eval
    return ast.literal_eval('[' + formatted_str + ']')

def make_clavicular_mips():


    file_location = "/UserData/Zach_Analysis/cog_sub_region_text/clavicula_locations_v2.xlsx"
    df = pd.read_excel(file_location)
    imageTr_path = "/UserData/Xin/Monai_Auto3dSeg/COG_lymph_seg/Auto3dseg/data/COG/imagesTr"


    # files_skip = ["850410", "851194", "851301", "851942", "852438"]
    files_skip = ["863930", "869757", "869859", "870197"]
    files_skip = []
    #files_skip = []863930_
    for index, row in df.iterrows():
        print(f"index: {index}")
        # if row["file"].contains["850410"] or row["file"]: # == "COG_AHO33_850410_baseline.nii.gz" or row["file"] == "COG_AHO33_851194_baseline.nii.gz":
        #    continue
        skip_flag = False
        for string in files_skip:
            # print(string in row["file"])
            # print(row["file"])
            if string in row["FileName"]:
                skip_flag = True

        if skip_flag:
            continue
        #if index < 78:
        #   continue
        # f index == 20:
        #   break
        # gets the file path to be loaded in
        image_path = row["FileName"][:25] + "_0000" + row["FileName"][25:]
        file_path = os.path.join(imageTr_path, image_path)
        print(file_path)
        # loads in the data in file_path and projects it down to a 2d coronal mip
        try:
            img = resample_nii(file_path, (3, 3, 3))
        except FileNotFoundError:
            print(f"cannot file file: {file_path}")
            continue
        # img = nib.load(file_path)
        volume_data = img.get_fdata()
        mip_2d_axis1 = np.max(volume_data, axis=1)

        mask_coordinates = df[df['FileName'] == row["FileName"]]
        #left_clavicula = mask_coordinates["left clavicula"]
        #right_clavicula = mask_coordinates["right clavicula"]
        left_clavicula = [mask_coordinates["x1"], mask_coordinates["y1"], mask_coordinates["z1"]]
        right_clavicula = [mask_coordinates["x2"], mask_coordinates["y2"], mask_coordinates["z2"]]
        # print(type(left_clavicula.iloc[0]))
        # print(left_clavicula)
        # print(right_clavicula)
        # print(left_clavicula.iloc[0])
        # print(type(right_clavicula.iloc[0]))
        # print(right_clavicula.iloc[0][1])

        # Convert the string representations of the lists back to actual lists
        #left_point = convert_to_list(left_clavicula.iloc[0])
        #right_point = convert_to_list(right_clavicula.iloc[0])
        left_point = left_clavicula
        right_point = right_clavicula
        print(type(left_point))
        print(left_point)
        print(right_point)

        plt.imshow(mip_2d_axis1, cmap='gray', vmin=0, vmax=5)
        plt.colorbar()
        plt.title("2D Image Display")

        # plt.plot([left_clavicula[0], right_clavicula[0]], [left_clavicula[1], right_clavicula[1]], color='red')  # Change color as needed
        plt.plot([left_point[2], right_point[2]], [left_point[0], right_point[0]], color='red')

        plt.savefig("/UserData/Zach_Analysis/cog_sub_region_text/clavicula_line_plots_v3/plot_"+ str(row['FileName']) + ".png")
        #plt.show()
        plt.clf()