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



def get_pixels_right_of_plane(data, z_plane):
    """
    Extracts pixels with non-zero labels to the right of the plane z = z_plane.

    :param data: 3D numpy array with labeled pixels.
    :param z_plane: The z-coordinate of the plane.
    :return: List of tuples representing the coordinates of the selected pixels.
    """
    z_plane = math.floor(z_plane)
    selected_pixels = set()

    # Iterate over the array and check the condition
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            for z in range(z_plane, data.shape[2]):
                if data[x, y, z] != 0:
                    #print(data[x,y,z])
                    selected_pixels.add(data[x,y,z])

    return selected_pixels

def filter_pixels(array, valid_values):
    # Create an empty array of the same shape as the input array
    filtered_array = np.zeros_like(array)

    # Iterate over each element in the array
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            for k in range(array.shape[2]):
                if array[i, j, k] in valid_values:
                    filtered_array[i, j, k] = 1 #array[i, j, k]
    return filtered_array
def make_connected_component_labels():

    file_location = "/UserData/Zach_Analysis/cog_sub_region_text/clavicula_locations_v3.xlsx"
    df = pd.read_excel(file_location)
    imageTr_path = "/UserData/Xin/Monai_Auto3dSeg/COG_lymph_seg/Auto3dseg/data/COG/labelsTr"


    files_skip = []
    for index, row in df.iterrows():
        print(f"index: {index}")
        # if row["file"].contains["850410"] or row["file"]: # == "COG_AHO33_850410_baseline.nii.gz" or row["file"] == "COG_AHO33_851194_baseline.nii.gz":
        #    continue
        skip_flag = False
        for string in files_skip:
            if string in row["FileName"]:
                skip_flag = True

        if skip_flag:
            continue
        #if index < 78:
        #   continue
        # f index == 20:
        #   break
        # gets the file path to be loaded in
        image_path = row["FileName"][:25] + row["FileName"][25:]
        file_path = os.path.join(imageTr_path, image_path)
        print(file_path)
        # loads in the data in file_path and projects it down to a 2d coronal mip
        try:
            img = resample_nii(file_path, (3, 3, 3))
        except FileNotFoundError:
            print(f"cannot file file: {file_path}")
            continue

        img = img.get_fdata()

        # gets all the connected components and labels them with their number 1 to n
        labels_out = cc3d.connected_components(img)

        min_z = math.floor(np.min((row["z1"], row["z2"])))
        lesion_labels = get_pixels_right_of_plane(labels_out, min_z)

        filtered_array = filter_pixels(labels_out, lesion_labels)

        mip_2d_axis1 = np.max(filtered_array, axis=1)

        save_location = "/UserData/Zach_Analysis/cog_data_splits/mips/head_and_neck_mips/labelsTr/" + image_path +"_head_neck_label" + ".png"

        # Convert the image to PIL Image format and ensure it's in 'L' mode for grayscale
        pil_image = Image.fromarray(filtered_array).convert('L')

        # Save the image in a lossless format (PNG)
        pil_image.save(save_location, format='PNG')