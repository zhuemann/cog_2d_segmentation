import os
import pydicom
import dicom2nifti
from platipy.dicom.io import rtstruct_to_nifti
import nibabel as nib
from datetime import datetime
import numpy as np
import glob
import cc3d
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def is_boundary_pixel(pixel, pixel_set):
    """
    Check if a pixel is a boundary pixel. Can check to see if any of the pixels immediate neighbors, up, down, left, right,
    forward, back is not in the set


    :param pixel: The (x, y, z) coordinates of the pixel to check.
    :param pixel_set: A set of tuples representing all pixels.
    :return: True if the pixel is on the boundary; False otherwise.
    """
    x, y, z = pixel
    # Define the 6 directions to check for neighbors
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    # Check each direction to see if a neighbor is missing
    for dx, dy, dz in directions:
        neighbor = (x + dx, y + dy, z + dz)
        if neighbor not in pixel_set:
            return True  # At least one neighbor is missing, so it's a boundary pixel

    return False  # All neighbors are present


def get_boundary_pixels(pixel_set):
    """
    Get the set of boundary pixels from a set of pixels.

    :param pixel_set: A set of tuples representing all pixels.
    :return: A set of tuples representing the boundary pixels.
    """
    boundary_pixels = set()

    # Check each pixel to see if it's a boundary pixel
    for pixel in pixel_set:
        if is_boundary_pixel(pixel, pixel_set):
            boundary_pixels.add(pixel)

    return boundary_pixels


def extend_pixels_21_neighbors(pixels):
    """
    Extend a set of pixels by including all neighbors within a 3x3x3 cube centered
    on each pixel in 3D space, excluding the corners of the cube.

    :param pixels: A set of tuples, where each tuple represents the (x, y, z) coordinates of a pixel.
    :return: A set containing the original and new extended pixels.
    """
    extended_pixels = set(pixels)  # Start with the original set of pixels

    # Define the 21 non-corner directions in a 3x3x3 cube
    directions = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                # Exclude the center (0, 0, 0) and the corners
                if (dx, dy, dz) != (0, 0, 0) and abs(dx) + abs(dy) + abs(dz) < 3:
                    directions.append((dx, dy, dz))

    # Iterate through each pixel in the original set
    for x, y, z in pixels:
        # Add each of the 21 neighboring pixels
        for dx, dy, dz in directions:
            # Calculate new pixel coordinates
            new_x, new_y, new_z = x + dx, y + dy, z + dz
            # Add the new pixel to the set
            extended_pixels.add((new_x, new_y, new_z))

    return extended_pixels

def extend_pixels_6_neighbors(pixels, extension=1):
    """
    Extend a set of pixels by 1 in each of the x, y, and z directions in 3D space,
    giving only 6 direct neighbors for each pixel.

    :param pixels: A set of tuples, where each tuple represents the (x, y, z) coordinates of a pixel.
    :param extension: This parameter is ignored in this version as the extension is fixed to 6 neighbors.
    :return: A set containing the original and new extended pixels.
    """
    extended_pixels = set(pixels)  # Start with the original set of pixels

    # Define the 6 directional extensions
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    # Iterate through each pixel in the original set
    for x, y, z in pixels:
        # Add each of the 6 neighboring pixels
        for dx, dy, dz in directions:
            # Calculate new pixel coordinates
            new_x, new_y, new_z = x + dx, y + dy, z + dz
            # Add the new pixel to the set
            extended_pixels.add((new_x, new_y, new_z))

    return extended_pixels

def extend_pixels(pixels, extension):
    """
    Extend a set of pixels in all directions by a specified amount in 3D space.

    :param pixels: A set of tuples, where each tuple represents the (x, y, z) coordinates of a pixel.
    :param extension: The number of pixels to extend in each direction.
    :return: A set containing the original and new extended pixels.
    """
    extended_pixels = set()

    # Iterate through each pixel in the original set
    for x, y, z in pixels:
        # Generate new pixels within the extension range in all directions
        for dx in range(-extension, extension + 1):
            for dy in range(-extension, extension + 1):
                for dz in range(-extension, extension + 1):
                    # Calculate new pixel coordinates
                    new_x, new_y, new_z = x + dx, y + dy, z + dz
                    # Add the new pixel to the set
                    extended_pixels.add((new_x, new_y, new_z))

    return extended_pixels

def calculate_threshold(volume, background, source):
    first_term = .078 / volume  # volume is in ml
    second_term = .617 * (background / source)
    # third_term = .2
    third_term = .316
    #print(f"first term: {first_term}")
    #print(f"second term: {second_term}")
    #print(f"thrid term: {third_term}")
    #print(f"sum: {first_term + second_term + third_term}")
    return first_term + second_term + third_term

def get_background_value(extension, pixel_set, img):
    initial_background_sum = 0

    all_pixels = extend_pixels(pixel_set, extension)
    # print(f"all pixel length: {len(all_pixels)}")
    boundary_coordinates = get_boundary_pixels(all_pixels)
    # print(len(boundary_coordinates))
    for coor in boundary_coordinates:
        # print(coor)
        initial_background_sum += img[coor[0], coor[1], coor[2]]
    # print(f"background sum: {initial_background_sum} average: {initial_background_sum/len(boundary_coordinates)}")
    background_suv = initial_background_sum / len(boundary_coordinates)
    return background_suv

def contour_above_threshold(img, threshold, pixels, already_contoured):
    #new_contour = set([])
    new_contour = already_contoured
    for pixel in pixels:
        if img[pixel[0]][pixel[1]][pixel[2]] > threshold:
            new_contour.add(pixel)

    return new_contour

def itm(start_point, suv_max, img, conversion, exit_early):
    old_threshold = suv_max
    i, j, k = start_point

    #if suv_max > 10:
    #    suv_max = 10
    change_threshold = .01
    # print(f"start point: {start_point}")
    background = get_background_value(extension=3, pixel_set={start_point}, img=img)
    volume = 1 * conversion  # more genearlly it will be len(countour)*conversion
    source = suv_max

    # new_threshold = calculate_threshold(volume, background, source)
    new_threshold = (.617 * (background / source) + .316) * source
    #print(f"first threshold: {new_threshold}")

    # get new adjacent pixels above threshold
    #adjacent_pixels = extend_pixels({start_point}, 1)
    #adjacent_pixels = extend_pixels_6_neighbors({start_point}, 1)
    adjacent_pixels = extend_pixels_21_neighbors({start_point})
    #adjacent_pixels = extend_pixels({start_point}, 1)
    new_contour = contour_above_threshold(img, new_threshold, adjacent_pixels, set([]))
    #adjacent_pixels = extend_pixels(new_contour, 1)
    #new_contour = contour_above_threshold(img, new_threshold, adjacent_pixels)

    change = (old_threshold - new_threshold) / old_threshold
    # print(f"percent change: {change}")
    while abs(change) > change_threshold:

        # print(f"suv max: {suv_max} current_threshold: {new_threshold}")
        # calculate new boundary based on new set
        background = get_background_value(extension=3, pixel_set=new_contour, img=img)
        #if len(new_contour) < 20 and background > .7*source:
        #    print(f"old background: {background}")
        #    background = get_background_value(extension=5, pixel_set=new_contour, img=img)
        #    print(f"new background: {background}")
            #print(new_contour)
        #print(f"background value: {background}")

        # calculate volume and background
        volume = len(new_contour) * conversion
        # print(f"volume: {volume}")
        # calculate new threshold
        old_threshold = new_threshold
        new_threshold = calculate_threshold(volume, background, source) * source
        # maybe try something like this
        if new_threshold / source > .9:
            #new_threshold = .4 * source
            #canidate_pixels = extend_pixels_6_neighbors(new_contour, 1)
            #canidate_pixels = extend_pixels_21_neighbors(new_contour)
            #canidate_pixels = extend_pixels(new_contour, 1)
            #new_contour = contour_above_threshold(img, new_threshold, canidate_pixels, new_contour)
            exit_early += 1
            print(f"exited early: {exit_early} suv max: {suv_max} pixels: {len(new_contour)} final threshold: {new_threshold} background: {background}")
            return None, exit_early
            #break
        # if new_threshold + .2 > suv_max:
        #    print("stopping loop thresold greater than suv_max")
        #    break
        # print(f"threshold: {new_threshold}")
        # add all adjacent pixels above threshold to the contour
        # get new adjacent pixels above threshold
        #canidate_pixels = extend_pixels(new_contour, 1)
        canidate_pixels = extend_pixels_6_neighbors(new_contour, 1)
        #canidate_pixels = extend_pixels_21_neighbors(new_contour)
        new_contour = contour_above_threshold(img, new_threshold, canidate_pixels, new_contour)

        # if new threhold is different by more than 5 percent then continue process else break and return countour
        change = (old_threshold - new_threshold) / old_threshold
        # print(f"percent change: {change}")
    # print(len(new_contour))
    #print(f"final_threshold: {new_threshold}")
    #print(f"lesion ml: {len(new_contour) * conversion}")
    #print(f"percent SUVmax: {new_threshold / suv_max}")
    #print(f"SUVmax: {suv_max}")
    # print(f"SUVmax to background: {suv_max/background}")
    # print(f"background: {background}")
    return new_contour, exit_early

def threshold_of_max(start_point, suv_max, img):

    threshold = .5 * suv_max
    segmented_regions = img > threshold
    labels_out = cc3d.connected_components(segmented_regions, connectivity=6)

    # Extract the label of the connected component at the start point
    component_label = labels_out[start_point]

    # Create a mask for the connected component at the start point
    component_mask = labels_out == component_label

    # Extract the indices of the pixels that belong to this component
    indices = set(zip(*np.where(component_mask)))

    return indices

def single_component(original_contour, start_point):

    labeled_image = cc3d.connected_components(original_contour, connectivity=6)

    # Extract the label of the connected component at the start point
    component_label = labeled_image[start_point]

    # Create a binary mask where the selected component is marked as 1, and all others as 0
    component_mask = np.where(labeled_image == component_label, 1, 0)

    return component_mask

def make_labels_from_suv_max_points(df, save_location):
    missing_conversion = 0
    petlymph_dic = {}
    image_path_base = "Z:/Zach_Analysis/suv_nifti/"
    image_path_base = "/UserData/Zach_Analysis/suv_nifti/"
    image_path_base = "/mnt/Bradshaw/UW_PET_Data/SUV_images/"
    exit_early = 0
    drop_later = []
    for index, row in df.iterrows():

        if index % 100 == 0:
            print(f"index: {index}")

        #if index > 50:
        #    continue
        # if index < 600:
        #    break
        # if index < 3:
        #    continue
        # if index > 10:
        #    break
        # if index > 25:
        #    break
        if row["SUV"] < 2.5:
            continue
        """
        if index > 10:
            break
        

        # get the petlymph number if availible
        petlymph = df_petlymph[df_petlymph["Accession Number"] == row["Accession Number"]]
        if len(petlymph) == 0:
            missing_conversion += 1
            continue
        else:
            petlymph = petlymph["Patient ID"].iloc[0]
        """
        petlymph = row["Petlymph"]
        if petlymph in petlymph_dic:
            petlymph_dic[petlymph] += 1
        else:
            petlymph_dic[petlymph] = 1
        # gets the location of the suv converted image if it exists
        folder_name = str(petlymph) + "_" + str(petlymph)
        image_path = os.path.join(image_path_base, folder_name)
        file_names = os.listdir(image_path)
        index_of_suv = [index for index, element in enumerate(file_names) if "suv" in element.lower()]
        image_path = os.path.join(image_path, file_names[index_of_suv[0]])

        # loads in the image as a numpy array
        nii_image = nib.load(image_path)
        img = nii_image.get_fdata()
        # print(f"pet scan dimensions: {img.shape}")
        header = nii_image.header  # Retrieve the header
        pixdim = header['pixdim']  # Extract the pixdim field from the header
        dims = pixdim[1:4]  # gets the dimensions for each pixel
        volume_conversion = (dims[0] * dims[1] * dims[2]) / 1000
        # print(f"volume conversion: {volume_conversion}")

        if volume_conversion < 0.04:
            volume_conversion = 0.04346516799926758

        starting_point = (row["i"], row["j"], row["k"])
        i, j, k = starting_point
        # print(f"listed suv: {row['SUV']}")
        # print(f"extracted sent: {row['Extracted Sentences']}")

        contour, exit_early = itm(starting_point, row['SUV'], img, volume_conversion, exit_early)
        #contour = threshold_of_max(starting_point, row['SUV'], img)

        if contour == None:
            drop_later.append(row["Label_Name"])
            continue
        # print(contour)
        # print(row["Extracted Sentences"])

        #print(row["Extracted Sentences"])
        #print(f"starting point: {starting_point}")
        #plot_points_on_slices(img, contour, starting_point[2])
        label_data = np.zeros(img.shape, dtype=nii_image.get_data_dtype())
        # Set specified pixels to 1
        for x, y, z in contour:
            label_data[x, y, z] = 1

        print(f"sum before: {np.sum(label_data)}")
        # make it so that the label has only 1 cc-6 from the maximum
        label_data = single_component(label_data, starting_point)

        print(f"sum after: {np.sum(label_data)}")

        affine = nii_image.affine
        # Create a new NIfTI image using the existing image's affine matrix and header
        new_nifti_img = nib.Nifti1Image(label_data, affine, header=header)

        # Save the new NIfTI image to a file
        # nib.save(new_nifti_img, 'Z:/Zach_Analysis/petlymph_image_data/labelsv2/' + str(petlymph) + '_label_' + str(petlymph_dic[petlymph])+ '.nii.gz')
        # nib.save(new_nifti_img, 'Z:/Zach_Analysis/petlymph_image_data/labels_v3_nifti/' + str(petlymph) + '_label_' + str(petlymph_dic[petlymph])+ '.nii.gz')
        #nib.save(new_nifti_img, 'Z:/Zach_Analysis/petlymph_image_data/labels_v6_nifti' + row["Label_Name"] + '.nii.gz')
        nib.save(new_nifti_img, '/UserData/Zach_Analysis/petlymph_image_data/' + save_location +"/"+ row["Label_Name"] + '.nii.gz')
    print(f"missing petlymph number: {missing_conversion}")
    print(f"exit early: {exit_early}")

    df = df[
        ~df.apply(lambda row: any(drop_word.lower() in str(cell).lower() for cell in row for drop_word in drop_later),
                  axis=1)]
    print(f"new_length of df: {len(df)}")
    df.rename(columns={'Extracted Sentences': 'sentence'}, inplace=True)
    return df