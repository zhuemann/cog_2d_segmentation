import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def find_z_plane_above_threshold_v1(threshold, data):
    midpoint_x = data.shape[0] // 2
    midpoint_y = data.shape[1] // 2

    for z in reversed(range(data.shape[2])):
        if data[midpoint_x, midpoint_y, z] > threshold:
            return z
    return None


def find_z_plane_above_threshold(threshold, data):
    midpoint_x = data.shape[0] // 2
    midpoint_y = data.shape[1] // 2

    # Define the region around the midline
    x_start = max(midpoint_x - 50, 0)
    x_end = min(midpoint_x + 50, data.shape[0])
    y_start = max(midpoint_y - 50, 0)
    y_end = min(midpoint_y + 50, data.shape[1])

    for z in reversed(range(data.shape[2])):
        # Calculate the mean value in the region for the current z-plane
        region_max = np.max(data[x_start:x_end, y_start:y_end, z])
        if region_max > threshold:
            return z
    return None


def plot_ct_head_projections():

    base_folder = "/mnt/Bradshaw/UW_PET_Data/SUV_images/"

    folder_list = os.listdir(base_folder)
    save_base = "/UserData/Zach_Analysis/petlymph_image_data/prediction_mips_for_presentations/ct_find_head_mips/"
    index = 0
    images_created = 0
    for folder in folder_list:
        print(f"index: {index}")
        index += 1
        if images_created == 100:
            break
        current_path = os.path.join(base_folder, folder)

        for file_name in os.listdir(current_path):
            #print(file_name)
            if "CT" in file_name:
                current_path = os.path.join(current_path, file_name)

        if "CT" not in current_path:
            continue


        save_destination = save_base + str(folder) + ".png"

        #print(current_path)

        # Load the NIfTI file
        nifti_image = nib.load(current_path)

        # Get the data array from the NIfTI image
        data = nifti_image.get_fdata()

        z_plane = find_z_plane_above_threshold(-250, data)

        # Check if a z-plane was found
        if z_plane is not None:
            #print(f'The z-plane along the midpoint line with a value above 1000 is: {z_plane}')

            # Create a 2D maximum intensity projection along axis 1
            max_projection_2d = np.max(data, axis=1)

            # Plotting the 2D projection with the line
            plt.figure(figsize=(10, 10))
            plt.imshow(max_projection_2d.T, cmap='jet', origin='lower', vmax=500,
                       vmin=-1000)  # Transposed for correct orientation
            plt.colorbar(label='Max intensity')
            plt.title('2D Maximum Intensity Projection (Axis 1)')
            plt.xlabel('X-axis')
            plt.ylabel('Z-axis')

            # Calculate the midpoint of the first dimension
            midpoint_x = data.shape[0] // 2

            # Plot the line at the calculated midpoint and z-plane
            plt.axhline(y=z_plane, color='r', linestyle='--', label=f'z-plane {z_plane}')
            plt.axvline(x=midpoint_x, color='b', linestyle='--', label=f'x-midpoint {midpoint_x}')
            plt.legend()

            # Save the figure to the save destination
            #save_path = os.path.join(save_destination, f'{file_name}_projection.png')
            plt.savefig(save_destination)
            plt.close()
            images_created += 1
        else:
            print(f'No z-plane with a value above 1000 found along the midpoint line for file: {file_name}')