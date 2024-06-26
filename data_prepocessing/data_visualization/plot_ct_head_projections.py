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
    y_start = max(midpoint_y - 150, 0)
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
    save_base = "/UserData/Zach_Analysis/petlymph_image_data/prediction_mips_for_presentations/ct_find_head_mips_v5/"
    index = 0
    images_created = 0
    cropped_frames = 0
    for folder in folder_list:
        print(f"index: {index} total frames cropped: {cropped_frames}")
        index += 1
        #if index < 1000:
        #    continue
        #if index > 1100 and index < 7000:
        #    continue

        #if images_created == 500:
        #    break
        current_path = os.path.join(base_folder, folder)

        ct_path_final = None
        suv_path_final = None

        for file_name in os.listdir(current_path):
            if "CT" in file_name:
                ct_path_final = os.path.join(current_path, file_name)
            if "SUV" in file_name:
                suv_path_final = os.path.join(current_path, file_name)

        if ct_path_final is None or suv_path_final is None:
            continue

        save_destination = os.path.join(save_base, str(folder) + ".png")
        print(save_destination)
        # Load the NIfTI files
        ct_nifti_image = nib.load(ct_path_final)
        suv_nifti_image = nib.load(suv_path_final)

        # Get the data arrays from the NIfTI images
        ct_data = ct_nifti_image.get_fdata()
        suv_data = suv_nifti_image.get_fdata()

        z_plane = find_z_plane_above_threshold(1000, ct_data)
        crop_offset = ct_data.shape[2] - z_plane
        cropped_frames += crop_offset
        # Check if a z-plane was found
        if z_plane is not None:
            # Create a 2D maximum intensity projection along axis 1 for CT
            ct_max_projection_2d = np.max(ct_data, axis=1)

            # Create a 2D maximum intensity projection along axis 1 for SUV
            suv_max_projection_2d = np.max(suv_data, axis=1)

            # Create a 2D maximum intensity projection along axis 0 for CT
            ct_max_projection_2d_axis0 = np.max(ct_data, axis=0)

            # Plotting the 2D projections side by side
            fig, axes = plt.subplots(1, 3, figsize=(30, 10))

            # Plot the CT projection
            axes[0].imshow(ct_max_projection_2d.T, cmap='jet', origin='lower', vmax=3000, vmin=-1000)
            axes[0].set_title('CT Maximum Intensity Projection (Axis 1)')
            axes[0].set_xlabel('X-axis')
            axes[0].set_ylabel('Z-axis')
            axes[0].axhline(y=z_plane, color='r', linestyle='--', label=f'z-plane {z_plane}')
            midpoint_x = ct_data.shape[0] // 2
            axes[0].axvline(x=midpoint_x, color='b', linestyle='--', label=f'x-midpoint {midpoint_x}')
            axes[0].legend()

            # Plot the SUV projection
            axes[1].imshow(suv_max_projection_2d.T, cmap='jet', origin='lower', vmax=10, vmin=0)
            axes[1].set_title(f'SUV Maximum Intensity Projection (Axis 1) offset: {crop_offset}')
            axes[1].set_xlabel('X-axis')
            axes[1].set_ylabel('Z-axis')
            axes[1].axhline(y=z_plane, color='r', linestyle='--', label=f'z-plane {z_plane}')

            # Plot the CT projection on axis 0
            axes[2].imshow(ct_max_projection_2d_axis0.T, cmap='jet', origin='lower', vmax=3000, vmin=-1000)
            axes[2].set_title('CT Maximum Intensity Projection (Axis 0)')
            axes[2].set_xlabel('Y-axis')
            axes[2].set_ylabel('Z-axis')
            axes[2].axhline(y=z_plane, color='r', linestyle='--', label=f'z-plane {z_plane}')
            midpoint_y = ct_data.shape[1] // 2
            axes[2].axvline(x=midpoint_y, color='b', linestyle='--', label=f'y-midpoint {midpoint_y}')
            axes[2].legend()

            # Save the combined figure to the save destination
            plt.savefig(save_destination)
            plt.close()
            images_created += 1
        else:
            print(f'No z-plane with a value above 1000 found along the midpoint line for folder: {folder}')

    print(f"total cropped frames: {cropped_frames}")
