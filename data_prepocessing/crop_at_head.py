import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pydicom



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

def get_radiotracer_info(path_to_dicom):
    # Get a list of all files in the folder
    dicom_files = [f for f in os.listdir(path_to_dicom) if f.endswith('.dcm')]

    # Check if there are any DICOM files in the folder
    if not dicom_files:
        return None

    # Load the first DICOM file
    first_dicom_path = os.path.join(path_to_dicom, dicom_files[0])
    dicom_data = pydicom.dcmread(first_dicom_path)

    # Check if the "Radiopharmaceutical Information Sequence" is present
    if 'RadiopharmaceuticalInformationSequence' in dicom_data:
        radiopharm_info = dicom_data.RadiopharmaceuticalInformationSequence[0]
        if 'Radiopharmaceutical' in radiopharm_info:
            radiotracer = radiopharm_info.Radiopharmaceutical
            #print(f"Radiotracer used: {radiotracer}")
            return radiotracer
        else:
            #print("Radiopharmaceutical information does not contain the radiotracer.")
            return None
    else:
        #print("No Radiopharmaceutical Information Sequence found in this DICOM file.")
        return None

def crop_at_head_calculation(df):

    base_folder = "/mnt/Bradshaw/UW_PET_Data/SUV_images/"
    folder_list = os.listdir(base_folder)
    save_base = "/UserData/Zach_Analysis/petlymph_image_data/prediction_mips_for_presentations/crop_at_head_v1/"
    master_dicom_location = "/UserData/UW_PET_Data/full_accounting_of_pet_ct_found_merged.xlsx"
    dicom_locations = pd.read_excel(master_dicom_location)
    images_created = 0
    cropped_frames = 0
    fdg_images = 0
    non_fdg_images = 0
    #for folder in folder_list:
    modality_used = ""

    id = []
    crop_list = []
    crop_type = []
    pet_path_list = []
    ct_path_list = []

    for index, row in df.iterrows():
        print(f"index: {index} total frames cropped: {cropped_frames} fdg images: {fdg_images} non fdg images: {non_fdg_images}")
        #index += 1
        #if index < 1000:
        #    continue
        #if index > 1100 and index < 7000:
        #    continue
        #if index < 10000:
        #    continue
        #if index > 11000:
        #    break

        #if images_created == 500:
        #    break
        folder = row["Petlymph"]
        #print(f"folder: {folder}")
        current_path = os.path.join(base_folder, folder)

        suv_dicom = dicom_locations[dicom_locations['Patient_Coding'] == folder]["PT_Path"].iloc[0]
        #print(suv_dicom)
        #print(type(suv_dicom))
        #print(suv_dicom.iloc[0])
        radiotracer = get_radiotracer_info(suv_dicom)
        #print(radiotracer)
        #if True:
        #    break

        ct_path_final = None
        suv_path_final = None

        for file_name in os.listdir(current_path):
            if "CT" in file_name:
                ct_path_final = os.path.join(current_path, file_name)
            if "SUV" in file_name:
                suv_path_final = os.path.join(current_path, file_name)

        if ct_path_final is None or suv_path_final is None:
            continue

        # Load the NIfTI files
        ct_nifti_image = nib.load(ct_path_final)
        suv_nifti_image = nib.load(suv_path_final)

        # Get the data arrays from the NIfTI images
        ct_data = ct_nifti_image.get_fdata()
        suv_data = suv_nifti_image.get_fdata()

        if "FDG" in radiotracer:
            fdg_images += 1
            #suv_nifti_image = nib.load(suv_path_final)
            #suv_data = suv_nifti_image.get_fdata()
            z_plane = find_z_plane_above_threshold(5, suv_data)
            crop_offset = suv_data.shape[2] - z_plane
            print(f"offset from pet: {crop_offset}")
            modality_used = "pet"
        else:
            non_fdg_images += 1
            #ct_nifti_image = nib.load(ct_path_final)
            #ct_data = ct_nifti_image.get_fdata()
            z_plane = find_z_plane_above_threshold(1000, ct_data)
            crop_offset = ct_data.shape[2] - z_plane
            modality_used = "ct"

        id.append(row["Petlymph"])
        crop_list.append(crop_offset)
        crop_type.append(modality_used)
        pet_path_list.append(suv_path_final)
        ct_path_list.append(ct_path_final)

        save_destination = os.path.join(save_base, str(folder) + ".png")
        print(save_destination)

        # Load the NIfTI files
        #ct_nifti_image = nib.load(ct_path_final)
        #suv_nifti_image = nib.load(suv_path_final)

        # Get the data arrays from the NIfTI images
        #ct_data = ct_nifti_image.get_fdata()
        #suv_data = suv_nifti_image.get_fdata()


        #z_plane = find_z_plane_above_threshold(1000, ct_data)
        #crop_offset = ct_data.shape[2] - z_plane
        #cropped_frames += crop_offset
        # Check if a z-plane was found

        #if z_plane is not None:
        make_plot = False
        if make_plot:
            # Create a 2D maximum intensity projection along axis 1 for CT
            ct_max_projection_2d = np.max(ct_data, axis=1)

            # Create a 2D maximum intensity projection along axis 1 for SUV
            suv_max_projection_2d = np.max(suv_data, axis=1)

            # Create a 2D maximum intensity projection along axis 0 for CT
            ct_max_projection_2d_axis0 = np.max(ct_data, axis=0)

            # Plotting the 2D projections side by side
            fig, axes = plt.subplots(1, 3, figsize=(30, 10))

            # Plot the CT projection
            axes[0].imshow(ct_max_projection_2d.T, cmap='gray', origin='lower', vmax=1000, vmin=-1000)
            axes[0].set_title(f'CT Maximum Intensity Projection (Axis 1) used: {modality_used}')
            axes[0].set_xlabel('X-axis')
            axes[0].set_ylabel('Z-axis')
            axes[0].axhline(y=z_plane, color='r', linestyle='--', label=f'z-plane {z_plane}')
            midpoint_x = ct_data.shape[0] // 2
            axes[0].axvline(x=midpoint_x, color='b', linestyle='--', label=f'x-midpoint {midpoint_x}')
            axes[0].legend()

            # Plot the SUV projection
            axes[1].imshow(suv_max_projection_2d.T, cmap='gray_r', origin='lower', vmax=10, vmin=0)
            axes[1].set_title(f'SUV Maximum Intensity Projection (Axis 1) offset: {crop_offset}')
            axes[1].set_xlabel('X-axis')
            axes[1].set_ylabel('Z-axis')
            axes[1].axhline(y=z_plane, color='r', linestyle='--', label=f'z-plane {z_plane}')

            # Plot the CT projection on axis 0
            axes[2].imshow(ct_max_projection_2d_axis0.T, cmap='gray', origin='lower') #, vmax=10, vmin=0)
            axes[2].set_title(f'CT Maximum Intensity Projection (Axis 0) max: {np.max(ct_max_projection_2d_axis0)}')
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

    df = pd.DataFrame({
        'id': id,
        'crop_offset': crop_list,
        'croped_type': crop_type,
        'pet_path': pet_path_list,
        'ct_path': ct_path_list
    })

    df.to_excel('/UserData/Zach_Analysis/suv_slice_text/uw_all_pet_preprocess_chain_v4/crop_offset_lookup.xlsx', index=False)