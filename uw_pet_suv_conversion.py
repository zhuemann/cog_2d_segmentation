
import os
import pandas as pd
import dicom2nifti
import nibabel as nib
from datetime import datetime
import numpy as np
import glob
import pydicom

from nilearn.image import resample_img

class missing_injection_time(Exception):
    pass

def get_suv_conversion_factor(test_dicom, weight=0):
    '''
        Get patients' weights (in kg, to g) / date difference (to second) / Half-life of a F-18 between the injection and acquisition
        SUV = tracer uptake (or activity concentration) in ROI (Bq/ml) / (injected dose (Bq) * 2^(-T/tau)) / patient weight (g))
        water density = 1 g/ml
        T: delay between the injection time and the scan time, tau: half-life of the radionuclides
    '''
    # print(test_dicom)
    dicom_corrections = test_dicom['00280051'].value
    if 'ATTN' not in dicom_corrections and 'attn' not in dicom_corrections:
        print('Not attenuation corrected -- SUV factor set to 1')
        return 1, 0

    dicom_manufacturer = test_dicom['00080070'].value.lower()
    if weight != 0:
        try:
            dicom_weight = test_dicom['00101030'].value
            if dicom_weight == 0 or dicom_weight == None:
                dicom_weight = weight
        except:
            dicom_weight = weight
    else:
        try:
            dicom_weight = test_dicom['00101030'].value
            if dicom_weight == 0 or dicom_weight == None:
                print('No weight info -- SUV factor set to 1')
                return 1, 0  # no weight info
        except:
            print('No weight info -- SUV factor set to 1')
            return 1, 0  # no weight info

    # scantime info
    if dicom_manufacturer[0:2] == 'ge' and '0009100D' in test_dicom:
        dicom_scan_datetime = test_dicom['0009100D'].value[0:14]  # need to check!
    else:
        dicom_scan_datetime = test_dicom['00080021'].value + test_dicom['00080031'].value
        # Series Date (8) + Series Time (6) = Series DateTime (14)
        dicom_scan_datetime = dicom_scan_datetime[:14]
    # find non-number characters
    non_decimal = [char for char in dicom_scan_datetime if char.isdigit()]
    dicom_scan_datetime = ''.join(non_decimal)
    # radiopharmaceutical info
    radiopharm_object = test_dicom['00540016'][0]
    # print(radiopharm_object)
    if '00181074' in radiopharm_object and '00181075' in radiopharm_object:
        dicom_half_life = radiopharm_object['00181075'].value  # Radionuclide Half Life
        dicom_dose = radiopharm_object['00181074'].value  # Radionuclide Total Dose
        if '00181078' in radiopharm_object:  # Radiopharmaceutical Start DateTime
            #print(f"radiopharm_object: {radiopharm_object}")
            #if radiopharm_object['00181078'].value != None:
            if radiopharm_object['00181078'].value != "":
                dicom_inj_datetime = radiopharm_object['00181078'].value[:14]  # Radiopharmaceutical Start DateTime
                #print(f"dicom after declared in 00181078: {dicom_inj_datetime}", flush=True)
            else:
                dicom_inj_datetime = dicom_scan_datetime[0:8] + radiopharm_object['00181072'].value
                #print(f"dicom after declared: {dicom_inj_datetime}", flush=True)
        else:
            dicom_inj_datetime = dicom_scan_datetime[0:8] + radiopharm_object['00181072'].value
        #print(f"before string conversion: {dicom_inj_datetime}", flush=True)
        # convert dicom_inj_datetime to string
        dicom_inj_datetime = str(dicom_inj_datetime)
        #print(f"dicom inject time end: {dicom_inj_datetime}", flush=True)
        non_decimal = [char for char in dicom_inj_datetime if char.isdigit()]
        dicom_inj_datetime = ''.join(non_decimal)
    # sometimes tracer info is wiped, and if GE, can be found in private tags
    else:
        print('No dose information -- SUV factor set to 1')
        return 1, 0

    #print(dicom_inj_datetime, flush=True)
    dicom_inj_datetime = dicom_inj_datetime[:14]  # year(4)/month(2)/day(2)/hour(2)/minute(2)/second(2)
    dicom_scan_datetime = dicom_scan_datetime[:14]  # year(4)/month(2)/day(2)/hour(2)/minute(2)/second(2)

    #print(f"dicom scan datetime: {dicom_scan_datetime}")
    #print(f"dicom_inj_datetime: {dicom_inj_datetime}")
    #print(f"type: {type(dicom_inj_datetime)}")
    if dicom_inj_datetime == "":
        raise missing_injection_time(f"no injection time")

    # date difference
    scan_datetime = datetime.strptime(dicom_scan_datetime, '%Y%m%d%H%M%S')
    inj_datetime = datetime.strptime(dicom_inj_datetime, '%Y%m%d%H%M%S')
    diff_seconds = (scan_datetime - inj_datetime).total_seconds()
    while diff_seconds < 0:  # scan time > injection time
        diff_seconds += 24 * 3600
        # SUV factor
    # print(dicom_dose, dicom_half_life, dicom_weight, diff_seconds, '\n')
    dose_corrected = dicom_dose * 2 ** (- diff_seconds / dicom_half_life)

    # Units = BQML
    suv_factor = 1 / (
                (dose_corrected / dicom_weight) * 0.001)  # 1/(dose_corrected (decay-corrected Bq) /dicom_weight (gram))
    if test_dicom.Units == "BQML":
        return suv_factor, dicom_weight
    else:
        return 1, dicom_weight

    # if test_dicom.Units == "CNTS":
    # suv_factor /= test_dicom['00280030'].value[0] * test_dicom['00280030'].value[1] * float(test_dicom['00180050'].value) * 0.001
    # raise ValueError('Unknown units: %s' % test_dicom.Units)

def convert_pet_nifti_to_suv_nifti(nifti_read_filename, test_dicom, nifti_save_filename, weight=0, norm_factor=1):
    suv_factor, dicom_weight = get_suv_conversion_factor(test_dicom, weight)
    if suv_factor != 1:
        orig = nib.load(nifti_read_filename)
        data = orig.get_fdata()
        new_data = data.copy()
        new_data = new_data * suv_factor
        if np.max(new_data) < 5:
            print('PET image values seem low. Check SUV conversion')
            return False, dicom_weight
        suv_img = nib.Nifti1Image(new_data, orig.affine, orig.header)
        nib.save(suv_img, nifti_save_filename)
        return True, dicom_weight
    elif norm_factor != 1:
        orig = nib.load(nifti_read_filename)
        data = orig.get_fdata()
        new_data = data.copy()
        new_data = new_data * norm_factor
        if np.max(new_data) < 5:
            print('PET image values seem low. Check SUV conversion')
            return False, dicom_weight
        suv_img = nib.Nifti1Image(new_data, orig.affine, orig.header)
        nib.save(suv_img, nifti_save_filename)
        return True, dicom_weight
    else:
        print("returned false in convert_pet_nifti_to_suv_nifti")
        return False, dicom_weight


def convert_PT_CT_files_to_nifti(top_dicom_folder, top_nifti_folder):
    # modality of interest is the modality that will be the reference size for the RTSTRUCT contours, defined by DICOM
    # type ('PT, 'CT', 'MR')
    files = glob.glob(top_dicom_folder + "/*.dcm")
    if len(files) < 1:
        print('Empty folder: ' + files)
        raise Exception("Fail to find DICOM files")

    # get dicom info for saving
    test_dicom = pydicom.dcmread(files[0])
    dicom_modality = test_dicom['00080060'].value
    dicom_name = str(test_dicom['00100010'].value).lower()
    dicom_id = test_dicom['00100020'].value.lower()
    dicom_study_date = test_dicom['00080020'].value
    dicom_series_description = test_dicom['0008103e'].value
    #print(f"top dicom folder: {top_dicom_folder}")
    folder_names = top_dicom_folder.split("/")
    indices_of_pet = [index for index, element in enumerate(folder_names) if "petwb_" in element.lower()]
    #print(f"indices: {indices_of_pet}")
    #print(f"test: {folder_names[indices_of_pet[0]]}")
    # unique names for subjects and scans
    #subject_save_name = dicom_id + '_' + dicom_name.replace(' ', '_').replace('__', '_')
    #print(f"subject_save_name: {subject_save_name}")
    subject_save_name = folder_names[indices_of_pet[0]]
    #print(f"subject_save_name: {subject_save_name}")

    subject_save_folder = os.path.join(top_nifti_folder, subject_save_name)
    os.makedirs(subject_save_folder, exist_ok=True)
    scan_save_name = '{}_{}_{}_{}'.format(subject_save_name, dicom_study_date, dicom_modality, \
                                          dicom_series_description.replace(' ', '_'))

    if dicom_modality in ['CT', 'MR', 'NM']:
        #print(f"dicom_modality: {dicom_modality}")
        dicom2nifti.dicom_series_to_nifti(top_dicom_folder,
                                          os.path.join(subject_save_folder, scan_save_name + '.nii.gz'),
                                          reorient_nifti=False)
    elif dicom_modality == 'PT':
        #print(f"dicom_modality: {dicom_modality}")
        #print(f"about to call dicom to nifiti")
        try:
            dicom2nifti.dicom_series_to_nifti(top_dicom_folder,
                                          os.path.join(subject_save_folder, scan_save_name + '.nii.gz'),
                                          reorient_nifti=False)
        #print(f" about to call convert to nifiti to suv nifiti")
            convert_pet_nifti_to_suv_nifti(os.path.join(subject_save_folder, scan_save_name + '.nii.gz'), test_dicom,
                                       os.path.join(subject_save_folder, scan_save_name + '_SUV.nii.gz'))
        except ValueError:
            print("slice error")
            os.rmdir(subject_save_folder)
        except Exception as e:
            print(e)
            os.rmdir(subject_save_folder)






def files_transfer_analysis():
    dir_path = "/mnt/Bradshaw/UW_PET_Data/dsb2b/"
    files_in_directory = os.listdir(dir_path)
    print(files_in_directory)

    df_path = "/UserData/UW_PET_Data/UWPETCTWB_Research-Image-Requests_20240311.xlsx"
    df = pd.read_excel(df_path)

    # Filter the DataFrame to include only rows where the filename is not in files_in_directory
    filtered_df = df[~df['Coded Accession Number'].isin(files_in_directory)]

    # Save the filtered DataFrame to an Excel file
    output_file_path = '/UserData/UW_PET_Data/missing_accession_numbers_v2.xlsx'
    filtered_df.to_excel(output_file_path, index=False)


def file_exploration_analysis_pet():
    dir_path = "/mnt/Bradshaw/UW_PET_Data/dsb2b/"
    dir_path = "/mnt/Bradshaw/UW_PET_Data/2024-07/"


    files_in_directory = os.listdir(dir_path)
    print(f"files in folder: {len(files_in_directory)}")
    no_pt_files_list = []
    index = 0

    missing_inject_info = 0
    potential_suv_images = 0

    num_dates = {} # key is number of dates in folder value is how many folders have that value
    num_dates[1] = 0
    num_modality = {"PT": 0, "CT": 0, "extra": 0}
    num_study_names = {1:0 , "extra": 0, 0: 0}
    types_of_scans_ct = {}
    types_of_scans_pt = {}
    types_of_scans_pt["12__WB_MAC"] = 0
    types_of_scans_pt["12__WB_3d_MAC"] = 0
    types_of_scans_pt["5__WB_MAC"] = 0
    types_of_scans_pt["4__WB_MAC"] = 0
    types_of_scans_pt["wb_ac_3d"] = 0
    types_of_scans_pt["4__PET_AC_3D"] = 0
    types_of_scans_pt["13__WB_3D_MAC"] = 0
    types_of_scans_pt["13__WB_MAC"] = 0
    types_of_scans_pt["12__PET_AC_3D"] = 0

    for file in files_in_directory:
        #print(f"index: {index} missing inject info: {missing_inject_info} potential found: {potential_suv_images}")
        #print(f"index: index")
        index += 1
        #if index > 100:
        #    break

        directory = os.path.join(dir_path, file)
        print(directory)
        date = os.listdir(directory)
        if len(date) == 1:
            directory = os.path.join(directory, date[0])
            num_dates[1] += 1
        else:
            print(f"multiple date files in this folder: {directory}")
            if len(date) not in num_dates:
                num_dates[len(date)] = 1
            else:
                num_dates[len(date)] += 1

        modality = os.listdir(directory)
        print(f"modality: {modality}")
        if len(modality) > 2:
            num_modality["extra"] += 1
        if "CT" in modality:
            num_modality["CT"] += 1
        #else:
            #print(f"file: {file} does not have ct scan modality: {modality}")
        #    continue

        if "PT" in modality:
            # directory = os.path.join(dir_path, file, "PT")
            directory = os.path.join(directory, "PT")
            num_modality["PT"] += 1
        else:
            #print(f"file: {file} does not have Pet scan modality: {modality}")
            continue

        """
        if "CT" in modality:
            # directory = os.path.join(dir_path, file, "PT")
            directory = os.path.join(directory, "CT")
            num_modality["CT"] += 1
        else:
            #print(f"file: {file} does not have Pet scan modality: {modality}")
            continue
        """

        # print(directory)
        study_name = os.listdir(directory)
        print(study_name)
        if len(study_name) == 0:
            print(f"something funny: {file}")
            no_pt_files_list.append(file)
            num_study_names[0] += 1
            #continue
        elif study_name == 1:
            num_study_names[1] += 1
        else:
            num_study_names["extra"] += 1


        directory = os.path.join(directory, study_name[0])
        recon_types = os.listdir(directory)
        print(f"recon types: {recon_types")
        if any("12__wb_3d_mac" in element.lower() for element in recon_types):
            types_of_scans_pt["12__WB_3d_MAC"] += 1
        elif any("wb_ac_3d" in element.lower() for element in recon_types):
            types_of_scans_pt["wb_ac_3d"] += 1
        elif any("12__WB_MAC" == element for element in recon_types):
            types_of_scans_pt["12__WB_MAC"] += 1
        elif any("5__WB_MAC" == element for element in recon_types):
            types_of_scans_pt["5__WB_MAC"] += 1
        elif any("4__WB_MAC" == element for element in recon_types):
            types_of_scans_pt["4__WB_MAC"] += 1
        elif any("4__PET_AC_3D" == element for element in recon_types):
            types_of_scans_pt["4__PET_AC_3D"] += 1
        elif any("13__WB_3D_MAC" == element for element in recon_types):
            types_of_scans_pt["13__WB_3D_MAC"] += 1
        elif any("13__WB_MAC" == element for element in recon_types):
            types_of_scans_pt["13__WB_MAC"] += 1
        elif any("12__PET_AC_3D" == element for element in recon_types):
            types_of_scans_pt["12__PET_AC_3D"] += 1

        else:
            for recon in recon_types:
                if recon in types_of_scans_pt:
                    types_of_scans_pt[recon] += 1
                else:
                    types_of_scans_pt[recon] = 1


    print(f"number of dates in files: {num_dates}")
    print(f"number of modality in date file: {num_modality}")
    #print(f"types of scans: {types_of_scans_pt}")
    #sum = 0
    #for key, value in types_of_scans_pt.items():
        #sum += value
    #    if value > 20:
    #        print(f"{key} {value}")
    for key, value in sorted(types_of_scans_pt.items(), key=lambda item: item[1], reverse=True):
        if value > 20:
            print(f"{key} {value}")
    #print(f"total images we will have: {sum}")

def resample_nii_to_3mm(nii_image, output_file_path):
    # Load the NIfTI file
    #nii = nib.load(input_file_path)

    new_voxel_size = [3, 3, 3]
    # Get the current affine
    affine = nii_image.affine

    # Create a new affine matrix for the desired voxel size
    new_affine = affine.copy()
    for i in range(3):
        new_affine[i, i] = new_voxel_size[i] * affine[i, i] / nii_image.header.get_zooms()[i]

    # Define the new affine matrix for 3mm x 3mm x 3mm voxels
    target_affine = np.eye(4) * 3
    target_affine[3, 3] = 1  # Set the affine translation parameters correctly

    # Resample the image
    resampled_img = resample_img(nii_image, target_affine=new_affine, interpolation='linear')
    return resampled_img
    # Save the resampled image
    #nib.save(resampled_img, output_file_path)


def calculate_new_dimensions(nii):
    # Load the NIfTI file
    #nii = nib.load(input_file_path)

    # Get current dimensions and voxel sizes
    current_shape = nii.header.get_data_shape()
    current_voxel_dims = nii.header.get_zooms()

    # New voxel size
    new_voxel_size = (3.65, 3.65, 3.65)

    # Calculate new dimensions based on the ratio of old voxel size to new voxel size
    new_dimensions = np.ceil(np.array(current_shape) * np.array(current_voxel_dims) / np.array(new_voxel_size)).astype(
        int)
    #print(new_dimensions)
    return tuple(new_dimensions)

def get_voxel_dimensions(root_directory):
    voxel_dims_count = {}
    image_size = {}
    i = -1
    # Loop through all subdirectories in the root directory
    for subdir in os.listdir(root_directory):
        subdir_path = os.path.join(root_directory, subdir)
        i += 1
        print(f"index:{i}")
        #if i > 100:
        #    break
        if os.path.isdir(subdir_path):
            # Loop through all files in the subdirectory
            for filename in os.listdir(subdir_path):
                if filename.endswith(".nii.gz") and "suv" in filename.lower():
                    filepath = os.path.join(subdir_path, filename)
                    try:
                        # Load the NIfTI file
                        nii = nib.load(filepath)
                        # Extract voxel dimensions
                        voxel_dims = tuple(nii.header.get_zooms()[:3])  # Get only the first three dimensions
                        #resampled_img = resample_nii_to_3mm(nii, "")
                        new_dim = calculate_new_dimensions(nii)
                        # Count the voxel dimensions
                        if voxel_dims in voxel_dims_count:
                            voxel_dims_count[voxel_dims] += 1
                        else:
                            voxel_dims_count[voxel_dims] = 1

                        if new_dim in image_size:
                            image_size[new_dim] += 1
                        else:
                            image_size[new_dim] = 1
                    except Exception as e:
                        print(f"Error processing {filename} in {subdir}: {e}")
    print(f"dictionary: {voxel_dims_count}")
    for key, value in sorted(voxel_dims_count.items(), key=lambda item: item[1], reverse=True):
        print(f"{key} {value}")
    print("image size")
    sum_short = 0
    sum_long = 0
    very_short = 0
    for key, value in sorted(image_size.items(), key=lambda item: item[1], reverse=True):
        print(f"{key} {value}")
        if key[2] <= 384:
            sum_short += value
        else:
            sum_long += value
        if key[2] <= 200:
            very_short += value
    print(f"number of images under 384 in length: {sum_short}")
    print(f"number that need cropping: {sum_long}")
    print(f"under 200, likely remove from dataset: {very_short}")
    return voxel_dims_count

def call_suv_helper(top_dicom_folder, top_nifti_folder, found_cts):
    try:
        convert_PT_CT_files_to_nifti(top_dicom_folder, top_nifti_folder)
    except Exception:
        found_cts -= 1
    found_cts += 1
    return found_cts


def get_dicom_dimensions(folder_path):
    # Iterate over files in the given folder
    for filename in os.listdir(folder_path):
        # Construct the full file path
        file_path = os.path.join(folder_path, filename)

        # Check if the file is a DICOM file
        if pydicom.misc.is_dicom(file_path):
            # Read the DICOM file
            dicom_file = pydicom.dcmread(file_path)

            # Retrieve dimensions
            dimensions = (dicom_file.Rows, dicom_file.Columns)
            return dimensions

    return None


def file_conversion_ct():
    dir_path = "/mnt/Bradshaw/UW_PET_Data/dsb2b/"
    dir_path_suv = "/mnt/Bradshaw/UW_PET_Data/SUV_images/"
    top_nifti_folder = "/mnt/Bradshaw/UW_PET_Data/SUV_images/"

    files_in_directory = os.listdir(dir_path_suv)
    print(f"files in folder: {len(files_in_directory)}")
    no_pt_files_list = []
    index = 0

    num_dates = {}  # key is number of dates in folder value is how many folders have that value
    num_dates[1] = 0
    num_modality = {"PT": 0, "CT": 0, "extra": 0}
    num_study_names = {1: 0, "extra": 0, 0: 0}

    found_cts = 0

    for file in files_in_directory:
        # print(f"index: {index} missing inject info: {missing_inject_info} potential found: {potential_suv_images}")
        # if index > 10:
        #    continue
        print(f"index: {index} filename: {file} found CTs: {found_cts}")
        # print(file)
        index += 1
        # if index > 100:
        #    break
        suv_dims = (0, 0, 0)
        suv_path = os.path.join(dir_path_suv, file)
        for filename in os.listdir(suv_path):
            if filename.endswith(".nii.gz") and "suv" in filename.lower():
                filepath = os.path.join(suv_path, filename)
                try:
                    # Load the NIfTI file
                    nii = nib.load(filepath)
                    suv_dims = nii.header.get_data_shape()
                except:

                    print("can't get dimensions from suv")

        # print(f"suv_dims: {suv_dims}")
        directory = os.path.join(dir_path, file)
        date = os.listdir(directory)
        if len(date) == 1:
            directory = os.path.join(directory, date[0])
            num_dates[1] += 1
        else:
            print(f"multiple date files in this folder: {directory}")
            if len(date) not in num_dates:
                num_dates[len(date)] = 1
            else:
                num_dates[len(date)] += 1

        modality = os.listdir(directory)

        if len(modality) > 2:
            num_modality["extra"] += 1
        if "CT" in modality:
            num_modality["CT"] += 1
        else:
            # print(f"file: {file} does not have ct scan modality: {modality}")
            continue
        if "CT" in modality:
            # directory = os.path.join(dir_path, file, "PT")
            directory = os.path.join(directory, "CT")
        else:
            print(f"file: {file} does not have Pet scan")
            continue

        # print(directory)
        study_name = os.listdir(directory)
        if len(study_name) == 0:
            print(f"something funny: {file}")
            no_pt_files_list.append(file)
            num_study_names[0] += 1
            # continue
        elif study_name == 1:
            num_study_names[1] += 1
        else:
            num_study_names["extra"] += 1

        directory = os.path.join(directory, study_name[0])
        recon_types = os.listdir(directory)


        substrings_to_check = ["CT_MAR", "CTAC", "WB_CT_SLICES", "CT_IMAGES", "WB_Standard"]
        # Iterate over each substring and check if it's present in any element of recon_types
        for substring in substrings_to_check:
            # Normalize to lower case for case-insensitive comparison
            matched_recon = next((recon for recon in recon_types if substring.lower() in recon.lower()), None)
            if matched_recon:
                # If a match is found, build the path
                top_dicom_folder = os.path.join(directory, matched_recon)
                z = len(os.listdir(top_dicom_folder))
                # checks if slices line up other wise don't convert and keep searching
                if z == suv_dims[2]:
                    # Perform your additional logic or function calls here
                    try:
                        found_cts = call_suv_helper(top_dicom_folder, top_nifti_folder, found_cts)
                    except:
                        continue  # If an error occurs, continue with the next substring





def file_exploration_analysis_ct():
    dir_path = "/mnt/Bradshaw/UW_PET_Data/dsb2b/"
    dir_path_suv = "/mnt/Bradshaw/UW_PET_Data/SUV_images/"
    top_nifti_folder = "/mnt/Bradshaw/UW_PET_Data/SUV_images/"

    files_in_directory = os.listdir(dir_path_suv)
    print(f"files in folder: {len(files_in_directory)}")
    no_pt_files_list = []
    index = 0

    missing_inject_info = 0
    potential_suv_images = 0

    num_dates = {}  # key is number of dates in folder value is how many folders have that value
    num_dates[1] = 0
    num_modality = {"PT": 0, "CT": 0, "extra": 0}
    num_study_names = {1: 0, "extra": 0, 0: 0}
    types_of_scans_ct = {}
    types_of_scans_pt = {}
    #types_of_scans_pt["12__WB_MAC"] = 0
    #types_of_scans_pt["12__WB_3d_MAC"] = 0
    #types_of_scans_pt["5__WB_MAC"] = 0
    #types_of_scans_pt["4__WB_MAC"] = 0
    #types_of_scans_pt["wb_ac_3d"] = 0
    #types_of_scans_pt["4__PET_AC_3D"] = 0
    #types_of_scans_pt["13__WB_3D_MAC"] = 0
    #types_of_scans_pt["13__WB_MAC"] = 0
    #types_of_scans_pt["12__PET_AC_3D"] = 0
    same_slice_nums = 0
    found_cts = 0
    multiple_recon_dic = {}

    matches_dic = {}
    for file in files_in_directory:
        # print(f"index: {index} missing inject info: {missing_inject_info} potential found: {potential_suv_images}")
        #if index > 10:
        #    continue
        print(f"index: {index} filename: {file}")
        #print(file)
        index += 1
        # if index > 100:
        #    break
        suv_dims = (0, 0, 0)
        suv_path = os.path.join(dir_path_suv, file)
        for filename in os.listdir(suv_path):
            if filename.endswith(".nii.gz") and "suv" in filename.lower():
                filepath = os.path.join(suv_path, filename)
                try:
                    # Load the NIfTI file
                    nii = nib.load(filepath)
                    suv_dims = nii.header.get_data_shape()
                except:

                    print("can't get dimensions from suv")

        #print(f"suv_dims: {suv_dims}")
        directory = os.path.join(dir_path, file)
        date = os.listdir(directory)
        if len(date) == 1:
            directory = os.path.join(directory, date[0])
            num_dates[1] += 1
        else:
            print(f"multiple date files in this folder: {directory}")
            if len(date) not in num_dates:
                num_dates[len(date)] = 1
            else:
                num_dates[len(date)] += 1

        modality = os.listdir(directory)

        if len(modality) > 2:
            num_modality["extra"] += 1
        if "CT" in modality:
            num_modality["CT"] += 1
        else:
            # print(f"file: {file} does not have ct scan modality: {modality}")
            continue
        if "CT" in modality:
            #directory = os.path.join(dir_path, file, "PT")
            directory = os.path.join(directory, "CT")
        else:
            print(f"file: {file} does not have Pet scan")
            continue

        # print(directory)
        study_name = os.listdir(directory)
        if len(study_name) == 0:
            print(f"something funny: {file}")
            no_pt_files_list.append(file)
            num_study_names[0] += 1
            # continue
        elif study_name == 1:
            num_study_names[1] += 1
        else:
            num_study_names["extra"] += 1

        directory = os.path.join(directory, study_name[0])
        recon_types = os.listdir(directory)

        """
        if any("2__ctac" in element.lower() for element in recon_types):
            
            top_dicom_folder = os.path.join(recon_types, "2__CTAC")
            try:
                found_cts = call_suv_helper(top_dicom_folder, top_nifti_folder, found_cts)
            except:
                continue
        """
        number_matches = 0
        recon_type_list = []
        substrings_to_check = ["CT_MAR", "CTAC", "WB_CT_SLICES", "CT_IMAGES", "WB_Standard"]
        # Iterate over each substring and check if it's present in any element of recon_types
        for substring in substrings_to_check:
            # Normalize to lower case for case-insensitive comparison
            matched_recon = next((recon for recon in recon_types if substring.lower() in recon.lower()), None)
            if matched_recon:
                # If a match is found, build the path
                top_dicom_folder = os.path.join(directory, matched_recon)
                z = len(os.listdir(top_dicom_folder))
                #checks if slices line up other wise don't convert and keep searching
                if z == suv_dims[2]:
                    # Perform your additional logic or function calls here
                    try:
                        found_cts = call_suv_helper(top_dicom_folder, top_nifti_folder, found_cts)
                    except:
                        continue  # If an error occurs, continue with the next substring

        if any("2__ctac" in element.lower() for element in recon_types):
            top_dicom_folder = os.path.join(directory, "2__CTAC")
            # if there is already a CT in the top_dicom_folder continue
            try:
                found_cts = call_suv_helper(top_dicom_folder, top_nifti_folder, found_cts)
            except:
                continue
        elif any("wb_ac_3d" in element.lower() for element in recon_types):
            types_of_scans_pt["wb_ac_3d"] += 1
        elif any("12__WB_MAC" == element for element in recon_types):
            types_of_scans_pt["12__WB_MAC"] += 1
        elif any("5__WB_MAC" == element for element in recon_types):
            types_of_scans_pt["5__WB_MAC"] += 1
        elif any("4__WB_MAC" == element for element in recon_types):
            types_of_scans_pt["4__WB_MAC"] += 1
        elif any("4__PET_AC_3D" == element for element in recon_types):
            types_of_scans_pt["4__PET_AC_3D"] += 1
        elif any("13__WB_3D_MAC" == element for element in recon_types):
            types_of_scans_pt["13__WB_3D_MAC"] += 1
        elif any("13__WB_MAC" == element for element in recon_types):
            types_of_scans_pt["13__WB_MAC"] += 1
        elif any("12__PET_AC_3D" == element for element in recon_types):
            types_of_scans_pt["12__PET_AC_3D"] += 1

        else:
            for recon in recon_types:
                #print(directory)
                #print(recon)

                #print(f"recon name: {recon}")
                top_dicom_folder = os.path.join(directory, recon)
                #print(top_dicom_folder)
                z = len(os.listdir(top_dicom_folder))
                x, y = get_dicom_dimensions(top_dicom_folder)
                #print(f"x of dicom: {x} slices in suv: {suv_dims[0]}")
                #print(f"y of dicom: {y} slices in suv: {suv_dims[1]}")
                #print(f"z of dicom: {z} slices in suv: {suv_dims[2]}")

                if z == suv_dims[2]: # and x == suv_dims[0] and y == suv_dims[1] and z == suv_dims[2]:
                    same_slice_nums += 1
                    number_matches += 1
                    recon_type_list.append(recon)
                    if recon in types_of_scans_pt:
                        types_of_scans_pt[recon] += 1
                    else:
                        types_of_scans_pt[recon] = 1
            if number_matches == 1:
                recon = recon_type_list[0]
                #recon = tuple(recon_type_list)
                if recon in multiple_recon_dic:
                    multiple_recon_dic[recon] += 1
                else:
                     multiple_recon_dic[recon] = 1

            if number_matches in matches_dic:
                matches_dic[number_matches] += 1
            else:
                matches_dic[number_matches] = 1


    print(f"number of dates in files: {num_dates}")
    print(f"number of modality in date file: {num_modality}")
    # print(f"types of scans: {types_of_scans_pt}")
    # sum = 0
    # for key, value in types_of_scans_pt.items():
    # sum += value
    #    if value > 20:
    #        print(f"{key} {value}")
    for key, value in sorted(types_of_scans_pt.items(), key=lambda item: item[1], reverse=True):
        #if value > 20:
        print(f"{key} {value}")
    # print(f"total images we will have: {sum}")
    print(f"has matching ct with same z: {same_slice_nums}")
    print(f"matches dic: {matches_dic}")
    for key, value in sorted(multiple_recon_dic.items(), key=lambda item: item[1], reverse=True):
        #if value > 20:
        print(f"{key} {value}")


def uw_pet_suv_conversion():

    #file_exploration_analysis_pet()
    #file_exploration_analysis_ct()
    #get_voxel_dimensions("/mnt/Bradshaw/UW_PET_Data/SUV_images/")
    #print(fail)
    #files_transfer_analysis()
    #print(fail)

    #top_dicom_folder = "/UserData/1043/PETLYMPH_3004/PT/20150125/BODY/1203__PET_CORONAL/"
    #top_nifti_folder = "/UserData/Zach_Analysis/suv_nifti_test/"
    #top_nifti_folder = "/UserData/UW_PET_Data/uw_pet_suv/"
    top_nifti_folder = "/mnt/Bradshaw/UW_PET_Data/SUV_images/"
    #convert_PT_CT_files_to_nifti(top_dicom_folder, top_nifti_folder)

    #dir_path = "/UserData/1043/"
    #dir_path = "/mnt/dsb2b/"
    dir_path = "/mnt/Bradshaw/UW_PET_Data/dsb2b/"

    files_in_directory = os.listdir(dir_path)
    #print(files_in_directory)

    no_pt_files_list = []
    index = 0
    found_pet_images = 0
    multi_length = 0
    missing_inject_info = 0
    potential_suv_images = 0
    skip_files = set([])
    no_pt_files = set(["PETWB_015788_01"])
    no_pt_files = set()
    time_data_skip = set([])
    dicom_error = set(["PETWB_005434_01", "PETWB_004604_04"]) #PETWB_004604_04
    dicom_error = set([])
    weird_path_names = []
    time_errors = []
    for file in files_in_directory:
        print(f"index: {index} found pet images: {found_pet_images} file: {file}")
        index += 1
        #if index > 100:
        #    break
        #if index < 4630:
        #    continue
        if os.path.exists(os.path.join(top_nifti_folder, file)):
            found_pet_images += 1
            print("already found this image")
            continue

        #if index < 1970:
        #    continue
        if file in skip_files or file in no_pt_files or file in time_data_skip or file in dicom_error:
            continue
        directory = os.path.join(dir_path, file)
        date = os.listdir(directory)
        if len(date) == 1:
            directory = os.path.join(directory, date[0])
        else:
            print(f"multiple date files in this folder: {directory}")
        modality = os.listdir(directory)
        #print(modality)
        if "PT" in modality:
            #directory = os.path.join(dir_path, file, "PT")
            directory = os.path.join(directory, "PT")
        else:
            print(f"file: {file} does not have Pet scan")
            continue
        #print(directory)
        ref_num = os.listdir(directory)
        if len(ref_num) == 0:
            print(f"something funny: {file}")
            no_pt_files_list.append(file)
            continue
        # print(ref_num)
        directory = os.path.join(directory, ref_num[0])
        # print(test_directory)
        type_exam = os.listdir(directory)
        # print(modality)
        # print(test)
        """
        if 'PET_CT_SKULL_BASE_TO_THIGH' in type_exam:
            folder_name = 'PET_CT_SKULL_BASE_TO_THIGH'
        elif len(type_exam) > 1:
            weird_path_names.append(file)
            multi_length += 1
            print(f"type of exam greater than 1:{type_exam}")
            continue
        else:
            folder_name = type_exam[0]
        """
        recon_types = os.listdir(directory)
        substrings_to_check = ["CT_MAR", "CTAC", "WB_CT_SLICES", "CT_IMAGES", "WB_Standard"]
        # Iterate over each substring and check if it's present in any element of recon_types
        for substring in substrings_to_check:
            # Normalize to lower case for case-insensitive comparison
            matched_recon = next((recon for recon in recon_types if substring.lower() in recon.lower()), None)
            if matched_recon:
                # If a match is found, build the path
                top_dicom_folder = os.path.join(directory, matched_recon)
                #z = len(os.listdir(top_dicom_folder))
                # checks if slices line up other wise don't convert and keep searching
                #if z == suv_dims[2]:
                    # Perform your additional logic or function calls here
                try:
                    found_cts = call_suv_helper(top_dicom_folder, top_nifti_folder, found_cts)
                except:
                    continue  # If an error occurs, continue with the next substring

        #print(f"directory before recon checks: {directory} with contents: {os.listdir(directory)}")
        test_directory = directory
        #test_directory = os.path.join(directory, folder_name)
        test = os.listdir(directory)
        #print(test)
        #print("before check")
        if any("12__wb_3d_mac" in element.lower() for element in test):
            potential_suv_images += 1
            top_dicom_folder = os.path.join(test_directory, "12__WB_3D_MAC")
            #print(f"top: {top_dicom_folder}")
            try:
                convert_PT_CT_files_to_nifti(top_dicom_folder, top_nifti_folder)
            except ValueError:
                print("failed")
                time_errors.append(file)
                continue
            except Exception:
                found_pet_images -= 1
                continue

            except missing_injection_time as e:
                print("missing inject time: {e}")
                missing_inject_info += 1
                continue
            found_pet_images += 1
            continue
        elif any("wb_ac_3d" in element.lower() for element in test):
            potential_suv_images += 1
            indices_of_pet = [index for index, element in enumerate(test) if "wb_ac_3d" in element.lower()]
            top_dicom_folder = os.path.join(test_directory, test[indices_of_pet[0]])
            #print(f"top: {top_dicom_folder}")
            try:
                convert_PT_CT_files_to_nifti(top_dicom_folder, top_nifti_folder)
            except ValueError:
                print("failed")
                time_errors.append(file)
                continue
            except Exception:
                found_pet_images -= 1
                continue
            except missing_injection_time as e:
                print("missing inject time: {e}")
                missing_inject_info += 1
                continue
            found_pet_images += 1
            continue
        elif any("12__WB_MAC" == element for element in test):
            potential_suv_images += 1
            top_dicom_folder = os.path.join(test_directory, "12__WB_MAC")
            try:
                convert_PT_CT_files_to_nifti(top_dicom_folder, top_nifti_folder)
            except ValueError:
                print("failed")
                time_errors.append(file)
                continue
            except Exception:
                found_pet_images -= 1
                continue
            except missing_injection_time as e:
                print("missing inject time: {e}")
                missing_inject_info += 1
                continue
            found_pet_images += 1

        elif any("5__WB_MAC" == element for element in test):
            found_pet_images += 1
            top_dicom_folder = os.path.join(test_directory, "5__WB_MAC")
            try:
                convert_PT_CT_files_to_nifti(top_dicom_folder, top_nifti_folder)
            except ValueError:
                print("failed")
                time_errors.append(file)
                continue
            except Exception:
                found_pet_images -= 1
                continue
        elif any("4__WB_MAC" == element for element in test):
            found_pet_images += 1
            top_dicom_folder = os.path.join(test_directory, "4__WB_MAC")
            try:
                convert_PT_CT_files_to_nifti(top_dicom_folder, top_nifti_folder)
            except ValueError:
                print("failed")
                time_errors.append(file)
                continue
            except Exception:
                found_pet_images -= 1
                continue
        elif any("4__PET_AC_3D" == element for element in test):
            found_pet_images += 1
            top_dicom_folder = os.path.join(test_directory, "4__PET_AC_3D")
            try:
                convert_PT_CT_files_to_nifti(top_dicom_folder, top_nifti_folder)
            except ValueError:
                print("failed")
                time_errors.append(file)
                continue
            except Exception:
                found_pet_images -= 1
                continue
        elif any("13__WB_3D_MAC" == element for element in test):
            found_pet_images += 1
            top_dicom_folder = os.path.join(test_directory, "13__WB_3D_MAC")
            try:
                convert_PT_CT_files_to_nifti(top_dicom_folder, top_nifti_folder)
            except ValueError:
                print("failed")
                time_errors.append(file)
                continue
            except Exception:
                found_pet_images -= 1
                continue
        elif any("13__WB_MAC" == element for element in test):
            found_pet_images += 1
            top_dicom_folder = os.path.join(test_directory, "13__WB_MAC")
            try:
                convert_PT_CT_files_to_nifti(top_dicom_folder, top_nifti_folder)
            except ValueError:
                print("failed")
                time_errors.append(file)
                continue
            except Exception:
                found_pet_images -= 1
                continue
        elif any("12__PET_AC_3D" == element for element in test):
            found_pet_images += 1
            top_dicom_folder = os.path.join(test_directory, "12__PET_AC_3D")
            try:
                convert_PT_CT_files_to_nifti(top_dicom_folder, top_nifti_folder)
            except ValueError:
                print("failed")
                time_errors.append(file)
                continue
            except Exception:
                found_pet_images -= 1
                continue



def uw_pet_suv_conversion_v2():


    top_nifti_folder = "/mnt/Bradshaw/UW_PET_Data/SUV_images/"

    #dir_path = "/mnt/Bradshaw/UW_PET_Data/dsb2b/"
    dir_path = "/mnt/Bradshaw/UW_PET_Data/dsb2b/"

    dir_path = "/mnt/Bradshaw/UW_PET_Data/2024-07/"
    top_nifti_folder = "/mnt/Bradshaw/UW_PET_Data/external_testset/"

    files_in_directory = os.listdir(dir_path)

    no_pt_files_list = []
    index = 0
    found_pet_images = 0
    already_converted = 0
    dicom_error = set([])

    for file in files_in_directory:
        print(f"index: {index} already_converted: {already_converted } found pet images: {found_pet_images} file: {file}")
        index += 1
        #if index < 24200:
        #    continue
        folder_name_exists = os.path.join(top_nifti_folder, file)
        if os.path.exists(folder_name_exists):
            if any('SUV' in filename for filename in os.listdir(folder_name_exists)):
                found_pet_images += 1
                already_converted += 1
                print("already found this image with SUV")
                continue

        if file in dicom_error:
            continue
        directory = os.path.join(dir_path, file)
        date = os.listdir(directory)
        if len(date) == 1:
            directory = os.path.join(directory, date[0])
        else:
            print(f"multiple date files in this folder: {directory}")
        modality = os.listdir(directory)
        if "PT" in modality:
            #directory = os.path.join(dir_path, file, "PT")
            directory = os.path.join(directory, "PT")
        else:
            print(f"file: {file} does not have Pet scan")
            continue
        #print(directory)
        ref_num = os.listdir(directory)
        if len(ref_num) == 0:
            print(f"something funny: {file}")
            no_pt_files_list.append(file)
            continue
        directory = os.path.join(directory, ref_num[0])
        # print(test_directory)
        type_exam = os.listdir(directory)
        # print(modality)
        # print(test)

        recon_types = os.listdir(directory)
        substrings_to_check = ["wb_3d_mac", "WB_MAC", "wb_ac_3d", "PET_AC_3D", "WB_IRCTAC"]
        # Iterate over each substring and check if it's present in any element of recon_types
        for substring in substrings_to_check:
            # Normalize to lower case for case-insensitive comparison
            matched_recon = next((recon for recon in recon_types if substring.lower() in recon.lower()), None)
            if matched_recon:
                # If a match is found, build the path
                top_dicom_folder = os.path.join(directory, matched_recon)
                try:
                    found_pet_images = call_suv_helper(top_dicom_folder, top_nifti_folder, found_pet_images)
                except:
                    continue  # If an error occurs, continue with the next substring


def uw_ct_suv_conversion_v2():

    #dir_path = "/mnt/Bradshaw/UW_PET_Data/dsb2b/"
    #dir_path = "/mnt/Bradshaw/UW_PET_Data/dsb2c/"
    dir_path = "/mnt/Bradshaw/UW_PET_Data/dsb3/"
    dir_path_suv = "/mnt/Bradshaw/UW_PET_Data/SUV_images/"
    top_nifti_folder = "/mnt/Bradshaw/UW_PET_Data/SUV_images/"
    dir_path = "/mnt/Bradshaw/UW_PET_Data/2024-07/"
    dir_path_suv = "/mnt/Bradshaw/UW_PET_Data/external_testset/"
    top_nifti_folder = "/mnt/Bradshaw/UW_PET_Data/external_testset/"

    #files_in_directory = os.listdir(dir_path_suv)
    files_in_directory = os.listdir(dir_path)

    print(f"files in folder: {len(files_in_directory)}")
    no_pt_files_list = []
    index = 0

    missing_inject_info = 0
    potential_suv_images = 0

    num_dates = {}  # key is number of dates in folder value is how many folders have that value
    num_dates[1] = 0
    num_modality = {"PT": 0, "CT": 0, "extra": 0}
    num_study_names = {1: 0, "extra": 0, 0: 0}

    found_cts = 0
    already_found = 0
    matches_dic = {}
    for file in files_in_directory:
        # print(f"index: {index} missing inject info: {missing_inject_info} potential found: {potential_suv_images}")
        # if index > 10:
        #    continue
        print(f"index: {index} already found: {already_found} found cts: {found_cts} filename: {file}")
        # print(file)
        index += 1
        # if index > 100:
        #    break

        folder_name_exists = os.path.join(top_nifti_folder, file)
        if os.path.exists(folder_name_exists):
            print(folder_name_exists)
            list_of_folders = os.listdir(folder_name_exists)
            if any('CT' in filename for filename in list_of_folders) and not any("IRCTAC" in filename for filename in list_of_folders):
                found_cts += 1
                already_found += 1
                print("already found this image with CT")
                continue

        suv_dims = (0, 0, 0)
        suv_path = os.path.join(dir_path_suv, file)
        if not os.path.exists(suv_path):
            continue
        for filename in os.listdir(suv_path):
            if filename.endswith(".nii.gz") and "suv" in filename.lower():
                filepath = os.path.join(suv_path, filename)
                try:
                    # Load the NIfTI file
                    nii = nib.load(filepath)
                    suv_dims = nii.header.get_data_shape()
                except:

                    print("can't get dimensions from suv")

        # print(f"suv_dims: {suv_dims}")
        directory = os.path.join(dir_path, file)
        if not os.path.exists(directory):
            continue
        date = os.listdir(directory)
        if len(date) == 1:
            directory = os.path.join(directory, date[0])
            num_dates[1] += 1
        else:
            print(f"multiple date files in this folder: {directory}")
            if len(date) not in num_dates:
                num_dates[len(date)] = 1
            else:
                num_dates[len(date)] += 1

        modality = os.listdir(directory)

        if len(modality) > 2:
            num_modality["extra"] += 1
        if "CT" in modality:
            num_modality["CT"] += 1
        else:
            continue
        if "CT" in modality:
            directory = os.path.join(directory, "CT")
        else:
            print(f"file: {file} does not have Pet scan")
            continue

        study_name = os.listdir(directory)
        if len(study_name) == 0:
            print(f"something funny: {file}")
            no_pt_files_list.append(file)
            num_study_names[0] += 1
        elif study_name == 1:
            num_study_names[1] += 1
        else:
            num_study_names["extra"] += 1

        directory = os.path.join(directory, study_name[0])
        recon_types = os.listdir(directory)

        substrings_to_check = ["2__CTAC", "CT_MAR", "WB_CT_SLICES", "CT_IMAGES", "WB_Standard"]
        # Iterate over each substring and check if it's present in any element of recon_types
        for substring in substrings_to_check:
            # Normalize to lower case for case-insensitive comparison
            matched_recon = next((recon for recon in recon_types if substring.lower() in recon.lower()), None)
            if matched_recon:
                # If a match is found, build the path
                top_dicom_folder = os.path.join(directory, matched_recon)
                z = len(os.listdir(top_dicom_folder))
                # checks if slices line up other wise don't convert and keep searching
                if z == suv_dims[2]:
                    # Perform your additional logic or function calls here
                    try:
                        found_cts = call_suv_helper(top_dicom_folder, top_nifti_folder, found_cts)
                    except:
                        continue  # If an error occurs, continue with the next substring




def uw_ct_check():

    #dir_path = "/mnt/Bradshaw/UW_PET_Data/dsb2b/"
    #dir_path = "/mnt/Bradshaw/UW_PET_Data/dsb3/"
    #path_list = ["/mnt/Bradshaw/UW_PET_Data/dsb2b/", "/mnt/Bradshaw/UW_PET_Data/dsb2c/" ,"/mnt/Bradshaw/UW_PET_Data/dsb3/"]
    path_list = ["/mnt/Bradshaw/UW_PET_Data/2024-07/"]

    master_dic = {}
    for dir_path in path_list:
        files_in_directory = os.listdir(dir_path)

        print(f"files in folder: {len(files_in_directory)}")
        no_pt_files_list = []
        index = 0

        num_dates = {}  # key is number of dates in folder value is how many folders have that value
        num_dates[1] = 0
        num_modality = {"PT": 0, "CT": 0, "extra": 0}
        num_study_names = {1: 0, "extra": 0, 0: 0}

        found_cts = 0
        already_found = 0

        for file in files_in_directory:
            # print(f"index: {index} missing inject info: {missing_inject_info} potential found: {potential_suv_images}")
            # if index > 10:
            #    continue
            if index % 500 == 0:
                print(f"index: {index} already found: {already_found} found cts: {found_cts} filename: {file} path: {dir_path}")
            # print(file)
            index += 1

            directory = os.path.join(dir_path, file)
            if not os.path.exists(directory):
                continue
            date = os.listdir(directory)
            if len(date) == 1:
                directory = os.path.join(directory, date[0])
                num_dates[1] += 1
            else:
                print(f"multiple date files in this folder: {directory}")
                if len(date) not in num_dates:
                    num_dates[len(date)] = 1
                else:
                    num_dates[len(date)] += 1

            modality = os.listdir(directory)

            if len(modality) > 2:
                num_modality["extra"] += 1
            if "CT" in modality:
                num_modality["CT"] += 1
            else:
                continue
            if "PT" in modality:
                directory = os.path.join(directory, "PT")
            else:
                print(f"file: {file} does not have Pet scan")
                continue

            study_name = os.listdir(directory)
            if len(study_name) == 0:
                print(f"something funny: {file}")
                no_pt_files_list.append(file)
                num_study_names[0] += 1
            elif study_name == 1:
                num_study_names[1] += 1
            else:
                num_study_names["extra"] += 1

            directory = os.path.join(directory, study_name[0])
            recon_types = os.listdir(directory)
            recon_path_base = os.path.join(dir_path,file,date[0],modality[0],study_name[0])
            for recon in recon_types:
                if file in master_dic:
                    master_dic[file].append(recon_path_base + "/" + recon)
                else:
                    master_dic[file] = [recon_path_base + "/" + recon]

    # Convert dictionary to a list of lists for DataFrame
    data = []
    for file, strings in master_dic.items():
        row = [file] + strings
        data.append(row)

    # Find the maximum length of the lists (to ensure uniform columns)
    max_len = max(len(row) for row in data)

    # Pad lists with empty strings so all rows have the same length
    for row in data:
        while len(row) < max_len:
            row.append('')

    # Create DataFrame
    df = pd.DataFrame(data)

    # Define the column names
    columns = ['File'] + [f'String {i + 1}' for i in range(max_len - 1)]
    df.columns = columns

    # Save DataFrame to Excel file
    df.to_excel('/UserData/Zach_Analysis/external_testset.xlsx', index=False)