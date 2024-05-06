
import os
import pandas as pd
import dicom2nifti
import nibabel as nib
from datetime import datetime
import numpy as np
import glob
import pydicom

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

    # unique names for subjects and scans
    subject_save_name = dicom_id + '_' + dicom_name.replace(' ', '_').replace('__', '_')
    print(f"subject_save_name: {subject_save_name}")
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
        dicom2nifti.dicom_series_to_nifti(top_dicom_folder,
                                          os.path.join(subject_save_folder, scan_save_name + '.nii.gz'),
                                          reorient_nifti=False)
        #print(f" about to call convert to nifiti to suv nifiti")
        convert_pet_nifti_to_suv_nifti(os.path.join(subject_save_folder, scan_save_name + '.nii.gz'), test_dicom,
                                       os.path.join(subject_save_folder, scan_save_name + '_SUV.nii.gz'))





def files_transfer_analysis():
    dir_path = "/mnt/dsb2b/"
    files_in_directory = os.listdir(dir_path)
    print(files_in_directory)

    df_path = "/UserData/UW_PET_Data/UWPETCTWB_Research-Image-Requests_20240311.xlsx"
    df = pd.read_excel(df_path)

    # Filter the DataFrame to include only rows where the filename is not in files_in_directory
    filtered_df = df[~df['Coded Accession Number'].isin(files_in_directory)]

    # Save the filtered DataFrame to an Excel file
    output_file_path = '/UserData/UW_PET_Data/missing_accession_numbers.xlsx'
    filtered_df.to_excel(output_file_path, index=False)


def file_exploration_analysis():
    dir_path = "/mnt/Bradshaw/UW_PET_Data/dsb2b/"

    files_in_directory = os.listdir(dir_path)
    print(f"files in folder: {len(files_in_directory)}")
    no_pt_files_list = []
    index = 0

    missing_inject_info = 0
    potential_suv_images = 0

    num_dates = {} # key is number of dates in folder value is how many folders have that value
    num_dates[1] = 0
    num_modality = {"PT": 0, "CT": 0, "extra": 0}
    num_study_names = {1:0 , "extra": 0, 0:0}
    types_of_scans_ct = {}
    types_of_scans_pt = {}

    for file in files_in_directory:
        #print(f"index: {index} missing inject info: {missing_inject_info} potential found: {potential_suv_images}")
        #print(f"index: index")
        index += 1
        #if index > 100:
        #    break

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
            print(f"file: {file} does not have ct scan modality: {modality}")
            continue

        # print(modality)
        if "PT" in modality:
            # directory = os.path.join(dir_path, file, "PT")
            directory = os.path.join(directory, "PT")
            num_modality["PT"] += 1
        else:
            print(f"file: {file} does not have Pet scan modality: {modality}")
            continue
        # print(directory)
        study_name = os.listdir(directory)
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
        for recon in recon_types:

            if recon in types_of_scans_pt:
                types_of_scans_pt[recon] += 1
            else:
                types_of_scans_pt[recon] = 1


    print(f"number of dates in files: {num_dates}")
    print(f"number of modality in date file: {num_modality}")
    print(f"types of scans: {types_of_scans_pt}")


def uw_pet_suv_conversion():

    file_exploration_analysis()
    print(fail)
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
    no_pt_files = set([])
    time_data_skip = set([])
    dicom_error = set([])
    weird_path_names = []
    time_errors = []
    for file in files_in_directory:
        print(f"index: {index} missing inject info: {missing_inject_info} potential found: {potential_suv_images}")
        index += 1
        if index > 100:
            break
        #if index < 4630:
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

            except missing_injection_time as e:
                print("missing inject time: {e}")
                missing_inject_info += 1
                continue

            found_pet_images += 1
            continue
        if any("wb_ac_3d" in element.lower() for element in test):
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

            except missing_injection_time as e:
                print("missing inject time: {e}")
                missing_inject_info += 1
                continue
            found_pet_images += 1
            continue
        if any("12__WB_MAC" == element for element in test):
            potential_suv_images += 1
            top_dicom_folder = os.path.join(test_directory, "12__WB_MAC")
            try:
                convert_PT_CT_files_to_nifti(top_dicom_folder, top_nifti_folder)
            except ValueError:
                print("failed")
                time_errors.append(file)
                continue
            except missing_injection_time as e:
                print("missing inject time: {e}")
                missing_inject_info += 1
                continue
            found_pet_images += 1
            #continue