import os
import pydicom
import dicom2nifti
from platipy.dicom.io import rtstruct_to_nifti
import nibabel as nib
from datetime import datetime
import numpy as np
import glob

import os

def all_subdirs_search(top_folder):
    subfolders = [f.path for f in os.scandir(top_folder) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(all_subdirs_search(dirname))
    return subfolders

def get_all_terminal_subfolders(top_folder):
    dirs = all_subdirs_search(top_folder)
    terminal_subfolders = dirs.copy()
    for dir_i in dirs:
        for dir_j in dirs:
            if dir_i == dir_j:
                continue
            if dir_i in dir_j:
                terminal_subfolders.remove(dir_i)
                break
    return terminal_subfolders

def isdicom(file_path):
    #borrowed from pydicom filereader.py
    with open(file_path, 'rb') as fp:
        preamble = fp.read(128)
        magic = fp.read(4)
        if magic != b"DICM":
            return False
        else:
            return True


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
            if radiopharm_object['00181078'].value != None:
                dicom_inj_datetime = radiopharm_object['00181078'].value[:14]  # Radiopharmaceutical Start DateTime
            else:
                dicom_inj_datetime = dicom_scan_datetime[0:8] + radiopharm_object['00181072'].value
        else:
            dicom_inj_datetime = dicom_scan_datetime[0:8] + radiopharm_object['00181072'].value
        # convert dicom_inj_datetime to string
        dicom_inj_datetime = str(dicom_inj_datetime)
        non_decimal = [char for char in dicom_inj_datetime if char.isdigit()]
        dicom_inj_datetime = ''.join(non_decimal)
    # sometimes tracer info is wiped, and if GE, can be found in private tags
    else:
        print('No dose information -- SUV factor set to 1')
        return 1, 0

    dicom_inj_datetime = dicom_inj_datetime[:14]  # year(4)/month(2)/day(2)/hour(2)/minute(2)/second(2)
    dicom_scan_datetime = dicom_scan_datetime[:14]  # year(4)/month(2)/day(2)/hour(2)/minute(2)/second(2)
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
        return False, dicom_weight

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
    subject_save_folder = os.path.join(top_nifti_folder, subject_save_name)
    os.makedirs(subject_save_folder, exist_ok=True)
    scan_save_name = '{}_{}_{}_{}'.format(subject_save_name, dicom_study_date, dicom_modality, \
                                          dicom_series_description.replace(' ', '_'))

    if dicom_modality in ['CT', 'MR', 'NM']:
        dicom2nifti.dicom_series_to_nifti(top_dicom_folder,
                                          os.path.join(subject_save_folder, scan_save_name + '.nii.gz'),
                                          reorient_nifti=False)
    elif dicom_modality == 'PT':
        dicom2nifti.dicom_series_to_nifti(top_dicom_folder,
                                          os.path.join(subject_save_folder, scan_save_name + '.nii.gz'),
                                          reorient_nifti=False)
        convert_pet_nifti_to_suv_nifti(os.path.join(subject_save_folder, scan_save_name + '.nii.gz'), test_dicom,
                                       os.path.join(subject_save_folder, scan_save_name + '_SUV.nii.gz'))


def convert_rtstruct_to_nifti(annotator_dicom_folder: str, top_nifti_folder: str, mismatch_case: str,
                              modality_of_interest: str = 'PT'):
    # modality of interest is the modality that will be the reference size for the RTSTRUCT contours

    files = glob.glob(annotator_dicom_folder + "/*.dcm")
    if len(files) < 1:
        print('Empty folder: ' + files)
        raise Exception("Fail to find DICOM files")

    # get dicom info for saving
    test_dicom = pydicom.dcmread(files[0])
    dicom_modality = test_dicom['00080060'].value
    dicom_name = str(test_dicom['00100010'].value).lower()
    dicom_id = test_dicom['00100020'].value.lower()

    # unique names for subjects and scans
    subject_save_name = dicom_id + '_' + dicom_name.replace(' ', '_').replace('__', '_')
    subject_save_folder = os.path.join(top_nifti_folder, subject_save_name)
    os.makedirs(subject_save_folder, exist_ok=True)

    # if dicom_modality == 'RTSTRUCT' and rtstruct_string_identifier.lower() in dicom_series_description.lower():
    if dicom_modality == 'RTSTRUCT':
        # might be multiple rtstructs in folder
        for file_i in files:
            if isdicom(file_i) == True:
                rt_dicom = pydicom.dcmread(file_i)
                if rt_dicom['00080060'].value == 'RTSTRUCT':
                    dicom_modality = rt_dicom['00080060'].value
                    dicom_name = str(rt_dicom['00100010'].value).lower()
                    dicom_id = rt_dicom['00100020'].value.lower()
                    dicom_study_date = rt_dicom['00080020'].value
                    dicom_series_description = rt_dicom['0008103e'].value

                    # unique names for subjects and scans
                    subject_save_name = dicom_id + '_' + dicom_name.replace(' ', '_').replace('__', '_')
                    subject_save_folder = os.path.join(top_nifti_folder, subject_save_name)
                    scan_save_name = dicom_study_date + '_' + dicom_modality + '_' + dicom_series_description.replace(
                        ' ', '_')

                    # find the corresponding DICOM series
                    one_folder_up = os.path.dirname(annotator_dicom_folder)
                    subdirs = os.listdir(one_folder_up)
                    subdirs = [os.path.join(one_folder_up, s) for s in subdirs if mismatch_case in s]
                    corresp_dicom_path = find_path_to_dicom_image_that_corresponds_with_rtsrtuct(annotator_dicom_folder,
                                                                                                 dicom_id,
                                                                                                 dicom_study_date,
                                                                                                 modality_of_interest,
                                                                                                 subdirs)
                    if corresp_dicom_path == '':
                        print(
                            '***!!! Unable to find correspoding DICOM images for %s. RTStruct will not be made ***!!'.format(
                                subject_save_name))
                        continue
                    # save
                    rtstruct_nifti_save_path = os.path.join(subject_save_folder, scan_save_name)
                    if not os.path.exists(rtstruct_nifti_save_path):
                        os.makedirs(rtstruct_nifti_save_path)
                    try:
                        rtstruct_to_nifti.convert_rtstruct(corresp_dicom_path, file_i,
                                                           output_dir=rtstruct_nifti_save_path)
                    except:
                        print("!!!!!!!!!!!XXXXXXX   Problem with {}    XXXXXX!!!!!!!!!!!!!! ".format(file_i))

                    # check dimensions match
                    roi_files = os.listdir(rtstruct_nifti_save_path)
                    roi_file = os.path.join(rtstruct_nifti_save_path, roi_files[0])
                    roi_nii = nib.load(roi_file)
                    nii_dims = roi_nii.header.get_data_shape()
                    dicom_files = os.listdir(corresp_dicom_path)
                    dicom_file = os.path.join(corresp_dicom_path, dicom_files[0])
                    dicom_info = pydicom.dcmread(dicom_file)
                    rows = dicom_info['00280010'].value
                    if rows != nii_dims[0] or len(dicom_files) != nii_dims[2]:
                        print('!!!**** ROI nifti is not same shape as DICOM (maybe) for {}'.format(
                            rtstruct_nifti_save_path))


def test():



    #top_dicom_folder = "/UserData/1043/PETLYMPH_3004/PT/20150125/BODY/1203__PET_CORONAL/"
    top_nifti_folder = "/UserData/Zach_Analysis/suv_nifti_test/"
    #convert_PT_CT_files_to_nifti(top_dicom_folder, top_nifti_folder)

    dir_path = "/UserData/1043/"
    files_in_directory = os.listdir(dir_path)
    no_pt_files_list = []
    index = 0
    found_pet_images = 0
    multi_length = 0
    skip_files = set(
        ["PETLYMPH_4491", "PETLYMPH_4490", "PETLYMPH_4262", "PETLYMPH_0261", "PETLYMPH_0903", "PETLYMPH_3588",
         "PETLYMPH_0902"])
    no_pt_files = set(
        ['PETLYMPH_2965', 'PETLYMPH_2729', 'PETLYMPH_2831', 'PETLYMPH_4357', 'PETLYMPH_3233', 'PETLYMPH_2685',
         'PETLYMPH_3014', 'PETLYMPH_2795', 'PETLYMPH_3876', 'PETLYMPH_2883', 'PETLYMPH_3914', 'PETLYMPH_2628',
         'PETLYMPH_3804', 'PETLYMPH_3936', 'PETLYMPH_0222', 'PETLYMPH_4222', 'PETLYMPH_2776', 'PETLYMPH_2694',
         'PETLYMPH_3730', 'PETLYMPH_4451', 'PETLYMPH_2954', 'PETLYMPH_4050', 'PETLYMPH_2468', 'PETLYMPH_0130',
         'PETLYMPH_2907', 'PETLYMPH_2498', 'PETLYMPH_2697', 'PETLYMPH_4392', 'PETLYMPH_3232', 'PETLYMPH_2432',
         'PETLYMPH_2852', 'PETLYMPH_4065'])
    time_data_skip = set(["PETLYMPH_2565", "PETLYMPH_2529"])
    dicom_error = set(["PETLYMPH_1259", "PETLYMPH_3042", "PETLYMPH_1258", "PETLYMPH_1686", "PETLYMPH_3180"])
    weird_path_names = []
    time_errors = []
    for file in files_in_directory:
        print(index)
        index += 1
        if index < 4630:
            continue
        if file in skip_files or file in no_pt_files or file in time_data_skip or file in dicom_error:
            continue
        test_directory = os.path.join(dir_path, file)
        modality = os.listdir(test_directory)

        if "PT" in modality:
            test_directory = os.path.join(dir_path, file, "PT")
        else:
            print(f"file: {file} does not have Pet scan")
            continue

        ref_num = os.listdir(test_directory)
        if len(ref_num) == 0:
            print(f"something funny: {file}")
            no_pt_files_list.append(file)
            continue
        # print(ref_num)
        test_directory = os.path.join(test_directory, ref_num[0])
        # print(test_directory)
        type_exam = os.listdir(test_directory)
        # print(modality)
        # print(test)

        if 'PET_CT_SKULL_BASE_TO_THIGH' in type_exam:
            folder_name = 'PET_CT_SKULL_BASE_TO_THIGH'
        elif len(type_exam) > 1:
            weird_path_names.append(file)
            multi_length += 1
            continue
        else:
            folder_name = type_exam[0]

        test_directory = os.path.join(test_directory, folder_name)
        test = os.listdir(test_directory)
        print(test)
        if any("12__wb_3d_mac" in element.lower() for element in test):
            top_dicom_folder = os.path.join(test_directory, "12__WB_3D_MAC")
            print(f"top: {top_dicom_folder}")
            try:
                convert_PT_CT_files_to_nifti(top_dicom_folder, top_nifti_folder)
            except ValueError:
                time_errors.append(file)
                continue

            found_pet_images += 1
            continue
        if any("wb_ac_3d" in element.lower() for element in test):
            indices_of_pet = [index for index, element in enumerate(test) if "wb_ac_3d" in element.lower()]
            top_dicom_folder = os.path.join(test_directory, test[indices_of_pet[0]])
            print(f"top: {top_dicom_folder}")
            try:
                convert_PT_CT_files_to_nifti(top_dicom_folder, top_nifti_folder)
            except ValueError:
                time_errors.append(file)
                continue
            found_pet_images += 1
            continue
        if any("12__WB_MAC" == element for element in test):
            top_dicom_folder = os.path.join(test_directory, "12__WB_MAC")
            try:
                convert_PT_CT_files_to_nifti(top_dicom_folder, top_nifti_folder)
            except ValueError:
                time_errors.append(file)
                continue
            found_pet_images += 1
            continue