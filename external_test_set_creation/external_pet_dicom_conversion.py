
import os
import pandas as pd
import dicom2nifti
import nibabel as nib
from datetime import datetime
import numpy as np
import glob
import pydicom

from nilearn.image import resample_img
from pathlib import Path

#from uw_pet_suv_conversion import call_suv_helper

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
    print(f"units: {test_dicom.Units}")
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

    #print(top_dicom_folder)
    # Ensure the path is correct
    top_dicom_folder_path = Path(top_dicom_folder)

    # Search for .dcm files using Pathlib
    dcm_files = list(top_dicom_folder_path.glob("*.dcm"))
    #print(f"lengh: {len(dcm_files)}")
    # Convert PosixPath objects to strings
    files = [str(file) for file in dcm_files]
    #print(len(files))
    #files = glob.glob(top_dicom_folder + "/*.dcm")

    if len(files) < 1:
        print("will error not calling right folder")
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
    #print(folder_names)
    #indices_of_pet = [index for index, element in enumerate(folder_names) if "petwb_" in element.lower()]
    print(f"top dicom folder: {folder_names}")
    indices_of_pet = [index for index, element in enumerate(folder_names) if "sah" in element.lower()]

    print(f"indices: {indices_of_pet}")
    print(f"test: {folder_names[indices_of_pet[0]]}")
    # unique names for subjects and scans
    #subject_save_name = dicom_id + '_' + dicom_name.replace(' ', '_').replace('__', '_')
    #print(f"subject_save_name: {subject_save_name}")
    subject_save_name = folder_names[indices_of_pet[0]]
    print(f"subject_save_name: {subject_save_name}")

    subject_save_folder = os.path.join(top_nifti_folder, subject_save_name)
    print(f"subject save folder: {subject_save_folder}")
    os.makedirs(subject_save_folder, exist_ok=True)
    scan_save_name = '{}_{}_{}_{}'.format(subject_save_name, dicom_study_date, dicom_modality, \
                                          dicom_series_description.replace(' ', '_'))
    #scan_save_name = '{}_{}_{}'.format(subject_save_name, dicom_study_date, dicom_modality)
    scan_save_name = scan_save_name.replace("/", "_")
    scan_save_name = scan_save_name.replace(":", "_")
    scan_save_name = scan_save_name.replace("[", "_")
    scan_save_name = scan_save_name.replace("]", "_")


    print(f"save save name: {scan_save_name}")

    if dicom_modality in ['CT', 'MR', 'NM']:
        #print(f"dicom_modality: {dicom_modality}")
        dicom2nifti.dicom_series_to_nifti(top_dicom_folder,
                                          os.path.join(subject_save_folder, scan_save_name + '.nii.gz'),
                                          reorient_nifti=False)
    elif dicom_modality == 'PT':
        #print(f"dicom_modality: {dicom_modality}")
        #print(f"about to call dicom to nifiti")
        print(f"top dicom folder: {top_dicom_folder}")
        print(f"{os.listdir(top_dicom_folder)}")
        dicom2nifti.dicom_series_to_nifti(top_dicom_folder,
                                          os.path.join(subject_save_folder, scan_save_name + '.nii.gz'),
                                          reorient_nifti=False)
        print(f" about to call convert to nifiti to suv nifiti")
        convert_pet_nifti_to_suv_nifti(os.path.join(subject_save_folder, scan_save_name + '.nii.gz'), test_dicom,
                                   os.path.join(subject_save_folder, scan_save_name + '_SUV.nii.gz'))
        """
        try:
            dicom2nifti.dicom_series_to_nifti(top_dicom_folder,
                                          os.path.join(subject_save_folder, scan_save_name + '.nii.gz'),
                                          reorient_nifti=False)
        #print(f" about to call convert to nifiti to suv nifiti")
            #convert_pet_nifti_to_suv_nifti(os.path.join(subject_save_folder, scan_save_name + '.nii.gz'), test_dicom,
            #                           os.path.join(subject_save_folder, scan_save_name + '_SUV.nii.gz'))
        except ValueError:
            print("slice error")
            os.rmdir(subject_save_folder)
        except Exception as e:
            print("error doing conversion")
            print(e)
            os.rmdir(subject_save_folder)
        """




def call_suv_helper(top_dicom_folder, top_nifti_folder, found_cts):

    convert_PT_CT_files_to_nifti(top_dicom_folder, top_nifti_folder)
    """
    try:
        convert_PT_CT_files_to_nifti(top_dicom_folder, top_nifti_folder)
    except Exception:
        print("call suv helper exception")
        found_cts -= 1
    """

    found_cts += 1
    return found_cts

def pet_suv_conversion_external_v3():

    dir_path = "/mnt/dsb2/BRADSHAWtyler.20240716__201511/RefactoredBags/"
    top_nifti_folder = "/mnt/Bradshaw/UW_PET_Data/external_testset_try3/"


    df = pd.read_excel("/UserData/Zach_Analysis/suv_slice_text/swedish_hospital_external_data_set/Swedish_sentences_with_uw_ids.xlsx")

    files_in_directory = os.listdir(dir_path)

    no_pt_files_list = []
    index = 0
    found_pet_images = 0
    already_converted = 0
    dicom_error = set([])

    missing_pet = 0

    # for file in files_in_directory:
    for index, row in df.iterrows():
        file = row["ID"]
        print(f"index: {index} already_converted: {already_converted } found pet images: {found_pet_images} file: {file} missing pet: {missing_pet}")
        index += 1
        #if index < 24200:
        #    continue

        folder_name_exists = os.path.join(top_nifti_folder, file)
        if os.path.exists(folder_name_exists):
            if any('SUV' in filename for filename in os.listdir(folder_name_exists)):
                #print(f"folder name: {folder_name_exists}")
                found_pet_images += 1
                already_converted += 1
                print("already found this image with SUV")
                continue

        if file not in os.listdir(dir_path):
            print("don't have folder")
            missing_pet += 1
            continue

        if file in dicom_error:
            continue
        directory = os.path.join(dir_path, file)

        random_id = os.listdir(directory)
        if len(random_id) == 1:
            directory = os.path.join(directory, random_id[0])

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
        substrings_to_check = ["WB_CTAC"]
        #print(f"recon_types: {recon_types}")
        # Iterate over each substring and check if it's present in any element of recon_types
        print(f"recon types: {recon_types}")
        for substring in substrings_to_check:
            # Normalize to lower case for case-insensitive comparison
            #matched_recon = next((recon for recon in recon_types if substring.lower() in recon.lower()), None)
            for matched_recon in recon_types:

                if "wb_ctac" not in matched_recon.lower() and "pet_ac_2d" not in matched_recon.lower():
                    continue

                #if matched_recon == None or "fused_trans" not in matched_recon.lower() or "mip" in matched_recon.lower():
                #    continue
                if matched_recon == None or "fused" in matched_recon.lower() or "mip" in matched_recon.lower():
                    continue
                print(f"matched: {matched_recon}")
                if matched_recon:
                    # If a match is found, build the path
                    top_dicom_folder = os.path.join(directory, matched_recon, file)
                    #top_dicom_folder = os.path.join(directory, matched_recon)

                    #top_dicom_folder = directory + "/" + str(matched_recon) + ""
                    print(f"top dicom folder: {top_nifti_folder}")
                    found_pet_images = call_suv_helper(top_dicom_folder, top_nifti_folder, found_pet_images)
                    """
                    try:
                        found_pet_images = call_suv_helper(top_dicom_folder, top_nifti_folder, found_pet_images)
                        break
                    except Exception as e:
                        print("this is the erorr thrown")
                        print(f"error: {e}")
                        continue  # If an error occurs, continue with the next substring
                    """