
import os
import pandas as pd

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



def uw_pet_suv_conversion():


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
    skip_files = set([])
    no_pt_files = set([])
    time_data_skip = set([])
    dicom_error = set([])
    weird_path_names = []
    time_errors = []
    for file in files_in_directory:
        print(index)
        index += 1
        if index > 5:
            break
        #if index < 4630:
        #    continue
        if file in skip_files or file in no_pt_files or file in time_data_skip or file in dicom_error:
            continue
        directory = os.path.join(dir_path, file)
        date = os.listdir(directory)
        directory = os.path.join(directory, date)
        modality = os.listdir(directory)
        print(modality)
        if "PT" in modality:
            directory = os.path.join(dir_path, file, "PT")
        else:
            print(f"file: {file} does not have Pet scan")
            continue
        print(directory)
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

        if 'PET_CT_SKULL_BASE_TO_THIGH' in type_exam:
            folder_name = 'PET_CT_SKULL_BASE_TO_THIGH'
        elif len(type_exam) > 1:
            weird_path_names.append(file)
            multi_length += 1
            continue
        else:
            folder_name = type_exam[0]

        test_directory = os.path.join(directory, folder_name)
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