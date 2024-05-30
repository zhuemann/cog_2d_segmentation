# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
import os

import pandas as pd
import argparse


from train_segmentation import train_image_text_segmentation
#from crop_images import crop_images_to_mips
#from sub_region_sub_text_creation import make_clavicular_mips, make_connected_component_labels
#from running_mixstral import run_mixstal
#from sub_region_label_and_image_creation import make_mips_from_3d_data
#from get_max_pixel import get_max_pixel_step3
#from create_sentence_mips_and_labels import create_mips
from data_prepocessing.data_pipeline import run_data_pipeline
from uw_pet_suv_conversion import uw_pet_suv_conversion
from data_prepocessing.utility import get_suv_file_names
from data_prepocessing.utility import finding_missing_images
from data_prepocessing.utility import analyze_ct_series_when_pt_matches
from data_prepocessing.utility import analyze_matching_ct_series_for_pt_substring
from uw_pet_suv_conversion import file_conversion_ct
from data_prepocessing.data_visualization.plot_ct_head_projections import plot_ct_head_projections
from data_prepocessing.utility import generate_data_sheet_on_uw_pet_dataset
# Press the green button in the gutter to run the script.
from uw_pet_suv_conversion import uw_pet_suv_conversion_v2
from uw_pet_suv_conversion import uw_ct_suv_conversion_v2
from data_prepocessing.data_pipeline import run_data_pipeline_final
from data_prepocessing.utility import count_files_in_suv_folder
if __name__ == '__main__':

    count_files_in_suv_folder()
    #print(fail)
    #generate_data_sheet_on_uw_pet_dataset()
    #print(fail)
    #df = "/UserData/Zach_Analysis/petlymph_image_data/uw_final_df_9.xlsx"
    #df = pd.read_excel(df)
    #get_suv_file_names(df)
    #uw_pet_suv_conversion()
    #uw_pet_suv_conversion_v2()
    #uw_ct_suv_conversion_v2()
    print(fail)
    #ct_series_count = analyze_ct_series_when_pt_matches(root_dir="/mnt/Bradshaw/UW_PET_Data/dsb2b/",  pt_substring="WB_IRCTAC")
    #print(ct_series_count)
    #ct_series_count = analyze_matching_ct_series_for_pt_substring(root_dir="/mnt/Bradshaw/UW_PET_Data/dsb2b/",  pt_substring="WB_IRCTAC")
    #print(ct_series_count)
    #finding_missing_images()
    #file_conversion_ct()
    #plot_ct_head_projections()
    #print(fail)

    #run_data_pipeline_final()
    #print(fail)
    #make_connected_component_labels_for_all_subregions()
    #make_mips_from_3d_data()
    #create_mips()
    #test()
    #get_max_pixel_step3()
    #print(fail)
    #run_mixstal()
    #make_connected_component_labels()
    #make_clavicular_mips()
    #print(fail)

    #crop_images_to_mips()
    #print(fail)
    local = False
    if local:
        directory_base = "Z:/"
    else:
        directory_base = "/UserData/"

    config = {"seed": 1, "batch_size": 32, "dir_base": directory_base, "epochs": 400, "n_classes": 2, "LR": 1e-3,
              "IMG_SIZE": (192, 384), "train_samples": .8, "test_samples": .5, "data_path": "D:/candid_ptx/", #"IMG_SIZE": (128, 256)
              "report_gen": False, "mlm_pretraining": False, "contrastive_training": False, "save_location": ""}

    print("here")
    #seeds = [98]
    seeds = [98, 117, 295, 456, 915]
    # seeds = [915]

    # seeds = [456]
    # seeds = [1289, 1734]
    # seeds = [456]
    # seeds = [295, 456, 915]

    accuracy_list = []

    for seed in seeds:

        #folder_name = "cropped_mips/baseline_nnunet_v17/seed" + str(seed) + "/"
        #save_string = "/UserData/Zach_Analysis/result_logs/cog_mip_segmentation/initial_testing/" + folder_name
        folder_name = "/contextual_net_1000ep_uw_dataset_unfrozen_at_250_v8/seed" + str(seed) + "/"
        save_string = "/UserData/Zach_Analysis/result_logs/visual_grounding/using_mips/initial_testing" + folder_name
        save_location = os.path.join(directory_base, save_string)
        # save_location = ""

        config["seed"] = seed
        config["save_location"] = save_location
        # make_images_on_dgx(config)
        # print(fail)

        acc, valid_log, correct_suv_log, max_predictions = train_image_text_segmentation(config)
        df = pd.DataFrame(valid_log)
        df["valid_correct_max"] = correct_suv_log
        df["test_accuracy"] = acc
        df["test_correct_max_predictions"] = max_predictions


        filepath = os.path.join(config["save_location"], "valid_1000ep_seed" + str(seed) + '.xlsx')
        df.to_excel(filepath, index=False)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
