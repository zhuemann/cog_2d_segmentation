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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    run_data_pipeline()
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

    config = {"seed": 1, "batch_size": 32, "dir_base": directory_base, "epochs": 1000, "n_classes": 2, "LR": 1e-3,
              "IMG_SIZE": (128, 256), "train_samples": .8, "test_samples": .5, "data_path": "D:/candid_ptx/", #"IMG_SIZE": (128, 256)
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
        folder_name = "/contextual_net_higher_lr_2000ep_threshold_labels5_v23/seed" + str(seed) + "/"
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
