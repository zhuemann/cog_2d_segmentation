# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
import os

import pandas as pd
import argparse


from train_segmentation import train_image_text_segmentation
from crop_images import crop_images_to_mips


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #crop_images_to_mips()
    #print(fail)
    local = False
    if local:
        directory_base = "Z:/"
    else:
        directory_base = "/UserData/"

    config = {"seed": 1, "batch_size": 8, "dir_base": directory_base, "epochs": 2000, "n_classes": 2, "LR": 1e-3,
              "IMG_SIZE": (128,256), "train_samples": .8, "test_samples": .5, "data_path": "D:/candid_ptx/",
              "report_gen": False, "mlm_pretraining": False, "contrastive_training": False, "save_location": ""}

    print("here")
    seeds = [98]
    # seeds = [98, 117, 295, 456]
    # seeds = [915]
    # seeds = [456]
    # seeds = [1289, 1734]
    # seeds = [456]

    accuracy_list = []

    for seed in seeds:

        folder_name = "cropped_mips/interim_VL_frozen_lang_long_hybrid_aug_v16/seed" + str(seed) + "/"
        save_string = "/UserData/Zach_Analysis/result_logs/cog_mip_segmentation/initial_testing/" + folder_name

        save_location = os.path.join(directory_base, save_string)
        # save_location = ""

        config["seed"] = seed
        config["save_location"] = save_location
        # make_images_on_dgx(config)
        # print(fail)

        acc, valid_log = train_image_text_segmentation(config)
        df = pd.DataFrame(valid_log)
        df["test_accuracy"] = acc

        filepath = os.path.join(config["save_location"], "valid_150ep_seed" + str(seed) + '.xlsx')
        df.to_excel(filepath, index=False)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
