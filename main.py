# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os

import pandas as pd

from train_segmentation import train_image_text_segmentation
#from crop_images import crop_images_to_mips
#from sub_region_sub_text_creation import make_clavicular_mips, make_connected_component_labels
#from running_mixstral import run_mixstal
#from sub_region_label_and_image_creation import make_mips_from_3d_data
#from get_max_pixel import get_max_pixel_step3
#from create_sentence_mips_and_labels import create_mips
# Press the green button in the gutter to run the script.
from segmentation_25d.mip_creation import mip_creation
from segmentation_25d.train_segmentation_25d import train_image_text_segmentation_25d
from segmentation_3d.train_3d_model import train_3d_image_text_segmentation
from data_prepocessing.data_visualization.post_processing_eval import post_processing_eval
from segmentation_3d.post_processing_eval_llmseg import post_processing_eval_llmseg
from external_test_set_creation.physican_labeling.physican_post_processing_eval import physician_post_processing_eval
from external_test_set_creation.physican_labeling.swedish_labeling.process_rt_structs_to_nifti_external import process_rt_strcuts_to_nifty_external
from external_test_set_creation.physican_labeling.swedish_labeling.plot_swedish_labels_external import plot_physican_contours_external
from external_test_set_creation.physican_labeling.swedish_labeling.crop_at_head_external import crop_at_head_calculation_external
from external_test_set_creation.physican_labeling.swedish_labeling.resampling_and_cropping_external import resampling_and_cropping_external
from external_test_set_creation.physican_labeling.swedish_labeling.plot_external_labels_for_josh import plot_final_testset_for_josh_v3_external
from data_prepocessing.data_visualization.plot_3d_predictions_single_image import plot_all_images
from data_prepocessing.data_visualization.intereactive_report_figure import make_interactive_figure
from data_prepocessing.data_visualization.intereactive_report_figure import compound_interactive_report_v2
if __name__ == '__main__':
    #plot_final_internal_dataset()
    #mip_creation()
    #process_rt_strcuts_to_nifty_external()
    #plot_physican_contours_external()
    #crop_at_head_calculation_external()
    #plot_final_testset_for_josh_v3_external()
    #resampling_and_cropping_external()
    #post_processing_eval()
    #plot_all_images()
    #post_processing_eval_llmseg()
    #copy_images_and_labels_to_folder()
    #df = pd.read_excel("/UserData/Zach_Analysis/physican_labeling_UWPET/Josh_worksheet_matched.xlsx")
    #resampling_and_cropping(df)
    #process_rt_strcuts_to_nifty()
    #rename_nifti_files()
    #plot_physican_contours()
    #post_processing_eval()
    #copy_physican_labels_to_folder()
    #physician_post_processing_eval()
    #copy_physican_labels_to_folder()
    #process_rt_strcuts_to_nifty()
    compound_interactive_report_v2()
    #make_interactive_figure()
    print(fail)
    #compound_interactive_report_v2()
    #testing_ploting_external_cog_data()
    #precomputed_language_embeddings()
    #uw_ct_conversion_external_dataset_v2()
    #external_get_max_pixel()
    #df = make_labels_from_suv_max_points()
    #plot_external_testset(df)
    #plot_for_orientation_and_modality()
    #external_get_max_pixel()
    #get_dicoms_external_testset()
    #pet_suv_conversion_external_v3()
    #uw_ct_conversion_external_dataset_v2()
    #get_orientation_from_dicom()
    #plot_physican_contours()
    #process_rt_strcuts_to_nifty()
    #print(fail)
    #make_interactive_figure()
    #print(fail)
    #get_dicoms_for_reading()
    #print(fail)
    #tracer_type_all_files()
    #print(fail)
    #ct_check()
    #pet_suv_conversion_external_v3()
    #make_all_rts()

    #uw_ct_conversion_external_dataset_v2()
    #external_get_max_pixel()
    #print(fail)
    #print(fail)
    #uw_ct_check()
    #print(fail)
    #count_files_in_suv_folder()
    #print(fail)
    #generate_data_sheet_on_uw_pet_dataset()
    #print(fail)
    #df = "/UserData/Zach_Analysis/petlymph_image_data/uw_final_df_9.xlsx"
    #df = pd.read_excel(df)
    #get_suv_file_names(df)
    #uw_pet_suv_conversion()
    #uw_pet_suv_conversion_v2()
    #uw_ct_suv_conversion_v2()
    #file_exploration_analysis_pet()
    #uw_ct_conversion_external_dataset_v2()
    #print(fail)
    #scanner_types_external_test_set()
    #print(fail)
    #ct_series_count = analyze_ct_series_when_pt_matches(root_dir="/mnt/Bradshaw/UW_PET_Data/dsb2b/",  pt_substring="WB_IRCTAC")
    #print(ct_series_count)
    #ct_series_count = analyze_matching_ct_series_for_pt_substring(root_dir="/mnt/Bradshaw/UW_PET_Data/dsb2b/",  pt_substring="WB_IRCTAC")
    #print(ct_series_count)
    #finding_missing_images()
    #file_conversion_ct()
    #plot_ct_head_projections()
    #print(fail)
    #uw_ct_suv_conversion_v2()
    #run_data_pipeline_final()
    #print(fail)
    #plot_training_and_inference_images()
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

    config = {"seed": 1, "batch_size": 16, "dir_base": directory_base, "epochs": 250, "n_classes": 2, "LR": 1e-3,
              "IMG_SIZE": (200, 350), "train_samples": .8, "test_samples": .5, "data_path": "D:/candid_ptx/", #"IMG_SIZE": (128, 256)
              "report_gen": False, "mlm_pretraining": False, "contrastive_training": False, "save_location": "/UserData/Zach_Analysis/result_logs/visual_grounding/25D_experiments/two_channel_input_sagittal_v2/seed1/"}


    acc, valid_log, correct_suv_log, max_predictions = train_image_text_segmentation_25d(config)
    df = pd.DataFrame(valid_log)
    df["valid_correct_max"] = correct_suv_log
    df["test_accuracy"] = acc
    df["test_correct_max_predictions"] = max_predictions

    filepath = os.path.join(config["save_location"], "valid_250ep_seed" + str(config["seed"]) + '.xlsx')
    df.to_excel(filepath, index=False)
    print("here")

    print(fail)
    #train_3d_image_text_segmentation(config)
    #print(fail)
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
        #folder_name = "/contextual_net_1000ep_uw_dataset_report_attention_v9/seed" + str(seed) + "/"
        #save_string = "/UserData/Zach_Analysis/result_logs/visual_grounding/using_mips/initial_testing" + folder_name
        folder_name = "/seed" + str(seed) + "/"
        save_string = "/UserData/Zach_Analysis/result_logs/visual_grounding/25D_experiments/sagital_channel_original" + folder_name
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
