#from data_prepocessing.remove_dups_non_anontomical_sent import get_anatomical_dataframe
from data_prepocessing.remove_non_anontomical_sent import remove_non_anatomical_sent_v2
from data_prepocessing.data_visualization.plot_final_test_set_josh import plot_final_testset_for_josh_v3
from data_prepocessing.data_visualization.plots_for_label_accuracy import plot_for_label_accuracy_assessment
from data_prepocessing.data_visualization.plot_3d_predictions_single_image import plot_all_images
from data_prepocessing.utility.select_250_images_from_json import select_250_images_from_json
from data_prepocessing.make_json_file_for_3d_training import make_json_file_for_3d_training
from data_prepocessing.data_visualization.plot_final_test_set import plot_final_testset_v2
from data_prepocessing.data_visualization.plot_final_test_set import plot_final_testset

from data_prepocessing.resampling_and_cropping import resampling_and_cropping
from data_prepocessing.data_visualization.post_processing_eval import post_processing_eval
from data_prepocessing.utility.ct_analysis import ct_analysis
from data_prepocessing.data_visualization.plot_3d_predictions import plot_3d_predictions
from data_prepocessing.utility.get_entries_to_remove import get_entries_to_remove
from uw_pet_suv_conversion import uw_ct_check
from data_prepocessing.data_visualization.plot_3d_predictions import plot_3d_predictions_single_image
import pandas as pd
from data_prepocessing.utility.scanner_types import scanner_types
def run_data_pipeline():

    #save_base = "/UserData/Zach_Analysis/suv_slice_text/uw_lymphoma_preprocess_chain_v3/"
    #save_base = "/UserData/Zach_Analysis/suv_slice_text/uw_lymphoma_preprocess_chain_v11/"
    save_base = "/UserData/Zach_Analysis/suv_slice_text/uw_all_pet_preprocess_chain_v1/"
    save_base_final = "/UserData/Zach_Analysis/petlymph_image_data/"

    """
    save_base_final = "/UserData/Zach_Analysis/petlymph_image_data/"

    #df_path = "/UserData/Zach_Analysis/text_data/indications.xlsx"
    df_path = "/UserData/Zach_Analysis/lymphoma_data/all_pet_reports_uw.xlsx"
    df = pd.read_excel(df_path)
    df.to_excel(save_base + "initial_data_0.xlsx", index=False)

    df = split_sentences(df)
    df.to_excel(save_base + "sentences_split_1.xlsx", index=False)

    # split the sentences further or drop ones that have too many suv or slice values.
    df = detect_and_remove_multiple_suv_slice(df)
    df.to_excel(save_base + "remove_multiple_suv_and_slice_2.xlsx") # replace with llm that splits later

    # then into radgraph with next and previous sentence
    print(len(df))
    df = remove_non_anontomical_sent(df)
    df.to_excel(save_base + "remove_non_anotomical_info_3.xlsx", index=False)
    print(len(df))
    
    df = pd.read_excel(save_base + "remove_non_anotomical_info_3.xlsx")


    df = llm_slice_suv_extraction(df)
    df.to_excel(save_base + "model_predictions_for_suv_slice_extraction_4.xlsx", index=False, sheet_name='Predictions')

    #df = pd.read_excel(save_base + "model_predictions_for_suv_slice_extraction_2.xlsx")
    df = concenus_voting(df)
    df.to_excel(save_base + "concenus_output_5.xlsx", index=False)
    #df = pd.read_excel(save_base + "concenus_output_3.xlsx")
    """
    """
    df = pd.read_excel(save_base + "concenus_output_5.xlsx")
    print(f"before 2.5 suv filter {len(df)}")
    # Filter the DataFrame to keep only rows where 'suv' is 2.5 or higher
    df = df[df['SUV'] >= 2.5]
    print(f" After2.5 suv filter {len(df)}")

    df_coded = pd.read_excel("/UserData/UW_PET_Data/UWPETCTWB_Research-Image-Requests_20240311.xlsx")
    # Create a mapping from Original to Coded Accession Numbers
    mapping = df_coded.set_index('Original Accession Number')['Coded Accession Number']
    # Replace the Original Accession Numbers with Coded Accession Numbers in df_original
    df['Petlymph'] = df['Original Accession Number'].map(mapping)

    df.to_excel(save_base + "processed_sentenced_above_threshold_6.xlsx", index=False)
    """

    #df = pd.read_excel(save_base + "processed_sentenced_above_threshold_6.xlsx")
    #df = get_max_pixel_step3(df)
    #df.to_excel(save_base + "max_pixel_7_end.xlsx", index=False)

    """remove this later just concating dataframes"""
    """
    df = pd.read_excel(save_base + "max_pixel_7.xlsx")
    df_1 = pd.read_excel(save_base + "max_pixel_7_140000.xlsx")
    df_2 = pd.read_excel(save_base + "max_pixel_7_21000.xlsx")
    df_3 = pd.read_excel(save_base + "max_pixel_7_end.xlsx")

    df = pd.concat([df, df_1, df_2, df_3])
    print(len(df))
    #df.to_excel(save_base + "max_pixel_7_all.xlsx", index=False)

    df = remove_duplicates(df)
    df = assign_label_numbers(df)

    #df = pd.read_excel(save_base + "max_pixel_4_test_rerun_slice_ref_fixed.xlsx")
    #df = detect_and_remove_multiple_suv_slice(df)
    #print(len(df))
    df.to_excel(save_base + "final_df_8_33000.xlsx", index=False)

    #df = remove_dups_non_anontomical_sent(df)
    #print(df)
    #df.to_excel(save_base + "remove_dups_df_6.xlsx", index=False)
    #df = pd.read_excel(save_base + "remove_dups_df_6.xlsx")
    df = make_labels_from_suv_max_points(df, save_location = "uw_labels_v2_nifti")
    df.to_excel(save_base_final + "uw_final_df_9_all.xlsx", index=False)
    """
    #print(fail)
    """
    need a function that will check makels doing few things things
    1) check all labels for each petlymph/img and if any of the labels with different sentences overlap we drop the row
    2) check petlymph for sentence that are the same, if the labels overlap keep 1 of them
    3) sure label is connected via connectivity neighbors 6, cut positive pixels that are not connected
    4) final check of label to fall within slice range and suv max matches up with noted suv max (increasing purity)
    
    This should take care of the detect_and_remove_multiple_suv_slice call but should still call it after.
    """
    #df = pd.read_excel(save_base + "final_df_8_33000.xlsx")
    #df = remove_non_anontomical_sent(df)

    #df = pd.read_excel(save_base_final + "uw_final_df_9_all.xlsx")
    #print(f"length before template removal: {len(df)}")
    #df = df.rename(columns={'Extracted Sentences': 'sentence'})
    #df = template_removal(df)
    #print(f"length after template removal: {len(df)}")

    # df = count_left_right_sided(df)
    # df.to_excel(save_base_final + "uw_label_wrong_side_analysis_10.xlsx", index=False)
    #df_removal = pd.read_excel(save_base + "uw_label_wrong_side_analysis.xlsx")
    #labels_to_skip = df_removal['Label_Name'].tolist()
    #df = df[~df["Label_Name"].isin(labels_to_skip)]
    #df.to_excel(save_base + "uw_label_wrong_side_removed_test.xlsx", index=False)


    df = pd.read_excel(save_base + "uw_label_wrong_side_removed_test.xlsx")
    print(f"lenth after loading: {len(df)}")
    df = remove_non_anatomical_sent_v2(df)
    print(f"length after dropping: {len(df)}")
    df.to_excel(save_base + "test_remove_non_anatomical_sent_v2.xlsx", index=False)
    sampled_df = df.sample(n=250, random_state=42)
    sampled_df.to_excel(save_base + "for_daniel_250_round_2.xlsx", index=False)
    #dataframe_to_json(df)

    #sampled_df = pd.read_excel(save_base + "cases_for_labeling_accuracy_accessment_250.xlsx")
    #sampled_df = df.sample(n=250, random_state=1)
    #sampled_df.to_excel(save_base + "cases_for_labeling_accuracy_accessment_250.xlsx", index=False)
    #print("before plotting")
    sampled_df = pd.read_excel(save_base + "for_daniel_250_round_2.xlsx")
    plot_for_label_accuracy_assessment(sampled_df)
    #print(df)
    #df = llm_remove_non_anatomical_sent(df)
    #print(df)
    #df.to_excel(save_base + "test_remove_non_anatomical_sent.xlsx", index=False)

    #resampling_and_cropping(df)

    #plot_mips_with_labels(df)
    #create_mips(df, load_location = "uw_labels_v2_nifti", image_path_name = "images_coronal_mip_uw_v2", label_path_name = "labels_coronal_mip_uw_v2")
    #create_mips(df, load_location = "labels_v13_nifti_test_3", image_path_name = "images_coronal_mip_v13", label_path_name = "labels_coronal_mip_v13")


def run_data_pipeline_final():
    save_base = "/UserData/Zach_Analysis/suv_slice_text/uw_all_pet_preprocess_chain_v4/"
    save_base_final = "/UserData/Zach_Analysis/petlymph_image_data/"
    """
    df_path = "/UserData/Zach_Analysis/lymphoma_data/all_pet_reports_uw.xlsx"
    df = pd.read_excel(df_path)
    df.to_excel(save_base + "initial_data_0.xlsx", index=False)

    df = split_sentences(df)
    df.to_excel(save_base + "sentences_split_1.xlsx", index=False)

    # split the sentences further or drop ones that have too many suv or slice values.
    df = detect_and_remove_multiple_suv_slice(df)
    df.to_excel(save_base + "remove_multiple_suv_and_slice_2.xlsx")
    
    #rint(fail) # rerun radgraph on previous suv and slice
    #df = pd.read_excel(save_base + "remove_multiple_suv_and_slice_2.xlsx")
    df = remove_non_anatomical_sent_v2(df)
    df.to_excel(save_base + "remove_non_anotomical_info_3.xlsx", index=False)

    #df = pd.read_excel("/UserData/Zach_Analysis/suv_slice_text/uw_all_pet_preprocess_chain_v3/remove_non_anotomical_info_3.xlsx")
    #df = llm_remove_multiple_descriptions(df)
    #df.to_excel(save_base + "llm_remove_multipled_descriptions_test_5_1000_test_v5.xlsx", index=False, sheet_name='Predictions')
    #df = pd.read_excel(save_base + "remove_non_anotomical_info_3_try3.xlsx")
    df = llm_slice_suv_extraction(df)
    df.to_excel(save_base + "model_predictions_for_suv_slice_extraction_4.xlsx", index=False, sheet_name='Predictions')

    #df = llm_remove_multiple_descriptions(df)
    #df.to_excel(save_base + "llm_remove_multipled_descriptions_test_5_250_test.xlsx", index=False, sheet_name='Predictions')
    #df = pd.read_excel(save_base + "model_predictions_for_suv_slice_extraction_4_1000_test.xlsx")
    # df = pd.read_excel(save_base + "model_predictions_for_suv_slice_extraction_2.xlsx")
    df = concenus_voting(df)
    df.to_excel(save_base + "concenus_output_5.xlsx", index=False)

    df = threshold_suv_max(df)
    df.to_excel(save_base + "thresholded_concensus_outputs_6.xlsx")
    
    df = pd.read_excel(save_base + "thresholded_concensus_outputs_6.xlsx")
    df = map_to_new_coded_id(df)
    df.to_excel(save_base + "mapped_to_new_id_7.xlsx")
    """
    #df = pd.read_excel(save_base + "max_pixel_7_all.xlsx")
    #df = pd.read_excel(save_base + "mapped_to_new_id_7.xlsx")
    #df = get_max_pixel_step3(df)
    #df.to_excel(save_base + "max_pixel_9_30000.xlsx", index=False)
    """
    df_1 = pd.read_excel(save_base + "max_pixel_9_10000.xlsx")
    df_2 = pd.read_excel(save_base + "max_pixel_9_20000.xlsx")
    df_3 = pd.read_excel(save_base + "max_pixel_9_30000.xlsx")

    df = pd.concat([df_1, df_2, df_3])
    df.to_excel(save_base + "max_pixel_9_all.xlsx", index=False)
    print(len(df))


    df = remove_duplicates(df)
    df = assign_label_numbers(df)
    print(f"length after labesl are assigned: {len(df)}")
    df.to_excel(save_base + "remove_duplicate_maxes_and_label_10.xlsx", index=False)

    df = make_labels_from_suv_max_points(df, save_location="uw_labels_v4_nifti")
    df.to_excel(save_base + "sentences_with_labels_df_11.xlsx", index=False)
    """
    #df = pd.read_excel(save_base + "sentences_with_labels_df_11.xlsx")
    #df = count_left_right_sided(df, "/mnt/Bradshaw/UW_PET_Data/raw_nifti_uw_pet/uw_labels_v4_nifti/")
    #df.to_excel(save_base + "wrong_side_need_to_drop_12.xlsx", index=False)

    """
    df = pd.read_excel(save_base + "sentences_with_labels_df_11.xlsx")
    # add check for suv that matches the suv noted
    df_to_drop = pd.read_excel(save_base + "uw_label_wrong_side_analysis.xlsx")

    # Extract the Label_Name values from df_to_drop to create a list
    labels_to_drop = df_to_drop['Label_Name'].unique()

    # Drop rows where 'Label_Name' is in the list extracted from df_to_drop
    df = df[~df['Label_Name'].isin(labels_to_drop)]
    df.to_excel(save_base + "uw_label_wrong_side_dropped_12.xlsx")

    df = check_max_pixel_and_slice(df)
    # add check for slice range of label matches noted slice

    df.to_excel(save_base + "removed_wrong_suv_max_and_slices_13.xlsx", index=False)
    """
    #plot_all_images()
    #plot_3d_predictions_single_image()
    #ct_analysis()
    #scanner_types()
    #post_processing_eval()
    #df = pd.read_excel(save_base + "removed_wrong_suv_max_and_slices_13.xlsx")
    #df_used_images = resampling_and_cropping(df)
    #df_used_images.to_excel(save_base + "id_pet_used_ct_used.xlsx", index=False)
    #make_json_file_for_3d_training(df)

    #sampled_df, orig_df = select_250_images_from_json()
    #sampled_df.to_excel(save_base + "/final_250_testset/for_daniel_250_round_3_final_testset_json.xlsx", index=False)
    #orig_df.to_excel(save_base + "/final_250_testset/for_daniel_250_round_3_final_testset_standard.xlsx", index=False)
    sampled_df = pd.read_excel(save_base + "/final_250_testset/for_daniel_250_round_3_final_testset_json.xlsx")

    #sampled_df = sampled_df[sampled_df["patient_id"] == '017530']
    print(sampled_df)
    #plot_final_testset_v2(sampled_df)
    #plot_final_testset_for_josh_v3(sampled_df)
    plot_final_testset(sampled_df)

    #df = df.drop_duplicates(subset="Petlymph", keep="first")
    #print(f"unique images: {len(df)}")
    #crop_at_head_calculation(df)


    #df = pd.read_excel(save_base + "for_daniel_250_round_3_final_testset_standard.xlsx")
    #df_metadata = resampling_and_cropping(df)
    #df_metadata.to_excel(save_base + "pet_ct_images_used_metadata.xlsx", index=False)
    #df = pd.read_excel(save_base + "for_daniel_250_round_3_final_testset_json.xlsx")
    #plot_final_testset(df)
    #df = llm_remove_multiple_descriptions(df)
    #df.to_excel(save_base + "remove_multiple_descriptions_test_llama3_v2_14.xlsx", index=False)
    #uw_ct_check()
    #entries = get_entries_to_remove()
    #print(entries)
    #plot_3d_predictions()