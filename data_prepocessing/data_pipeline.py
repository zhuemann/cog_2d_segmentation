#from data_prepocessing.split_sentences import split_sentences
#from data_prepocessing.llm_slice_suv_extraction import llm_slice_suv_extraction
#from data_prepocessing.concenus_voting import concenus_voting
#from data_prepocessing.get_max_pixel import get_max_pixel_step3
from data_prepocessing.llm_sentence_splitting import detect_and_remove_multiple_suv_slice
from data_prepocessing.remove_dups_non_anontomical_sent import remove_dups_non_anontomical_sent
from data_prepocessing.make_labels_from_point import make_labels_from_suv_max_points
#from data_prepocessing.create_sentence_mips_and_labels import create_mips
from data_prepocessing.remove_dups_non_anontomical_sent import get_anatomical_dataframe
from data_prepocessing.data_visualization.plot_mips_with_labels import plot_mips_with_labels
import pandas as pd
def run_data_pipeline():

    #save_base = "/UserData/Zach_Analysis/suv_slice_text/uw_lymphoma_preprocess_chain_v3/"
    save_base = "/UserData/Zach_Analysis/suv_slice_text/uw_lymphoma_preprocess_chain_v11/"

    save_base_final = "/UserData/Zach_Analysis/petlymph_image_data/"
    #df = split_sentences()
    #df.to_excel(save_base + "sentences_split_1.xlsx", index=False)
    #df = pd.read_excel(save_base + "sentences_split_1.xlsx")
    #df_radgraph = get_anatomical_dataframe(df)
    #print(df_radgraph)
    #print(fail)
    #df = llm_slice_suv_extraction(df)
    #df.to_excel(save_base + "model_predictions_for_suv_slice_extraction_2.xlsx", index=False, sheet_name='Predictions')

    #df = pd.read_excel(save_base + "model_predictions_for_suv_slice_extraction_2.xlsx")
    #df = concenus_voting(df)
    #df.to_excel(save_base + "concenus_output_3.xlsx", index=False)
    #df = pd.read_excel(save_base + "concenus_output_3.xlsx")
    #df = get_max_pixel_step3(df)
    #print(df)
    #df.to_excel(save_base + "max_pixel_4_test_rerun_slice_ref_fixed.xlsx", index=False)
    #df = pd.read_excel(save_base + "max_pixel_4_test_rerun_slice_ref_fixed.xlsx")
    #df = detect_and_remove_multiple_suv_slice(df)
    #print(len(df))
    #df.to_excel(save_base + "remove_multiple_suv_slice_5.xlsx", index=False)

    #df = remove_dups_non_anontomical_sent(df)
    #print(df)
    #df.to_excel(save_base + "remove_dups_df_6.xlsx", index=False)
    df = pd.read_excel(save_base + "remove_dups_df_6.xlsx")
    df = make_labels_from_suv_max_points(df, save_location = "labels_v13_nifti_test_2")
    #df.to_excel(save_base_final + "dropped_problem_segs_6_v5.xlsx", index=False)
    #print(fail)

    #df = pd.read_excel(save_base_final + "dropped_problem_segs_6_v5.xlsx")

    plot_mips_with_labels(df)
    #create_mips(df, load_location = "labels_v12_nifti", image_path_name = "images_coronal_mip_v12", label_path_name = "labels_coronal_mip_v12")