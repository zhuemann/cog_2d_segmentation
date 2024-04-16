
from data_prepocessing.split_sentences import split_sentences
from data_prepocessing.llm_slice_suv_extraction import llm_slice_suv_extraction
from data_prepocessing.concenus_voting import concenus_voting
from data_prepocessing.get_max_pixel import get_max_pixel_step3
from data_prepocessing.remove_dups_non_anontomical_sent import remove_dups_non_anontomical_sent
from data_prepocessing.make_labels_from_point import make_labels_from_suv_max_points
from data_prepocessing.create_sentence_mips_and_labels import create_mips
from data_prepocessing.data_visualization.plot_mips_with_labels import plot_mips_with_labels
import pandas as pd
def run_data_pipeline():

    save_base = "/UserData/Zach_Analysis/suv_slice_text/uw_lymphoma_preprocess_chain_v3/"

    #df = split_sentences()
    #df.to_excel(save_base + "sentences_split_1.xlsx", index=False)

    #df = llm_slice_suv_extraction(df)
    #df.to_excel(save_base + "model_predictions_for_suv_slice_extraction_2.xlsx", index=False, sheet_name='Predictions')

    #df = pd.read_excel(save_base + "model_predictions_for_suv_slice_extraction_2.xlsx")
    #df = concenus_voting(df)
    #df.to_excel(save_base + "concenus_output_3.xlsx", index=False)
    #df = pd.read_excel(save_base + "concenus_output_3.xlsx")
    #df = get_max_pixel_step3(df)
    #print(df)
    #df.to_excel(save_base + "max_pixel_4_test_run_all.xlsx", index=False)
    df = pd.read_excel(save_base + "max_pixel_4_test_run_all.xlsx")
    df = remove_dups_non_anontomical_sent(df)
    df.to_excel(save_base + "remove_dups_df_5.xlsx", index=False)
    #make_labels_from_suv_max_points(df)
    #plot_mips_with_labels(df)
    create_mips(df)