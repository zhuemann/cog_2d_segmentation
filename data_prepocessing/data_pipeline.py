
from data_prepocessing.split_sentences import split_sentences
from data_prepocessing.llm_slice_suv_extraction import llm_slice_suv_extraction
from data_prepocessing.concenus_voting import concenus_voting
import pandas as pd
def run_data_pipeline():

    #df = split_sentences()
    #print(df)
    #print(fail)
    #df = llm_slice_suv_extraction(df)
    #print(df)
    #df.to_excel("test.xlsx", index=False, sheet_name='Predictions')
    df = pd.read_excel("test.xlsx")
    df = concenus_voting(df)
    print(df)