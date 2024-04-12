
from data_prepocessing.split_sentences import split_sentences

def run_data_pipeline():

    df = split_sentences()
    print(df)
    df.to_excel("test.xlsx")