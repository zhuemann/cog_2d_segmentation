

import pandas as pd

def map_to_new_coded_id(df):
    df_coded = pd.read_excel("/UserData/UW_PET_Data/UWPETCTWB_Research-Image-Requests_20240311.xlsx")
    print(df_coded.columns)
    # Create a mapping from Original to Coded Accession Numbers
    mapping = df_coded.set_index('Original Accession Number')['Coded Accession Number'].to_dict()
    # Replace the Original Accession Numbers with Coded Accession Numbers in df_original
    df['Petlymph'] = df['Petlymph'].map(mapping)

    return df