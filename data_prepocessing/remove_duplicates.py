

def remove_duplicates(df):


    df_filtered = df.drop_duplicates(subset=['Petlymph', 'Extracted Sentences'], keep=False)

    return df_filtered