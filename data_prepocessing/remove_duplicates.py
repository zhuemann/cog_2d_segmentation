

def remove_duplicates(df):


    df_filtered = df.drop_duplicates(subset=['Petlymph', 'Extracted Sentences'], keep=False)

    return df_filtered

def assign_label_numbers(df_filtered):

    # Create an enumeration column for each label per "Petlymph" ID
    df_filtered['Label_Number'] = df_filtered.groupby('Petlymph').cumcount() + 1

    # Create the new column by combining "Petlymph" with the label enumeration
    df_filtered['Label_Name'] = df_filtered['Petlymph'] + '_label_' + df_filtered['Label_Number'].astype(str)

    # Drop the temporary 'Label_Number' column if you don't need it
    df_filtered.drop('Label_Number', axis=1, inplace=True)
    return df_filtered