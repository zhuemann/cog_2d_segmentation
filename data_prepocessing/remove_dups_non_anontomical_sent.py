import pandas as pd


def remove_dups_non_anontomical_sent(df):

    # throw out the sentence if multiple label points were extracted
    df_filtered = df.drop_duplicates(subset=['Petlymph', 'Extracted Sentences'], keep=False)
    print(len(df_filtered))

    # Create an enumeration column for each label per "Petlymph" ID
    df_filtered['Label_Number'] = df_filtered.groupby('Petlymph').cumcount() + 1

    # Create the new column by combining "Petlymph" with the label enumeration
    df_filtered['Label_Name'] = df_filtered['Petlymph'] + '_label_' + df_filtered['Label_Number'].astype(str)

    # Drop the temporary 'Label_Number' column if you don't need it
    df_filtered.drop('Label_Number', axis=1, inplace=True)

    df_path = "Z:/Zach_Analysis/suv_slice_text/uw_pet_lymphoma_next_and_previous_sentence_annotated.xlsx"
    df_anotomical_info = pd.read_excel(df_path)

    # Merge the two DataFrames on 'Petlymph' and 'Extracted sentences'
    merged_df = pd.merge(df_filtered, df_anotomical_info, on=['Extracted Sentences'])
    filtered_df = merged_df[merged_df['anatomy_available'] != 0]
    df_dropped = filtered_df.drop(
    columns=['Accession Number', 'Report', 'Impression_y', 'Indication', 'Slice_y', 'SUV_y', 'Previous Sentence',
                 'Following Sentence', 'annotation', 'anatomy', 'anatomy_available'])
    df_dropped.rename(columns={'Slice_x': 'Slice'}, inplace=True)
    df_dropped.rename(columns={'SUV_x': 'SUV'}, inplace=True)

    return df_dropped